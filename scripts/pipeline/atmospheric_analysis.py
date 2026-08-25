# ./scripts/pipeline/atmospheric_analysis.py
#
# Analisis "data mining" atmosfer -- BUKAN model ML, murni analisis
# statistik/domain di atas deret CTT (tbb_13) mentah, untuk menjawab
# pertanyaan fisis: apakah pola yang ditangkap pipeline ML konsisten
# dengan perilaku awan konvektif tropis yang sudah dikenal di literatur,
# dan apakah pilihan desain pipeline (window, target channel) punya
# justifikasi statistik selain trial-and-error MAE.
#
# LATAR BELAKANG FISIS SINGKAT:
# CTT (Cloud Top Temperature) dari kanal IR window Himawari (tbb_13,
# ~10.4 micron) adalah proxy suhu puncak awan. Awan konvektif dalam
# (cumulonimbus, indikator badai/hujan lebat) punya puncak sangat tinggi
# -> CTT SANGAT RENDAH (bisa <220K / -53 derajat Celsius). Config sudah
# punya TBB_RISK_THRESHOLDS = (200.0, 270.0) K yang dipakai untuk
# visualisasi (08_visualize.py) -- modul ini REUSE threshold yang SAMA
# supaya klasifikasi konsisten di seluruh pipeline, bukan bikin threshold
# baru sendiri:
#   CTT < 200K            : sangat ekstrem / kemungkinan noise sensor
#   200K <= CTT < 270K     : awan tinggi/konvektif -- risiko cuaca buruk
#   CTT >= 270K            : langit cerah / awan rendah
#
# Bandung berada di WIB (UTC+7); siklus konveksi tropis di dataran tinggi
# umumnya puncak di sore-malam hari (pemanasan permukaan siang -> udara
# naik -> konveksi sore) -- modul ini memverifikasi pola itu dari data
# aktual, bukan asumsi.

import numpy as np
import pandas as pd
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

from pipeline.time_features import WIB_OFFSET_HOURS


TBB_RISK_LOW, TBB_RISK_HIGH = 200.0, 270.0


def diurnal_cycle_profile(data_matrix, timeline):
    """
    Rata-rata & std CTT per jam LOKAL (WIB), digabung SEMUA pixel & SEMUA
    hari -- mengungkap siklus diurnal konveksi. Dipakai untuk memverifikasi
    rasionalitas fitur hour_sin/hour_cos di pipeline/time_features.py
    (kalau siklus diurnal tidak jelas/flat, fitur waktu itu tidak akan
    banyak membantu model -- ini justifikasi empiris untuk desain fitur
    yang sudah dipilih).

    Returns
    -------
    pd.DataFrame, kolom: hour_wib, mean_ctt, std_ctt, n_obs
    """
    ts_wib = pd.DatetimeIndex(
        np.asarray(timeline.values, dtype="datetime64[ns]")
    ) + pd.Timedelta(hours=WIB_OFFSET_HOURS)

    hours = ts_wib.hour.to_numpy()
    # rata-rata antar pixel per timestep dulu (satu nilai representatif
    # per timestep), baru dikelompokkan per jam -- supaya pixel dengan
    # data lebih rapat/rapuh tidak mendominasi rata-rata jam tertentu.
    per_timestep_mean = np.nanmean(data_matrix, axis=1)

    df = pd.DataFrame({"hour_wib": hours, "ctt": per_timestep_mean})
    profile = df.groupby("hour_wib")["ctt"].agg(
        mean_ctt="mean", std_ctt="std", n_obs="count"
    ).reset_index()
    return profile.sort_values("hour_wib").reset_index(drop=True)


def classify_convective_risk(data_matrix):
    """
    Klasifikasi setiap nilai CTT ke 3 kelas risiko memakai
    TBB_RISK_LOW/HIGH (SAMA dengan threshold visualisasi di
    Config.TBB_RISK_THRESHOLDS) -- lihat header modul ini.

    Returns
    -------
    dict: proporsi (0..1) tiap kelas ATAS SELURUH data_matrix (semua
    pixel x semua waktu, NaN diabaikan).
    """
    flat = data_matrix[~np.isnan(data_matrix)]
    n = len(flat)
    if n == 0:
        return {"deep_convective": np.nan, "moderate_cloud": np.nan, "clear": np.nan, "n_obs": 0}

    deep = np.sum(flat < TBB_RISK_LOW) / n
    moderate = np.sum((flat >= TBB_RISK_LOW) & (flat < TBB_RISK_HIGH)) / n
    clear = np.sum(flat >= TBB_RISK_HIGH) / n

    return {
        "deep_convective": float(deep),
        "moderate_cloud": float(moderate),
        "clear": float(clear),
        "n_obs": int(n),
    }


def detect_convective_onset_events(y, timeline, drop_threshold_k=15.0, window_steps=3):
    """
    Deteksi "onset konvektif" pada SATU pixel: titik waktu di mana CTT
    turun >= drop_threshold_k dalam window_steps langkah (default 15K
    dalam 3 x 10 menit = 30 menit -- ambang dipilih longgar, bisa
    disesuaikan; tujuannya menandai penurunan CEPAT yang mengindikasikan
    pertumbuhan awan konvektif, bukan fluktuasi noise biasa).

    Ini murni deteksi berbasis rate-of-change (dCTT/dt), BUKAN bagian dari
    model ML manapun di pipeline -- dipakai untuk mengkarakterisasi
    seberapa SERING & seberapa CEPAT event seperti ini terjadi di data,
    yang relevan untuk menjelaskan kenapa forecasting CTT sulit di
    horizon panjang (event onset konvektif secara definisi mendadak/sulit
    diprediksi dari tren linear riwayat lag).

    Returns
    -------
    pd.DataFrame, kolom: onset_idx, onset_time, ctt_before, ctt_after,
        drop_k
    """
    T = len(y)
    if T <= window_steps:
        return pd.DataFrame(columns=["onset_idx", "onset_time", "ctt_before", "ctt_after", "drop_k"])

    diffs = y[window_steps:] - y[:-window_steps]
    onset_mask = diffs <= -drop_threshold_k

    rows = []
    for i in np.where(onset_mask)[0]:
        if np.isnan(diffs[i]):
            continue
        rows.append({
            "onset_idx": i + window_steps,
            "onset_time": timeline[i + window_steps],
            "ctt_before": y[i],
            "ctt_after": y[i + window_steps],
            "drop_k": float(-diffs[i]),
        })

    return pd.DataFrame(rows)


def spatial_cluster_pixels(data_matrix, pixel_meta, n_clusters=4):
    """
    Hierarchical clustering pixel berdasarkan KORELASI antar deret waktu
    CTT-nya (1 - korelasi = jarak) -- mengelompokkan pixel yang
    "berperilaku sama" secara temporal. Relevan untuk pertanyaan:
    apakah 35 pixel grid Bandung ini homogen (satu model global masuk
    akal) atau ada sub-region dengan pola berbeda (yang bisa jadi alasan
    tambahan kenapa spatial_correlation kolaps saat rollout -- lihat
    catatan di window_eval.py -- karena satu model dipaksa menggeneralisasi
    lintas pixel yang sebenarnya heterogen).

    Returns
    -------
    pd.DataFrame, kolom: pixel_id, latitude, longitude, cluster_id
    corr_matrix : np.ndarray, shape (P, P) -- untuk divisualisasikan
        sebagai heatmap.
    """
    P = data_matrix.shape[1]
    valid_cols = ~np.all(np.isnan(data_matrix), axis=0)

    corr_matrix = np.full((P, P), np.nan)
    if valid_cols.sum() >= 2:
        sub = data_matrix[:, valid_cols]
        df = pd.DataFrame(sub)
        corr_sub = df.corr().values
        idx = np.where(valid_cols)[0]
        corr_matrix[np.ix_(idx, idx)] = corr_sub

    # jarak = 1 - korelasi, clamp ke [0, 2] & isi NaN (pixel tak valid)
    # dengan jarak maksimum supaya tidak crash linkage().
    dist = 1.0 - np.nan_to_num(corr_matrix, nan=-1.0)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2.0  # pastikan simetris sempurna (floating point)

    condensed = squareform(dist, checks=False)
    Z = linkage(condensed, method="average")
    cluster_ids = fcluster(Z, t=n_clusters, criterion="maxclust")

    result = pd.DataFrame({
        "pixel_id": pixel_meta["pixel_id"],
        "latitude": pixel_meta["latitude"],
        "longitude": pixel_meta["longitude"],
        "cluster_id": cluster_ids,
    })

    return result, corr_matrix


def stationarity_test(y, max_lag=None):
    """
    Augmented Dickey-Fuller test (via statsmodels kalau tersedia, fallback
    ke pesan error yang jelas kalau tidak) -- menguji apakah deret CTT
    satu pixel STASIONER (rata-rata & varians konstan sepanjang waktu).

    Relevan secara metodologis: model lag-based (window_features.py)
    IMPLISIT mengasumsikan struktur deret cukup stabil supaya pola
    lag->target yang dipelajari dari TRAIN (Des'25-Mei'26) tetap berlaku
    di TEST (Jun-Jul'26). Non-stasioneritas kuat (mis. tren musiman
    panjang) adalah argumen mengapa MAE bisa naik di TEST dibanding
    VALIDATION walau modelnya sama.

    Returns
    -------
    dict: adf_stat, p_value, is_stationary_5pct (bool), n_obs_used
    """
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError as exc:
        raise ImportError(
            "statsmodels dibutuhkan untuk stationarity_test() "
            "(pip install statsmodels)."
        ) from exc

    y_clean = y[~np.isnan(y)]
    if len(y_clean) < 20:
        return {"adf_stat": np.nan, "p_value": np.nan, "is_stationary_5pct": None, "n_obs_used": len(y_clean)}

    result = adfuller(y_clean, maxlag=max_lag, autolag="AIC")
    adf_stat, p_value = result[0], result[1]

    return {
        "adf_stat": float(adf_stat),
        "p_value": float(p_value),
        "is_stationary_5pct": bool(p_value < 0.05),
        "n_obs_used": len(y_clean),
    }


def acf_pacf(y, n_lags=24):
    """
    Autocorrelation (ACF) & partial autocorrelation (PACF) dari SATU
    deret pixel, sampai n_lags step. Dipakai untuk justifikasi statistik
    Config.WINDOW_CANDIDATES: PACF yang meluruh cepat setelah lag ke-k
    mengindikasikan window pendek sudah cukup menangkap dependency utama;
    kalau PACF masih signifikan sampai lag jauh, window besar (mis. 8)
    lebih beralasan dipilih -- silang-cek terhadap hasil window search
    empiris (04_search_window.py) yang murni berbasis MAE rollout.

    Returns
    -------
    pd.DataFrame, kolom: lag, acf, pacf
    """
    try:
        from statsmodels.tsa.stattools import acf, pacf as pacf_fn
    except ImportError as exc:
        raise ImportError(
            "statsmodels dibutuhkan untuk acf_pacf() (pip install statsmodels)."
        ) from exc

    y_clean = y[~np.isnan(y)]
    n_lags = min(n_lags, len(y_clean) - 1)

    acf_vals = acf(y_clean, nlags=n_lags, fft=True)
    pacf_vals = pacf_fn(y_clean, nlags=n_lags)

    return pd.DataFrame({
        "lag": np.arange(n_lags + 1),
        "acf": acf_vals,
        "pacf": pacf_vals,
    })


def distribution_summary(data_matrix):
    """
    Statistik deskriptif distribusi CTT (semua pixel x semua waktu):
    mean, std, skewness, kurtosis. Awan konvektif tropis biasanya
    menghasilkan distribusi CTT SKEWED NEGATIF (ekor panjang ke suhu
    rendah -- kejadian awan tinggi jarang tapi ekstrem) -- dicek di sini
    secara langsung dari data, bukan diasumsikan.

    Returns
    -------
    dict
    """
    flat = data_matrix[~np.isnan(data_matrix)]
    return {
        "mean": float(np.mean(flat)),
        "std": float(np.std(flat)),
        "skewness": float(stats.skew(flat)),
        "kurtosis": float(stats.kurtosis(flat)),
        "min": float(np.min(flat)),
        "max": float(np.max(flat)),
        "n_obs": int(len(flat)),
    }