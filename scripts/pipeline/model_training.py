# ./scripts/pipeline/model_training.py
# Menyediakan fungsi-fungsi untuk memuat dataset expanding window, membagi
# data secara stratified per-bulan, melatih model (XGBoost/LightGBM/
# CatBoost), dan evaluasi dasar (MAE/RMSE/R2 keseluruhan, BUKAN per-step --
# evaluasi per-step ada di recursive_eval.py, Tahap 04, lihat CLAUDE.md §7).
#
# stratified_monthly_split() di bawah ini REUSE VERBATIM dari repo lama
# (nugrahsdhka/Bandung-Weather-Forecast-Himawari-09, scripts/pipeline/
# model_training.py) -- logikanya generic (parameterized time_col), jadi
# tidak ada modifikasi, cuma dipanggil dengan time_col="anchor_t0" (bukan
# "base_time" seperti di repo lama). Lihat CLAUDE.md §6 kenapa dipakai
# dari awal (bukan mulai dari chronological_split() biasa).

import time

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.expanding_features import FEATURE_COLUMNS, compute_window_features_matrix
from ui.terminal_display import make_progress_bar, say_info, say_ok, say_error

TARGET_COLUMN = f"target_{Config.TARGET_CHANNEL}"
MIN_WINDOW_SIZE = Config.MIN_WINDOW_SIZE

# Purge/embargo default untuk stratified_monthly_split() (lihat docstring
# fungsi itu + Config.PURGE_STEPS di config.py untuk alasan lengkap --
# investigasi temporal leakage sesi ini: window training dekat cutoff bisa
# menyentuh raw observasi yang sama dengan target test).
PURGE_STEPS = Config.PURGE_STEPS

# Fitur yang dipakai buat rekonstruksi window mentah dari cache raw waktu
# noise injection (lihat inject_recursive_style_noise()). anchor_t0 +
# pixel_id -> posisi start window di cache; n_points -> panjang window
# (udah salah satu dari FEATURE_COLUMNS, dibuat pas dataset_builder.py).
_NOISE_JOIN_COLUMNS = ["pixel_id", "anchor_t0", "n_points"]


def load_expanding_dataset(path):
    """Baca dataset expanding window hasil dataset_builder.py, pastikan
    anchor_t0/target_time bertipe datetime."""
    df = pd.read_csv(path)
    df["anchor_t0"] = pd.to_datetime(df["anchor_t0"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    return df


def stratified_monthly_split(df, test_frac=None, time_col="anchor_t0", purge_steps=None):
    """
    Split train/test PER-BULAN: di dalam tiap bulan, ambil test_frac fraksi
    TERAKHIR secara kronologis sebagai test -- bukan ekor kronologis dari
    SELURUH dataset seperti chronological split biasa.

    ASALNYA reuse verbatim dari repo lama (lihat header file), logikanya
    generic (parameterized time_col). Kenapa dibutuhkan: kalau rentang data
    mencakup banyak bulan, ekor kronologis dari seluruh dataset bisa membuat
    beberapa bulan (terutama musim konvektif) 0% representasi di test --
    model tidak pernah dievaluasi untuk kondisi bulan itu walau datanya
    penuh di training. Ini persis masalah yang ditemukan di repo lama dan
    sengaja dihindari dari awal di project ini (CLAUDE.md §6).

    Tetap kronologis DI DALAM tiap grup bulan (bukan random) supaya tidak
    ada kebocoran temporal dalam bulan yang sama -- cuma unit
    stratifikasinya yang berubah (per-bulan, bukan seluruh dataset).

    PURGE/EMBARGO (ditambahkan sesi ini, TIDAK ADA di versi repo lama --
    lihat investigasi temporal leakage): karena window expanding untuk step
    besar bisa menjangkau sampai ANCHOR_SPAN-1 titik (230 menit dgn default
    MIN_WINDOW_SIZE=6/HORIZON_STEPS=18) ke DEPAN dari `anchor_t0`-nya
    sendiri, anchor train yang `anchor_t0`-nya terlalu dekat ke `cutoff_time`
    bisa punya window/target yang menyentuh raw observasi yang secara waktu
    sudah masuk wilayah test -- bahkan bisa jadi nilai yang PERSIS SAMA
    dengan target salah satu test row (temporal leakage, dibuktikan lewat
    simulasi terhadap kode ini sendiri).

    Purge DUA SISI, diterapkan LINTAS BULAN (bukan per grup bulan yang
    independen):
    1. SISI SEBELUM cutoff: anchor train dgn `anchor_t0` kurang dari
       `purge_steps * Config.FREQ_MINUTES` menit sebelum `cutoff_time`
       bulan itu -- ini kasus "biasa" (window training menjangkau maju ke
       wilayah test bulan yang sama).
    2. SISI SETELAH anchor test terakhir suatu bulan: ditemukan lewat
       validasi (validate_no_leakage.py) bahwa anchor test PALING AKHIR di
       suatu bulan (mis. akhir Januari) bisa punya window/target yang
       menjorok ke bulan BERIKUTNYA (mis. awal Februari) -- karena
       `df.groupby("__bulan")` memproses tiap bulan independen, anchor awal
       Februari yang notabene train (jauh dari cutoff Februari SENDIRI)
       ternyata bisa berbagi raw value dengan target test Januari yang
       menjorok itu. Fix: anchor mana pun (bulan apa pun) dgn `anchor_t0`
       dalam `purge_steps * Config.FREQ_MINUTES` menit SETELAH anchor test
       terakhir bulan sebelumnya juga di-purge. Tanpa sisi ini, purge cuma
       menutup separuh mekanisme leakage yang terbukti ada.

    `cutoff_time` & definisi TEST **TIDAK BERUBAH** oleh purge ini (tetap
    persis `anchor_t0 >= cutoff_time`, dihitung dari `test_frac` yang sama
    seperti sebelumnya) -- purge HANYA mengurangi train, supaya
    `select_test_anchors()` (recursive_eval.py, yang manggil fungsi INI
    dengan parameter default yang sama) otomatis tetap menghasilkan anchor
    test yang identik dengan sebelum purge ditambahkan.

    PRECONDITION (implisit, TIDAK di-assert di sini -- lihat catatan audit):
    perhitungan interval purge memakai batas tertutup `cutoff_time -
    FREQ_MINUTES` sebagai pengganti batas terbuka `< cutoff_time` (begitu
    juga sisi setelah). Ini HANYA benar kalau seluruh nilai `time_col` di
    `df` grid-aligned -- kelipatan `Config.FREQ_MINUTES` dari origin yang
    SAMA (mis. semuanya kelipatan 10 menit dari titik nol yang sama).
    Selalu benar untuk `df` yang berasal dari `dataset_builder.build_dataset()`
    (timeline dibangun via `pd.date_range(freq=...)`, sudah diverifikasi
    lewat `validate_no_leakage.py` termasuk skenario gap/sparse-month) --
    TIDAK dijamin kalau fungsi ini suatu saat dipanggil dengan `df` dari
    sumber lain yang `time_col`-nya tidak seragam grid-nya.

    Parameters
    ----------
    test_frac : float, optional. Default None -> pakai Config.TEST_FRAC (0.15).
    time_col : str, default "anchor_t0" (beda dari repo lama yang pakai
        "base_time", karena skema kolom expanding window beda -- lihat
        dataset_builder.py). WAJIB grid-aligned per `Config.FREQ_MINUTES`
        (lihat PRECONDITION di atas).
    purge_steps : int, optional. Default None -> pakai Config.PURGE_STEPS
        (= Config.ANCHOR_SPAN - 1, batas konservatif -- lihat penjelasan di
        atas & Config.PURGE_STEPS di config.py). purge_steps=0 mengembalikan
        perilaku LAMA (tanpa purge sama sekali) -- disediakan untuk keperluan
        perbandingan/debugging, BUKAN default produksi.

    Return
    ------
    (train_df, test_df, cutoffs) -- `cutoffs` adalah dict
    {str(bulan): cutoff_time}. `cutoff_time` tetap batas test seperti
    sebelumnya (BUKAN batas purge) -- dipertahankan supaya caller yang sudah
    ada (mis. select_test_anchors) tidak perlu berubah.
    """
    if test_frac is None:
        test_frac = Config.TEST_FRAC
    if purge_steps is None:
        purge_steps = PURGE_STEPS

    df = df.copy()
    df["__bulan"] = df[time_col].dt.to_period("M")
    purge_delta = pd.Timedelta(minutes=purge_steps * Config.FREQ_MINUTES)

    # --- Pass 1: tentukan cutoff_time & zona test PER BULAN dulu. Zona
    # purge dikumpulkan sebagai interval waktu (bukan langsung difilter di
    # sini), karena zona purge "sisi setelah" milik bulan m HARUS diterapkan
    # ke baris bulan m+1 juga -- filtering per-grup independen (loop lama)
    # tidak bisa menjangkau baris di grup lain.
    cutoffs = {}
    purge_intervals = []  # list of (start, end), interval TERTUTUP [start, end] yang di-drop dari train
    is_test = pd.Series(False, index=df.index)

    for bulan, group in df.groupby("__bulan"):
        unique_times = sorted(group[time_col].unique())
        cutoff_idx = int(len(unique_times) * (1 - test_frac))
        # Guard bulan dengan timestamp sedikit: pastikan minimal ada 1 test
        # (kalau test_frac > 0) dan tidak keluar dari rentang list.
        cutoff_idx = min(max(cutoff_idx, 0), len(unique_times) - 1)
        cutoff_time = unique_times[cutoff_idx]
        cutoffs[str(bulan)] = cutoff_time

        test_mask_bulan = group[time_col] >= cutoff_time
        is_test.loc[group.index[test_mask_bulan]] = True
        last_test_time = group.loc[test_mask_bulan, time_col].max()

        # Sisi sebelum cutoff (zona purge: [cutoff_time - purge_delta, cutoff_time)).
        purge_intervals.append((cutoff_time - purge_delta, cutoff_time - pd.Timedelta(minutes=Config.FREQ_MINUTES)))
        # Sisi setelah anchor test terakhir bulan ini (zona purge:
        # (last_test_time, last_test_time + purge_delta] -- bisa menjorok ke
        # bulan berikutnya, makanya diterapkan global di Pass 2, bukan cuma
        # ke `group` bulan ini).
        purge_intervals.append((last_test_time + pd.Timedelta(minutes=Config.FREQ_MINUTES), last_test_time + purge_delta))

    # --- Pass 2: TRAIN = baris yang BUKAN test DAN tidak jatuh di interval
    # purge manapun (dicek lintas SELURUH df, bukan per grup bulan). ---
    is_purged = pd.Series(False, index=df.index)
    for start, end in purge_intervals:
        if start > end:
            continue  # interval kosong (mis. purge_steps=0), lewati
        is_purged |= (df[time_col] >= start) & (df[time_col] <= end)

    train_df = (
        df[(~is_test) & (~is_purged)]
        .drop(columns="__bulan").sort_values(time_col).reset_index(drop=True)
    )
    test_df = (
        df[is_test]
        .drop(columns="__bulan").sort_values(time_col).reset_index(drop=True)
    )
    return train_df, test_df, cutoffs


def get_feature_target(df, feature_columns=None):
    """Ambil X (9 kolom fitur closed-form, lihat expanding_features.FEATURE_COLUMNS)
    dan y (target_tbb_13) dari dataframe."""
    X = df[feature_columns or FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


def build_step_noise_profile(summary_path, scale=1.0):
    """Bangun profil noise_std PER KEDALAMAN ROLLOUT dari
    `recursive_mae_summary.csv` yang SUDAH ADA (biasanya hasil evaluasi
    model produksi saat ini) -- rata-rata MAE lintas model di tiap step,
    dipakai sbg magnitude noise KHUSUS titik window yang mensimulasikan
    kedalaman step itu (lihat `inject_recursive_style_noise()` parameter
    `step_noise_profile`).

    KENAPA: `noise_std` konstan (dipakai default) menyuntik magnitude yang
    SAMA ke semua titik "dianggap prediksi", padahal error rollout asli
    membesar seiring kedalaman (MAE step 2 ~4K, step 18 ~13K) -- window
    training yang mensimulasikan kedalaman dangkal jadi OVER-noised
    relatif ke error asli-nya, window yang mensimulasikan kedalaman dalam
    jadi UNDER-noised. Profil ini bikin magnitude noise per titik
    mengikuti kurva error yang BENERAN terukur, bukan angka konstan.

    scale : float, default 1.0
        Faktor pengali seluruh profil (SEBELUM dipakai inject) -- ditambah
        setelah eksperimen pertama (scale=1.0, profil mentah) terbukti
        signifikan membaikin spatial_collapse_ratio/correlation TAPI
        nge-regresi MAE step 1-4 (window "dangkal" ternoise proporsi besar
        dari dataset training, geser fit model). `scale<1.0` buat cari
        titik tengah -- lihat scripts/tools/sweep_step_noise_scale.py.

    Return
    ------
    pd.Series, index=step (int, 1..HORIZON_STEPS-1 cukup -- step
    HORIZON_STEPS sendiri nggak pernah jadi bagian window manapun, cuma
    target), value=mae rata-rata lintas model di step itu, dikali `scale`.
    """
    df = pd.read_csv(summary_path)
    return df.groupby("step")["mae"].mean() * scale


def inject_recursive_style_noise(train_df, cache_path, noise_std, random_state=42, step_noise_profile=None):
    """Redesain noise injection buat skema fitur closed-form project ini
    (BUKAN port verbatim dari `inject_lag_noise()` repo lama -- lihat
    CLAUDE.md §17 poin "Belum dikerjakan" versi sebelumnya, skema fitur di
    situ raw lag columns (tbb_13_t/tm1/tm2), di sini 9 fitur AGREGAT window
    (mean/std/min/max/first/last/delta/slope/n_points) -- noise independen
    per kolom bisa bikin kombinasi fisik nggak konsisten, mis. min_window >
    max_window, atau std_window nggak sinkron sama mean/slope-nya).

    KENAPA & DESAIN: root cause exposure bias (CLAUDE.md §17 poin 4) adalah
    saat recursive rollout, window makin lama makin isi PREDIKSI model
    sendiri (bukan observasi real) mulai step ke-2 (titik ke MIN_WINDOW_SIZE
    dst di window) -- fitur dihitung dari window "kotor" itu, model belum
    pernah lihat kombinasi fitur seperti itu pas training (window training
    selalu 100% observasi real). Fix: SIMULASIKAN kondisi itu pas training --
    untuk tiap baris training (anchor_t0, pixel_id, n_points sudah nentuin
    window persis via cache raw, sama seperti recursive_eval.py), tambahkan
    noise Gaussian HANYA ke titik-titik window yang posisinya > MIN_WINDOW_SIZE
    dari awal (posisi yang di kondisi recursive sungguhan bakal diisi
    prediksi, bukan observasi -- titik 0..MIN_WINDOW_SIZE-1 SELALU real, sama
    kayak IS1 di recursive_eval.py), lalu HITUNG ULANG semua 9 fitur dari
    window yang sudah dinoise pakai compute_window_features_matrix() (fungsi
    closed-form yang SAMA dipakai recursive_eval.py) -- ini otomatis jaga
    konsistensi fisik antar fitur (min<=mean<=max, std/slope ikut berubah
    proporsional), BUKAN noise 5 kolom independen yang bisa saling
    kontradiksi.

    Baris dengan n_points == MIN_WINDOW_SIZE (step 1, window 100% real) TIDAK
    disentuh sama sekali (no-op, dilewati dari grouping) -- konsisten dengan
    fakta recursive rollout beneran juga selalu punya window 100% real di
    step 1.

    Parameters
    ----------
    train_df : pd.DataFrame
        HARUS masih punya kolom pixel_id/anchor_t0/n_points (jangan panggil
        get_feature_target() dulu sebelum ini -- kolom itu dibuang di sana).
    cache_path : str
        Path cache raw .npz (Config.EXPANDING_RAW_CACHE_FILE), SAMA yang
        dipakai 04_recursive_evaluate.py -- dibutuhkan buat rekonstruksi
        window mentah (dataset CSV cuma nyimpen fitur agregat, bukan raw).
    noise_std : float
        Std deviasi noise Gaussian KONSTAN (satuan Kelvin) yang ditambahkan
        per TITIK di bagian window yang "dianggap prediksi" -- dipakai
        HANYA kalau `step_noise_profile` TIDAK diisi (None). noise_std <= 0
        DAN step_noise_profile None -> no-op, return train_df apa adanya
        (backward-compat). Idealnya mulai dari ~seukuran MAE step-1 aktual,
        WAJIB divalidasi lewat sweep (`scripts/tools/sweep_noise_std.py`).
    step_noise_profile : pd.Series, optional (default None)
        Kalau diisi, magnitude noise jadi PER KEDALAMAN ROLLOUT (bukan
        konstan) -- index=step (1..HORIZON_STEPS-1), value=std (Kelvin) yg
        dipakai utk titik window yang mensimulasikan kedalaman step itu
        (lihat `build_step_noise_profile()`). `noise_std` DIABAIKAN kalau
        parameter ini diisi. WAJIB mencakup semua step 1..HORIZON_STEPS-1
        yang mungkin muncul di data -- KeyError eksplisit kalau ada step
        yang kepakai tapi nggak ada di profile (bukan silent fallback ke
        NaN/0, yang bisa diam-diam merusak training).

    Return
    ------
    pd.DataFrame, sama seperti train_df tapi kolom FEATURE_COLUMNS sudah
    diganti versi noised (kolom lain, termasuk target, TIDAK berubah).
    """
    if step_noise_profile is None and noise_std <= 0:
        return train_df

    from pipeline.dataset_builder import load_raw_cache

    missing_cols = [c for c in _NOISE_JOIN_COLUMNS if c not in train_df.columns]
    if missing_cols:
        raise KeyError(
            f"train_df kehilangan kolom {missing_cols} yang dibutuhkan buat noise "
            "injection -- panggil inject_recursive_style_noise() SEBELUM "
            "get_feature_target() (yang buang kolom pixel_id/anchor_t0/n_points)."
        )

    if step_noise_profile is not None:
        say_info(
            f"Noise injection: profil PER-STEP (step 1..{int(step_noise_profile.index.max())}, "
            f"std {step_noise_profile.min():.2f}K-{step_noise_profile.max():.2f}K) pada window > step 1"
        )
    else:
        say_info(f"Noise injection: std={noise_std}K KONSTAN pada window > step 1 (rekonstruksi dari cache)")
    data_matrix, timeline, pixel_meta = load_raw_cache(cache_path)
    timeline_index = {ts: i for i, ts in enumerate(timeline)}
    pixel_index = {pid: i for i, pid in enumerate(pixel_meta["pixel_id"].values)}

    df = train_df.copy()
    start_idx = df["anchor_t0"].map(timeline_index)
    pixel_col = df["pixel_id"].map(pixel_index)
    valid = start_idx.notna() & pixel_col.notna()
    n_invalid = (~valid).sum()
    if n_invalid > 0:
        say_info(
            f"PERINGATAN noise injection: {n_invalid} baris nggak ketemu di cache "
            "(kemungkinan cache beda run dengan dataset CSV) -- fitur ASLI (tanpa "
            "noise) dipakai buat baris ini, bukan di-skip dari training."
        )

    rng = np.random.default_rng(random_state)
    noised_parts = []

    for L, group in df[valid].groupby(df.loc[valid, "n_points"].round().astype(np.int64)):
        L = int(L)
        n_noised_cols = L - MIN_WINDOW_SIZE
        if n_noised_cols <= 0:
            continue  # step 1: window 100% real, no-op (lihat docstring)

        starts = start_idx[group.index].astype(np.int64).values
        pcols = pixel_col[group.index].astype(np.int64).values
        window_idx = starts[:, None] + np.arange(L)[None, :]
        series = data_matrix[window_idx, pcols[:, None]].astype(np.float64)

        if step_noise_profile is not None:
            # Kolom j (0-indexed) di region ternoise mensimulasikan hasil
            # prediksi rollout STEP (j+1) -- lihat docstring parameter.
            depth_steps = np.arange(1, n_noised_cols + 1)
            missing_steps = sorted(set(depth_steps) - set(step_noise_profile.index))
            if missing_steps:
                raise KeyError(
                    f"step_noise_profile tidak punya nilai utk step {missing_steps} "
                    f"(dibutuhkan utk window n_points={L}) -- profile harus mencakup "
                    "step 1..HORIZON_STEPS-1 penuh."
                )
            col_stds = step_noise_profile.loc[depth_steps].values
            noise = rng.normal(0.0, 1.0, size=(series.shape[0], n_noised_cols)) * col_stds[None, :]
        else:
            noise = rng.normal(0.0, noise_std, size=(series.shape[0], n_noised_cols))
        series[:, MIN_WINDOW_SIZE:] = series[:, MIN_WINDOW_SIZE:] + noise

        noised_feats = compute_window_features_matrix(series)
        noised_feats.index = group.index
        noised_parts.append(noised_feats)

    if noised_parts:
        noised_all = pd.concat(noised_parts)
        df.loc[noised_all.index, FEATURE_COLUMNS] = noised_all[FEATURE_COLUMNS]

    return df


def train_xgboost(X_train, y_train):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


# SVR SUDAH DI-DROP (keputusan final, CLAUDE.md §5) -- terlalu lambat untuk
# ukuran training set expanding window per-pixel (potensi jutaan baris).
TRAINERS = {
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "catboost": train_catboost,
}


def train_one_model(model_name, X_train, y_train):
    """Dispatch ke fungsi training yang sesuai; return (model, detik_training).

    model_name harus salah satu dari Config.MODEL_NAMES (xgboost/lightgbm/
    catboost). Kalau ada nama lain (mis. sisa "svr" dari config lama),
    KeyError eksplisit -- bukan silent skip -- supaya ketauan cepat kalau
    ada inkonsistensi antara Config.MODEL_NAMES dan TRAINERS di sini.
    """
    if model_name not in TRAINERS:
        raise KeyError(
            f"Model '{model_name}' tidak dikenal di TRAINERS ({list(TRAINERS.keys())}). "
            "Cek Config.MODEL_NAMES di pipeline/config.py."
        )
    start = time.time()
    model = TRAINERS[model_name](X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def evaluate(model, X_test, y_test):
    """Evaluasi dasar (MAE/RMSE/R2) di SELURUH test set (semua step
    dicampur). BUKAN evaluasi per-step -- itu tugas recursive_eval.py
    (Tahap 04), yang jalanin model secara recursive per t0 dan hitung MAE
    per step 1..18 secara terpisah (lihat CLAUDE.md §7)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_all_models(df, models_dir=None, test_frac=None, noise_std=0.0, cache_path=None,
                      step_noise_profile=None):
    """Fungsi utama: split (stratified monthly) -> [opsional noise injection
    ke train_df] -> train semua model di Config.MODEL_NAMES -> evaluasi dasar
    -> simpan model (.joblib) + ringkasan metrik (training_summary.csv) ke
    `models_dir`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset hasil dataset_builder.build_dataset() / load_expanding_dataset().
    models_dir : str, optional. Default None -> Config.EXPANDING_MODELS_DIR.
    test_frac : float, optional. Default None -> Config.TEST_FRAC.
    noise_std : float, default 0.0 (tidak ada noise -- perilaku lama,
        backward-compatible). Kalau > 0 DAN `step_noise_profile` None,
        X_train (BUKAN X_test) di-redesain pakai magnitude KONSTAN lewat
        inject_recursive_style_noise() -- lihat docstring fungsi itu.
    cache_path : str, optional. WAJIB diisi (atau Config.EXPANDING_RAW_CACHE_FILE
        dipakai) kalau noise_std > 0 ATAU step_noise_profile diisi --
        dibutuhkan buat rekonstruksi window mentah dari cache.
    step_noise_profile : pd.Series, optional (default None). Kalau diisi,
        noise per titik window jadi PER KEDALAMAN ROLLOUT (bukan konstan
        `noise_std`) -- lihat `build_step_noise_profile()` &
        `inject_recursive_style_noise()`.

    Return
    ------
    summary_df : pd.DataFrame, satu baris per model, kolom: model, n_train,
        n_test, waktu_training_detik, mae, rmse, r2, noise_std (isinya
        "step_profile" -- bukan angka -- kalau step_noise_profile dipakai).
    """
    import os
    import joblib

    if models_dir is None:
        models_dir = Config.EXPANDING_MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    train_df, test_df, cutoffs = stratified_monthly_split(df, test_frac=test_frac)
    say_info(f"Split selesai -- train: {len(train_df)} baris, test: {len(test_df)} baris "
             f"({len(cutoffs)} bulan).")

    noise_std_label = noise_std
    if step_noise_profile is not None:
        train_df = inject_recursive_style_noise(
            train_df, cache_path or Config.EXPANDING_RAW_CACHE_FILE, noise_std,
            step_noise_profile=step_noise_profile,
        )
        noise_std_label = f"step_profile[{step_noise_profile.min():.2f}-{step_noise_profile.max():.2f}]"
    elif noise_std > 0:
        train_df = inject_recursive_style_noise(
            train_df, cache_path or Config.EXPANDING_RAW_CACHE_FILE, noise_std,
        )

    X_train, y_train = get_feature_target(train_df)
    X_test, y_test = get_feature_target(test_df)

    summary_rows = []
    # Progress bar per-model -- training XGBoost/LightGBM/CatBoost bisa makan
    # waktu beberapa menit tergantung ukuran dataset, jadi WAJIB keliatan
    # progresnya (style sama kayak Tahap 1/2), bukan cuma diem nunggu.
    model_progress = make_progress_bar(Config.MODEL_NAMES, desc="Training model", unit="model")
    for model_name in model_progress:
        model_progress.set_postfix_str(model_name)
        say_info(f"Mulai training: {model_name} ...")
        try:
            model, elapsed = train_one_model(model_name, X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)

            say_ok(
                f"{model_name} selesai ({elapsed:.1f}s) -- "
                f"MAE={metrics['mae']:.4f}K RMSE={metrics['rmse']:.4f}K R2={metrics['r2']:.4f}"
            )

            summary_rows.append({
                "model": model_name,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "waktu_training_detik": round(elapsed, 1),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
                "noise_std": noise_std_label,
            })
        except ImportError as e:
            say_error(f"{model_name} dilewati -- library belum terinstall: {e}")
            summary_rows.append({
                "model": model_name, "n_train": len(train_df), "n_test": len(test_df),
                "waktu_training_detik": None, "mae": None, "rmse": None, "r2": None,
                "noise_std": noise_std_label,
                "error": f"Library belum terinstall: {e}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(models_dir, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return summary_df