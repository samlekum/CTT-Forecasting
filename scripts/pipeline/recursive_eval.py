# ./scripts/pipeline/recursive_eval.py
# Tahap 5: evaluasi RECURSIVE (bukan flat/teacher-forced seperti evaluate()
# di model_training.py). Untuk tiap anchor di test set, window mulai dari
# IS1 (6 titik REAL observasi), lalu prediksi step 1 (OS1). Mulai step 2,
# window di-extend pakai HASIL PREDIKSI MODEL SENDIRI (bukan observasi real
# lagi) -- ini yang bikin "recursive": error dari step sebelumnya bisa
# nge-compound ke step berikutnya. Lihat CLAUDE.md §7.
#
# KENAPA PAKAI CACHE .npz, BUKAN BACA ULANG .nc:
# Baca 34rb+ file NetCDF makan ~12.5 jam (I/O bound, sudah divalidasi di
# sesi build dataset). Modul ini re-use cache raw data_matrix yang disimpan
# dataset_builder.save_raw_cache() saat 02_build_expanding_features.py
# jalan, jadi load raw time series di sini tinggal hitungan detik.
#
# KENAPA SLOPE TETAP VALID PAKAI INDEKS LOKAL (BUKAN GLOBAL):
# expanding_features.py menghitung slope dari cumsum_ty dengan t = indeks
# GLOBAL (posisi di timeline penuh). Tapi slope regresi linear y~t itu
# INVARIAN terhadap pergeseran konstan pada t (slope = Cov(t,y)/Var(t),
# keduanya tidak berubah kalau t digeser rata). Jadi menghitung slope
# dengan t lokal (0..L-1, relatif ke window) di modul ini menghasilkan
# nilai identik dengan closed-form training yang pakai t global -- FITUR
# LAIN (mean/std/min/max/first/last) sama sekali tidak bergantung pada t,
# jadi otomatis identik juga. Ini yang bikin kita nggak perlu lacak indeks
# global anchor di sini, cukup array lokal per anchor yang tumbuh tiap step.

import os

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.dataset_builder import load_raw_cache
from pipeline.expanding_features import FEATURE_COLUMNS
from pipeline.model_training import load_expanding_dataset, stratified_monthly_split
from ui.terminal_display import make_progress_bar, say_info, say_ok

MIN_WINDOW_SIZE = Config.MIN_WINDOW_SIZE
HORIZON_STEPS = Config.HORIZON_STEPS
TARGET_COLUMN = f"target_{Config.TARGET_CHANNEL}"


def select_test_anchors(dataset_csv_path=None, test_frac=None):
    """Ambil daftar anchor (pixel_id, anchor_t0) yang masuk test set, pakai
    stratified_monthly_split() PERSIS SAMA seperti waktu training (bukan
    split baru) -- supaya recursive eval dites di anchor yang sama yang
    model belum pernah lihat pas training, konsisten dengan metrik flat di
    03_train_model.py.

    Return
    ------
    pd.DataFrame kolom [pixel_id, anchor_t0], satu baris per anchor unik
    (bukan per step -- 18 step per anchor dibuang duplikatnya di sini).
    """
    if dataset_csv_path is None:
        dataset_csv_path = Config.EXPANDING_DATASET_FILE

    df = load_expanding_dataset(dataset_csv_path)
    _train_df, test_df, _cutoffs = stratified_monthly_split(df, test_frac=test_frac)

    anchors = test_df[["pixel_id", "anchor_t0"]].drop_duplicates().reset_index(drop=True)
    return anchors


def _map_anchors_to_indices(anchors_df, timeline, pixel_meta):
    """Konversi (pixel_id, anchor_t0) jadi (pixel_col_idx, start_idx) numerik
    yang bisa dipakai buat index langsung ke data_matrix.

    Anchor yang timestamp/pixel_id-nya nggak ketemu di cache (harusnya
    nggak terjadi kalau cache berasal dari run 02 yang sama persis dengan
    yang menghasilkan dataset CSV) di-drop dengan warning count, bukan
    bikin crash seluruh proses.
    """
    timeline_index = {ts: i for i, ts in enumerate(timeline)}
    pixel_index = {pid: i for i, pid in enumerate(pixel_meta["pixel_id"].values)}

    start_idx = anchors_df["anchor_t0"].map(timeline_index)
    pixel_col = anchors_df["pixel_id"].map(pixel_index)

    valid = start_idx.notna() & pixel_col.notna()
    n_dropped = (~valid).sum()
    if n_dropped > 0:
        say_info(
            f"PERINGATAN: {n_dropped} anchor di test set nggak ketemu di cache raw "
            "(kemungkinan cache berasal dari run .nc yang beda dengan dataset CSV "
            "saat ini) -- anchor ini di-skip dari recursive eval."
        )

    out = anchors_df[valid].copy()
    out["start_idx"] = start_idx[valid].astype(np.int64)
    out["pixel_col"] = pixel_col[valid].astype(np.int64)
    return out.reset_index(drop=True)


def _compute_window_features_local(series_matrix):
    """Hitung 9 fitur closed-form (FEATURE_COLUMNS) dari window LOKAL yang
    sama panjangnya untuk semua anchor sekaligus (vectorized), tanpa cumsum
    trick -- window di sini selalu pendek (<=23 titik) dan full-recompute
    tiap step tetap murah dibanding kompleksitas cumsum untuk kasus ini
    (beda dari dataset_builder.py yang butuh cumsum karena window bisa
    berapa pun & anchor jumlahnya jutaan sekaligus per pixel).

    Parameters
    ----------
    series_matrix : np.ndarray, shape (n_anchor, L)
        L = panjang window saat ini (sama untuk semua anchor pada step yg
        sama, karena expanding window tumbuh sinkron per step).

    Return
    ------
    pd.DataFrame, kolom = FEATURE_COLUMNS, baris sejajar dengan series_matrix.
    """
    n_anchor, L = series_matrix.shape

    mean_window = series_matrix.mean(axis=1)
    var_window = np.clip(series_matrix.var(axis=1), 0.0, None)  # var(ddof=0), guard floating point sama seperti expanding_features.py
    std_window = np.sqrt(var_window)
    min_window = series_matrix.min(axis=1)
    max_window = series_matrix.max(axis=1)
    first_value = series_matrix[:, 0]
    last_value = series_matrix[:, -1]
    delta_first_last = last_value - first_value

    # Slope pakai t lokal 0..L-1 -- identik dengan t global karena slope
    # invarian terhadap pergeseran konstan (lihat catatan di header modul).
    t = np.arange(L, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = np.sum(t_centered**2)
    if denom == 0:  # L == 1, harusnya nggak pernah kejadian (MIN_WINDOW_SIZE >= 3 dijamin config), guard saja
        slope = np.zeros(n_anchor)
    else:
        numer = ((series_matrix - mean_window[:, None]) * t_centered).sum(axis=1)
        slope = numer / denom

    n_points = np.full(n_anchor, float(L))

    return pd.DataFrame({
        "mean_window": mean_window,
        "std_window": std_window,
        "min_window": min_window,
        "max_window": max_window,
        "first_value": first_value,
        "last_value": last_value,
        "delta_first_last": delta_first_last,
        "slope": slope,
        "n_points": n_points,
    })[FEATURE_COLUMNS]


def run_recursive_evaluation(model_name, model, anchors_idx, data_matrix, timeline):
    """Jalankan recursive forecast untuk SATU model di semua anchor test
    set sekaligus (vectorized lintas anchor, sekuensial lintas step -- step
    k butuh hasil prediksi step k-1, jadi nggak bisa divectorize lintas step).

    Return
    ------
    pd.DataFrame long-format, kolom: model, pixel_id, anchor_t0, step,
    target_time, y_true, y_pred, abs_error. Satu baris per (anchor, step).
    """
    n_anchor = len(anchors_idx)
    starts = anchors_idx["start_idx"].values
    pixel_cols = anchors_idx["pixel_col"].values

    # Window awal (IS1): MIN_WINDOW_SIZE titik REAL, dijamin bebas-NaN
    # karena anchor ini lolos gap-skip rule pas build_dataset() dulu.
    series = np.stack([
        data_matrix[s : s + MIN_WINDOW_SIZE, p]
        for s, p in zip(starts, pixel_cols)
    ])  # shape (n_anchor, MIN_WINDOW_SIZE)

    records = []
    step_progress = make_progress_bar(range(1, HORIZON_STEPS + 1), desc=f"Recursive [{model_name}]", unit="step")
    for step in step_progress:
        L = series.shape[1]
        features_df = _compute_window_features_local(series)
        y_pred = model.predict(features_df)

        target_idx = starts + L  # posisi tepat setelah window saat ini (0-based)
        y_true = data_matrix[target_idx, pixel_cols]
        abs_error = np.abs(y_pred - y_true)

        records.append(pd.DataFrame({
            "model": model_name,
            "pixel_id": anchors_idx["pixel_id"].values,
            "anchor_t0": anchors_idx["anchor_t0"].values,
            "step": step,
            "target_time": timeline[target_idx].values,
            "y_true": y_true,
            "y_pred": y_pred,
            "abs_error": abs_error,
        }))

        # Extend window pakai prediksi sendiri (recursive) -- INI KUNCI
        # BEDANYA dengan evaluate() flat di model_training.py yang selalu
        # pakai observasi real di semua step.
        series = np.concatenate([series, y_pred.reshape(-1, 1)], axis=1)

    return pd.concat(records, ignore_index=True)


def spatial_collapse_ratio(preds, actuals):
    """std(prediksi) / std(aktual) ANTAR PIXEL, pada satu anchor_t0 & step
    yang sama. Mendekati 0 berarti model kolaps jadi rata secara spasial
    dan kehilangan variasi antar pixel -- INI metrik yang langsung
    menjawab masalah utama yang jadi alasan project ini di-rebuild
    (CLAUDE.md §1: "prediction standard deviation collapsed dramatically
    across recursive steps"). MAE saja TIDAK bisa mendeteksi ini (model
    bisa MAE rendah tapi flat/rata secara spasial).

    Naming & definisi disamakan dengan repo lama
    (Bandung-Weather-Forecast-Himawari-09/scripts/pipeline/recursive_eval.py)
    per koreksi CLAUDE.md §12 -- cek preseden dulu sebelum desain dari nol.
    """
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    return float(np.std(preds) / (np.std(actuals) + 1e-6))


def spatial_correlation(preds, actuals):
    """Korelasi spasial prediksi vs aktual ANTAR PIXEL pada satu anchor_t0
    & step yang sama. Dipakai bareng spatial_collapse_ratio -- ratio bisa
    tinggi (variasi terjaga) tapi korelasi rendah (variasinya di tempat
    yang salah / pola spasial nggak match). Sama seperti repo lama."""
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if len(preds) < 2 or np.std(preds) < 1e-9 or np.std(actuals) < 1e-9:
        return 0.0
    return float(np.corrcoef(preds, actuals)[0, 1])


def _spatial_metrics_per_step(detail_df):
    """Untuk tiap (model, step, anchor_t0) yang punya >= 2 pixel (butuh
    minimal 2 titik buat hitung std/korelasi antar pixel yang bermakna),
    hitung spatial_collapse_ratio & spatial_correlation, lalu rata-ratakan
    lintas anchor_t0 per (model, step). anchor_t0 dengan cuma 1 pixel
    (gap-skip rule per-pixel bisa bikin nggak semua pixel valid di
    anchor_t0 yang sama) di-skip dari perhitungan ini -- bukan bug, cuma
    nggak ada "antar pixel" buat dibandingin.
    """
    rows = []
    for (model, step, t0), group in detail_df.groupby(["model", "step", "anchor_t0"]):
        if len(group) < 2:
            continue
        rows.append({
            "model": model,
            "step": step,
            "anchor_t0": t0,
            "collapse_ratio": spatial_collapse_ratio(group["y_pred"].values, group["y_true"].values),
            "correlation": spatial_correlation(group["y_pred"].values, group["y_true"].values),
        })

    if not rows:
        # Nggak ada anchor_t0 dengan >=2 pixel sama sekali (mis. dataset uji
        # sintetis 1 pixel) -- return kolom kosong yang konsisten, jangan crash.
        return pd.DataFrame(columns=["model", "step", "spatial_collapse_ratio", "spatial_correlation", "n_t0_groups"])

    per_t0 = pd.DataFrame(rows)
    summary = per_t0.groupby(["model", "step"]).agg(
        spatial_collapse_ratio=("collapse_ratio", "mean"),
        spatial_correlation=("correlation", "mean"),
        n_t0_groups=("collapse_ratio", "size"),
    ).reset_index()
    return summary


def summarize_by_step(detail_df):
    """Ringkas MAE per (model, step), plus metrik spasial (lihat
    _spatial_metrics_per_step) -- indikator langsung buat ngecek gejala
    'spatial collapse', masalah utama yang jadi alasan rebuild pipeline
    ini (CLAUDE.md §1). BUKAN reliability calibration (CLAUDE.md §7
    sengaja skip itu) -- ini murni ringkasan deskriptif dari detail_df
    yang sudah ada, tanpa nambah kompleksitas desain evaluasi inti.
    """
    grouped = detail_df.groupby(["model", "step"])
    mae_summary = grouped.agg(
        mae=("abs_error", "mean"),
        rmse=("abs_error", lambda s: np.sqrt(np.mean(s**2))),
        n_samples=("abs_error", "size"),
        pred_std=("y_pred", "std"),
        true_std=("y_true", "std"),
    ).reset_index()

    spatial_summary = _spatial_metrics_per_step(detail_df)
    summary = mae_summary.merge(spatial_summary, on=["model", "step"], how="left")
    return summary.sort_values(["model", "step"]).reset_index(drop=True)


def run_all_models(models_dir=None, dataset_csv_path=None, cache_path=None, test_frac=None):
    """Fungsi utama Tahap 5: load cache + test anchors + semua model, jalankan
    recursive eval per model, simpan detail & summary CSV.

    Return
    ------
    (detail_df, summary_df)
    """
    import joblib

    if models_dir is None:
        models_dir = Config.EXPANDING_MODELS_DIR
    if dataset_csv_path is None:
        dataset_csv_path = Config.EXPANDING_DATASET_FILE
    if cache_path is None:
        cache_path = Config.EXPANDING_RAW_CACHE_FILE

    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Cache raw time series tidak ditemukan: {cache_path}\n"
            "Cache ini dibuat otomatis oleh 02_build_expanding_features.py (versi terbaru). "
            "Kalau dataset CSV kamu dibangun sebelum cache ditambahkan, jalankan ulang "
            "02_build_expanding_features.py (tanpa --no-cache) supaya cache-nya kebuat."
        )

    say_info(f"Load cache raw time series: {cache_path}")
    data_matrix, timeline, pixel_meta = load_raw_cache(cache_path)

    say_info(f"Load test anchors dari: {dataset_csv_path}")
    anchors_df = select_test_anchors(dataset_csv_path, test_frac=test_frac)
    anchors_idx = _map_anchors_to_indices(anchors_df, timeline, pixel_meta)
    say_info(f"Jumlah anchor test unik: {len(anchors_idx)}")

    detail_frames = []
    for model_name in Config.MODEL_NAMES:
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            say_info(f"PERINGATAN: model {model_name} tidak ditemukan di {model_path}, dilewati.")
            continue
        model = joblib.load(model_path)
        detail_frames.append(run_recursive_evaluation(model_name, model, anchors_idx, data_matrix, timeline))

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = summarize_by_step(detail_df)

    os.makedirs(Config.EXPANDING_EVAL_DIR, exist_ok=True)
    detail_df.to_csv(Config.RECURSIVE_EVAL_DETAIL_FILE, index=False)
    summary_df.to_csv(Config.RECURSIVE_EVAL_SUMMARY_FILE, index=False)
    say_ok(f"Detail evaluasi disimpan ke: {Config.RECURSIVE_EVAL_DETAIL_FILE}")
    say_ok(f"Ringkasan MAE per step disimpan ke: {Config.RECURSIVE_EVAL_SUMMARY_FILE}")

    return detail_df, summary_df