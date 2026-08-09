# ./scripts/pipeline/recursive_eval.py

import os

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.dataset_builder import load_raw_cache
from pipeline.expanding_features import FEATURE_COLUMNS, compute_window_features_matrix
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


def _apply_damping(raw_pred, last_value, step, damping_factor):
    """Redam delta prediksi model terhadap nilai terakhir di window, makin
    kuat seiring step recursive bertambah.

    KENAPA: diagnostic sesi ini (diagnose_recursive_drift.py +
    check_true_spatial_variance.py, lihat CLAUDE.md update) nemuin
    spatial_collapse_ratio NAIK (bukan turun ke 0) sampai >2 di step 18,
    ternyata dari pred_std yang MEMBESAR ~50% (bukan true_std yang
    mengecil) -- exposure bias klasik: window pas recursive rollout makin
    lama makin isi hasil prediksi sendiri, mendorong fitur (terutama
    std_window/slope/delta_first_last) ke wilayah yang jarang/nggak
    pernah dilihat model pas training (overlap p5-p95 turun sampai ~0.5
    di step 10-18), model jadi ekstrapolasi liar -- paling parah di pixel
    tepi/pojok grid (mis. 4_6, konsisten top-1 di semua model/step).

    FIX: pisahkan prediksi mentah model jadi (last_value + delta), lalu
    kecilkan delta itu secara geometris tiap step (damping_factor ** (step-1))
    SEBELUM dipakai sebagai forecast final DAN sebelum di-feed balik ke
    window. Ini membatasi seberapa jauh model boleh "lari" dari nilai
    persisten terakhir tiap langkah recursive, tanpa perlu retrain atau
    rebuild dataset -- damping_factor=1.0 (default) = TIDAK ADA PERUBAHAN
    (persis perilaku lama, backward-compatible).

    Trade-off: damping_factor < 1 bikin forecast condong ke persistence
    (nilai terakhir) makin lama makin kuat -- bagus buat cegah divergensi,
    tapi kalau terlalu kecil (mis. 0.5) forecast jangka panjang jadi
    hampir flat/tidak informatif. WAJIB di-tuning & dibandingkan MAE per
    step + spatial_collapse_ratio (lihat 04_recursive_evaluate.py --damping-factor),
    bukan asal pilih nilai kecil.
    """
    if damping_factor >= 1.0:
        return raw_pred
    delta = raw_pred - last_value
    damped_delta = delta * (damping_factor ** (step - 1))
    return last_value + damped_delta


def run_recursive_evaluation(model_name, model, anchors_idx, data_matrix, timeline, damping_factor=1.0,
                              capture_features=False):
    """Jalankan recursive forecast untuk SATU model di semua anchor test
    set sekaligus (vectorized lintas anchor, sekuensial lintas step -- step
    k butuh hasil prediksi step k-1, jadi nggak bisa divectorize lintas step).

    Ini SATU-SATUNYA implementasi rollout recursive di pipeline ini --
    scripts/tools/diagnose_recursive_drift.py (diagnostic feature-shift)
    memanggil fungsi INI dengan capture_features=True, BUKAN implementasi
    rollout terpisah, supaya cuma ada satu tempat yang perlu diubah kalau
    logika rollout (mis. formula damping) berubah lagi nanti.

    Parameters
    ----------
    damping_factor : float, default 1.0 (TIDAK ADA REDAMAN -- perilaku lama,
        backward-compatible). Kalau < 1.0, delta prediksi model terhadap
        nilai terakhir di window diredam geometris tiap step -- lihat
        _apply_damping() untuk alasan & detail. Harus di rentang (0, 1].
    capture_features : bool, default False. Kalau True, snapshot 9 fitur
        closed-form TIAP STEP (sebelum dipakai predict) ikut dikumpulkan &
        di-return sebagai DataFrame kedua -- dipakai diagnose_recursive_drift.py
        buat bandingkan distribusi fitur rollout vs training (feature
        distribution shift). Default False = perilaku lama persis (return
        satu DataFrame saja), 100% backward-compatible untuk caller yang
        sudah ada (run_all_models di file ini, sweep_damping.py).

    Return
    ------
    Kalau capture_features=False (default): pd.DataFrame long-format, kolom:
    model, pixel_id, anchor_t0, step, target_time, y_true, y_pred,
    y_pred_raw, abs_error. Satu baris per (anchor, step). y_pred_raw =
    prediksi mentah model SEBELUM damping (sama dengan y_pred kalau
    damping_factor=1.0) -- disimpan supaya bisa dibandingkan langsung efek
    damping-nya tanpa re-run.

    Kalau capture_features=True: tuple (detail_df, features_df) -- detail_df
    sama seperti di atas, features_df long-format kolom FEATURE_COLUMNS +
    "step" (satu baris per (anchor, step), fitur SEBELUM damping diterapkan
    ke prediksi -- damping cuma mempengaruhi angka yang di-feed-back ke
    window step berikutnya, bukan fitur yang dipakai predict di step ini).
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
    feature_records = [] if capture_features else None
    step_progress = make_progress_bar(range(1, HORIZON_STEPS + 1), desc=f"Recursive [{model_name}]", unit="step")
    for step in step_progress:
        L = series.shape[1]
        features_df = compute_window_features_matrix(series)
        y_pred_raw = model.predict(features_df)
        last_value = series[:, -1]
        y_pred = _apply_damping(y_pred_raw, last_value, step, damping_factor)

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
            "y_pred_raw": y_pred_raw,
            "abs_error": abs_error,
        }))

        if capture_features:
            snap = features_df.copy()
            snap["step"] = step
            feature_records.append(snap)

        # Extend window pakai prediksi (SUDAH didamping kalau damping_factor<1)
        # -- ini kunci bedanya dengan evaluate() flat di model_training.py
        # yang selalu pakai observasi real di semua step. Feed-back pakai
        # y_pred yang didamping (bukan y_pred_raw) supaya efek redaman ikut
        # kebawa ke fitur step berikutnya, bukan cuma di angka yang dilaporkan.
        series = np.concatenate([series, y_pred.reshape(-1, 1)], axis=1)

    detail_df = pd.concat(records, ignore_index=True)
    if capture_features:
        return detail_df, pd.concat(feature_records, ignore_index=True)
    return detail_df


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


def run_all_models(models_dir=None, dataset_csv_path=None, cache_path=None, test_frac=None, damping_factor=1.0):
    """Fungsi utama Tahap 5: load cache + test anchors + semua model, jalankan
    recursive eval per model, simpan detail & summary CSV.

    Parameters
    ----------
    damping_factor : float, default 1.0 (tidak ada redaman). Diteruskan ke
        run_recursive_evaluation() -- lihat _apply_damping() untuk alasan
        fix ini ada (investigasi spatial_collapse_ratio naik >2x di step 18,
        lihat CLAUDE.md).

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
        detail_frames.append(run_recursive_evaluation(
            model_name, model, anchors_idx, data_matrix, timeline, damping_factor=damping_factor,
        ))

    if not detail_frames:
        raise FileNotFoundError(
            f"Tidak ada model ditemukan di {models_dir} (dicek: {', '.join(Config.MODEL_NAMES)}). "
            "Jalankan 03_train_models.py dulu, atau cek --models-dir."
        )

    detail_df = pd.concat(detail_frames, ignore_index=True)
    summary_df = summarize_by_step(detail_df)

    # Kalau damping aktif (damping_factor != 1.0), JANGAN timpa file baseline
    # (recursive_evaluation.csv / recursive_mae_summary.csv hasil run tanpa
    # damping) -- suffix nama file pakai damping_factor supaya bisa dibanding
    # langsung sebelum/sesudah tanpa perlu re-run baseline lagi. Default
    # (damping_factor=1.0) TETAP nulis ke path asli, 100% backward-compatible.
    detail_path = Config.RECURSIVE_EVAL_DETAIL_FILE
    summary_path = Config.RECURSIVE_EVAL_SUMMARY_FILE
    if damping_factor < 1.0:
        suffix = f"_damp{damping_factor:.2f}".replace(".", "")
        detail_path = detail_path.replace(".csv", f"{suffix}.csv")
        summary_path = summary_path.replace(".csv", f"{suffix}.csv")

    os.makedirs(Config.EXPANDING_EVAL_DIR, exist_ok=True)
    detail_df.to_csv(detail_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    say_ok(f"Detail evaluasi disimpan ke: {detail_path}")
    say_ok(f"Ringkasan MAE per step disimpan ke: {summary_path}")

    return detail_df, summary_df