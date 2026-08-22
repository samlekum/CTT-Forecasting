# ./scripts/pipeline/inference.py
#
# Tahap 7: forecast produksi -- baca window observasi TERBARU (atau di
# sekitar --t0) dari file .nc yang SUDAH ada di data_bandung/, rollout
# recursive HORIZON_STEPS ke depan pakai model window final (hasil
# 05_train_final_models.py), simpan hasil sebagai CSV + GeoJSON.
#
# BEDA dari scripts/_legacy/pipeline/inference.py (metode expanding lama):
#   - Window BUKAN satu nilai global (Config.MIN_WINDOW_SIZE) -- window
#     dibaca PER MODEL dari model_manifest.json (satu-satunya sumber
#     kebenaran, sama seperti 06_evaluate_test.py).
#   - TIDAK ADA damping. Metode window baru tidak pernah memakai damping
#     di training/evaluasi (lihat window_model_training.py) -- rollout
#     produksi harus konsisten dengan itu, jadi prediksi dipakai APA
#     ADANYA sebagai lag_1 di step berikutnya.
#   - Rollout TIDAK ditulis ulang di sini. Loop shift-window sudah ada &
#     sudah divalidasi di window_model_training.recursive_rollout_predict
#     (dipakai bareng oleh 04_search_window.py & 06_evaluate_test.py) --
#     dipanggil ulang, bukan diduplikasi, supaya perilaku rollout evaluasi
#     & rollout produksi TIDAK BISA divergen diam-diam.
#   - Model auto-detect dibaca dari Config.WINDOW_TEST_EVAL_SUMMARY_FILE
#     (output 06_evaluate_test.py), BUKAN evaluation/recursive_mae_summary.csv
#     (itu output metode expanding lama).
#
# SENGAJA TIDAK mengimpor apapun dari pipeline/_legacy/ -- rantai import
# itu sudah rusak (lihat catatan di scripts/_legacy/05_run_inference.py).
# Fungsi baca-NetCDF & cari-anchor (discover_nc_files/build_uniform_timeline/
# load_pixel_grid/find_valid_anchors) di-reuse dari pipeline/dataset_builder.py
# -- fungsi-fungsi itu general-purpose, BUKAN spesifik metode expanding,
# jadi aman dipakai lintas metode.

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.dataset_builder import (
    discover_nc_files,
    build_uniform_timeline,
    load_pixel_grid,
    find_valid_anchors,
)
from pipeline.window_model_training import (
    load_manifest,
    load_model,
    model_path_for,
    recursive_rollout_predict,
)
from ui.terminal_display import say_info, say_ok

FREQ_MINUTES = Config.FREQ_MINUTES
HORIZON_STEPS = Config.HORIZON_STEPS

# Kolom CSV output, urutan tetap. TIDAK ada y_pred_raw/damping_factor
# (beda dari CSV_COLUMNS metode lama) -- metode window tidak punya damping.
CSV_COLUMNS = [
    "pixel_id", "lat_idx", "lon_idx", "latitude", "longitude",
    "window_end_time", "step", "target_time", "y_pred",
    "y_true", "abs_error", "model_name", "window",
]


# ============================================================================
# LOAD DATA TERBARU
# ============================================================================

def load_recent_window_data(
    data_dir=None,
    tail_files=None,
    target_t0=None,
    freq_minutes=None,
    n_workers=None,
):
    """Baca `tail_files` file .nc TERAKHIR dari `data_dir` -- BUKAN seluruh
    riwayat (data_bandung/ bisa berisi puluhan ribu file historis).

    Parameters
    ----------
    target_t0 : str atau pd.Timestamp, optional. Default None -> pakai file
        PALING AKHIR (data terbaru). Kalau diisi, `tail_files` file diambil
        MUNDUR dari `target_t0` (bukan dari file paling akhir).

    Return
    ------
    (data_matrix, timeline, pixel_meta)
    """
    if data_dir is None:
        data_dir = Config.FINAL_BASE_DIR
    if tail_files is None:
        tail_files = Config.INFERENCE_TAIL_FILES

    if tail_files < 1:
        # entries[-0:] (tail_files=0) diam-diam balik jadi entries[0:]
        # (SEMUA entries, karena -0==0 di Python) -- guard eksplisit biar
        # tidak re-trigger baca puluhan ribu file tanpa sadar.
        raise ValueError(f"--tail-files harus >= 1, dapat {tail_files}.")

    entries = discover_nc_files(data_dir)
    if not entries:
        raise ValueError(f"Tidak ada file subset_*.nc ditemukan di {data_dir}")

    if target_t0 is not None:
        target_t0 = pd.Timestamp(target_t0)
        entries_before = [e for e in entries if e[0] <= target_t0]
        if not entries_before:
            raise ValueError(
                f"--t0 {target_t0} lebih awal dari seluruh data yang ada di "
                f"{data_dir} (data paling awal: {entries[0][0]})."
            )
        entries = entries_before[-tail_files:]
        say_info(
            f"Mode --t0={target_t0}: pakai {len(entries)} file s/d "
            f"{entries[-1][0]} (dari {len(entries_before)} file yang <= t0)."
        )
    else:
        n_total = len(entries)
        entries = entries[-tail_files:]
        say_info(
            f"Mode data terbaru: pakai {len(entries)} file terakhir dari "
            f"{n_total} total (s/d {entries[-1][0]})."
        )

    timeline = build_uniform_timeline(entries, freq_minutes=freq_minutes)
    data_matrix, pixel_meta = load_pixel_grid(
        entries, timeline, n_workers=n_workers
    )
    return data_matrix, timeline, pixel_meta


# ============================================================================
# PILIH ANCHOR (window observasi terbaru) PER PIXEL
# ============================================================================

def select_anchor_per_pixel(
    data_matrix,
    timeline,
    pixel_meta,
    window_size,
    target_t0=None,
):
    """Cari window observasi TERBARU per pixel (atau terbaru yang berakhir
    di/sebelum `target_t0` kalau dioverride), panjang `window_size`,
    bebas-NaN.

    `window_size` di sini WAJIB diisi eksplisit (beda dari versi lama yang
    fallback ke Config.MIN_WINDOW_SIZE) -- setiap model window punya window
    optimalnya sendiri (lihat model_manifest.json), jadi tidak ada satu
    default global yang benar untuk semua model.

    Pixel tanpa window valid di-skip dgn warning agregat, BUKAN crash.

    Return
    ------
    pd.DataFrame kolom: pixel_id, lat_idx, lon_idx, latitude, longitude,
    start_idx, end_idx, window_end_time, pixel_col.
    """
    T = len(timeline)
    if target_t0 is not None:
        target_t0 = pd.Timestamp(target_t0)
        eligible_timeline_idx = np.where(timeline <= target_t0)[0]
        if len(eligible_timeline_idx) == 0:
            raise ValueError(
                f"--t0 {target_t0} lebih awal dari seluruh window data yang "
                f"di-load (mulai {timeline[0]})."
            )
        max_end_idx = int(eligible_timeline_idx[-1])
    else:
        max_end_idx = T - 1

    rows = []
    skipped_pixel_ids = []
    for p in range(data_matrix.shape[1]):
        valid_mask = ~np.isnan(data_matrix[:, p])
        anchors = find_valid_anchors(valid_mask, span=window_size, stride=1)
        pixel_id = pixel_meta.iloc[p]["pixel_id"]

        if len(anchors) == 0:
            skipped_pixel_ids.append(pixel_id)
            continue

        end_idxs = anchors + window_size - 1
        eligible = anchors[end_idxs <= max_end_idx]
        if len(eligible) == 0:
            skipped_pixel_ids.append(pixel_id)
            continue

        a = int(eligible[-1])
        row = pixel_meta.iloc[p]
        rows.append({
            "pixel_id": row["pixel_id"],
            "lat_idx": row["lat_idx"],
            "lon_idx": row["lon_idx"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "start_idx": a,
            "end_idx": a + window_size - 1,
            "window_end_time": timeline[a + window_size - 1],
            "pixel_col": p,
        })

    if skipped_pixel_ids:
        say_info(
            f"PERINGATAN: {len(skipped_pixel_ids)} dari "
            f"{data_matrix.shape[1]} pixel di-skip (tidak ada window "
            f"{window_size} titik berturut-turut bebas-NaN yang memenuhi "
            f"batas waktu): {skipped_pixel_ids}"
        )

    if not rows:
        raise ValueError(
            f"Tidak ada satupun pixel yang punya window observasi valid "
            f"({window_size} titik berturut-turut tanpa NaN) di rentang "
            "data yang di-load -- coba naikkan --tail-files atau cek "
            "kualitas data terbaru (cloud cover ekstensif / gap file besar)."
        )

    return pd.DataFrame(rows)


# ============================================================================
# AUTO-DETECT MODEL PRODUKSI
# ============================================================================

def select_inference_model(eval_summary_path=None, priority_step_range=None):
    """Pilih model production OTOMATIS berdasar rata-rata MAE di
    `priority_step_range` (default Config.INFERENCE_PRIORITY_STEP_RANGE =
    (12, 18), prioritas horizon panjang -- rollout recursive bikin error
    terakumulasi paling signifikan di step-step akhir), dibaca dari
    Config.WINDOW_TEST_EVAL_SUMMARY_FILE (output 06_evaluate_test.py,
    metode window BARU -- bukan evaluation/recursive_mae_summary.csv yang
    itu punya metode expanding lama).

    Return
    ------
    (model_name, avg_mae, ranking_df)
    """
    if eval_summary_path is None:
        eval_summary_path = Config.WINDOW_TEST_EVAL_SUMMARY_FILE
    if priority_step_range is None:
        priority_step_range = Config.INFERENCE_PRIORITY_STEP_RANGE

    if not os.path.exists(eval_summary_path):
        raise FileNotFoundError(
            f"File ringkasan evaluasi tidak ditemukan: {eval_summary_path}\n"
            "Jalankan 06_evaluate_test.py dulu supaya model production "
            "bisa dipilih otomatis, atau pilih model manual lewat --model."
        )

    df = pd.read_csv(eval_summary_path)
    lo, hi = priority_step_range
    sub = df[df["step"].between(lo, hi)]
    if sub.empty:
        raise ValueError(
            f"Tidak ada baris dengan step di rentang {priority_step_range} "
            f"di {eval_summary_path} -- cek apakah file ini hasil run yang "
            "HORIZON_STEPS-nya konsisten dgn Config saat ini."
        )

    ranking = sub.groupby("model")["mae"].mean().sort_values().reset_index()
    ranking.columns = ["model", "avg_mae_priority_range"]

    model_name = ranking.iloc[0]["model"]
    avg_mae = float(ranking.iloc[0]["avg_mae_priority_range"])
    return model_name, avg_mae, ranking


# ============================================================================
# BANGUN INITIAL WINDOWS (lag_1..lag_w, lag_1 = paling baru)
# ============================================================================

def build_initial_windows(data_matrix, anchors_df, window_size):
    """Susun array (n_pixel, window_size) dari observasi ASLI di
    data_matrix, urutan kolom SAMA seperti convention training/eval
    (lihat pipeline/window_features.py): kolom 0 = lag_1 (observasi
    paling baru / window_end_time) ... kolom -1 = lag_w (paling lama).

    Ini yang jadi `initial_windows` untuk
    window_model_training.recursive_rollout_predict().
    """
    starts = anchors_df["start_idx"].values
    ends = anchors_df["end_idx"].values
    pixel_cols = anchors_df["pixel_col"].values

    windows = np.stack([
        data_matrix[s:e + 1, p][::-1]
        for s, e, p in zip(starts, ends, pixel_cols)
    ])  # shape (n_pixel, window_size), kolom 0 = paling baru

    return windows


# ============================================================================
# ROLLOUT FORECAST -> LONG-FORMAT DETAIL DF
# ============================================================================

def run_forecast_rollout(model, anchors_df, initial_windows, horizon_steps=None):
    """Rollout recursive dari SATU window per pixel (forecast produksi,
    TIDAK ada y_true dari observasi masa depan yang belum terjadi).

    Loop shift-window-nya sendiri TIDAK ditulis di sini -- dipanggil dari
    window_model_training.recursive_rollout_predict() (SAMA persis yang
    dipakai window search & evaluasi TEST), supaya rollout produksi tidak
    bisa diam-diam berbeda perilakunya dari yang sudah divalidasi di sana.

    Return
    ------
    pd.DataFrame long-format, kolom: pixel_id, lat_idx, lon_idx, latitude,
    longitude, window_end_time, step, target_time, y_pred.
    """
    if horizon_steps is None:
        horizon_steps = HORIZON_STEPS

    predictions = recursive_rollout_predict(
        model, initial_windows, horizon_steps=horizon_steps
    )  # shape (n_pixel, horizon_steps)

    window_end_time = pd.to_datetime(anchors_df["window_end_time"].values)
    n_pixel = len(anchors_df)
    steps = np.arange(1, horizon_steps + 1)

    records = []
    for step in steps:
        target_time = window_end_time + pd.Timedelta(
            minutes=int(step) * FREQ_MINUTES
        )
        records.append(pd.DataFrame({
            "pixel_id": anchors_df["pixel_id"].values,
            "lat_idx": anchors_df["lat_idx"].values,
            "lon_idx": anchors_df["lon_idx"].values,
            "latitude": anchors_df["latitude"].values,
            "longitude": anchors_df["longitude"].values,
            "window_end_time": anchors_df["window_end_time"].values,
            "step": int(step),
            "target_time": target_time,
            "y_pred": predictions[:, step - 1],
        }))

    return pd.concat(records, ignore_index=True)


# ============================================================================
# CARI OBSERVASI ASLI (kalau target_time sudah punya file .nc-nya)
# ============================================================================

def load_raw_values_lookup(data_dir, min_time, max_time, freq_minutes=None, n_workers=None):
    """Cari nilai tbb_13 ASLI dari file .nc yang SUDAH ADA di `data_dir`
    untuk rentang waktu `[min_time, max_time]` (inklusif).

    Kalau tidak ada file yang cocok rentang waktu ini (mis. genuinely masa
    depan yang belum terjadi/didownload), return dict KOSONG, BUKAN error
    -- itu kondisi NORMAL untuk forecast produksi murni.

    Return
    ------
    dict {(pixel_id, pd.Timestamp): float}
    """
    min_time = pd.Timestamp(min_time)
    max_time = pd.Timestamp(max_time)

    entries = discover_nc_files(data_dir)
    entries_in_range = [e for e in entries if min_time <= e[0] <= max_time]
    if not entries_in_range:
        say_info(
            f"Tidak ada file .nc di {data_dir} untuk rentang waktu "
            f"[{min_time}, {max_time}]."
        )
        return {}

    timeline = build_uniform_timeline(entries_in_range, freq_minutes=freq_minutes)
    data_matrix, pixel_meta = load_pixel_grid(
        entries_in_range, timeline, n_workers=n_workers
    )

    lookup = {}
    pixel_ids = pixel_meta["pixel_id"].values
    for t_idx, ts in enumerate(timeline):
        row = data_matrix[t_idx, :]
        for p_idx, pixel_id in enumerate(pixel_ids):
            val = row[p_idx]
            if not np.isnan(val):
                lookup[(pixel_id, pd.Timestamp(ts))] = float(val)

    return lookup


def load_actual_values(detail_df, data_dir=None, freq_minutes=None, n_workers=None):
    """Cari nilai tbb_13 ASLI utk rentang `target_time` di `detail_df` --
    dipakai buat isi `y_true`/`abs_error`. Return dict kosong (bukan
    error) kalau memang belum ada file-nya (forecast produksi murni ke
    masa depan).
    """
    if data_dir is None:
        data_dir = Config.FINAL_BASE_DIR

    min_time = pd.Timestamp(detail_df["target_time"].min())
    max_time = pd.Timestamp(detail_df["target_time"].max())

    lookup = load_raw_values_lookup(
        data_dir, min_time, max_time, freq_minutes=freq_minutes, n_workers=n_workers
    )
    if not lookup:
        say_info(
            "Kemungkinan ini forecast produksi murni (masa depan asli "
            "belum terjadi/didownload) -- kolom y_true/abs_error akan NaN."
        )
        return lookup

    n_possible = len(detail_df)
    n_found = sum(
        1 for _, r in detail_df.iterrows()
        if (r["pixel_id"], pd.Timestamp(r["target_time"])) in lookup
    )
    say_info(f"Observasi asli ketemu untuk {n_found}/{n_possible} baris forecast (sisanya NaN).")
    return lookup


# ============================================================================
# SIMPAN OUTPUT (CSV + GeoJSON)
# ============================================================================

def build_geojson_feature_collection(detail_df, model_name, window, generated_at):
    """Satu Feature per pixel (Point geometry di lat/lon pixel), forecast
    semua step di-embed di properties."""
    features = []
    for pixel_id, group in detail_df.groupby("pixel_id", sort=False):
        group = group.sort_values("step")
        first = group.iloc[0]
        forecast = [
            {
                "step": int(r["step"]),
                "target_time": pd.Timestamp(r["target_time"]).isoformat(),
                "y_pred": float(r["y_pred"]),
                "y_true": None if pd.isna(r["y_true"]) else float(r["y_true"]),
                "abs_error": None if pd.isna(r["abs_error"]) else float(r["abs_error"]),
            }
            for _, r in group.iterrows()
        ]
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(first["longitude"]), float(first["latitude"])],
            },
            "properties": {
                "pixel_id": first["pixel_id"],
                "lat_idx": int(first["lat_idx"]),
                "lon_idx": int(first["lon_idx"]),
                "window_end_time": pd.Timestamp(first["window_end_time"]).isoformat(),
                "forecast": forecast,
            },
        })

    return {
        "type": "FeatureCollection",
        "model_name": model_name,
        "window": window,
        "generated_at": generated_at.isoformat() if hasattr(generated_at, "isoformat") else str(generated_at),
        "features": features,
    }


def save_forecast_outputs(
    detail_df,
    model_name,
    window,
    output_dir=None,
    run_timestamp=None,
    t0_label=None,
):
    """Simpan CSV + GeoJSON dalam SATU FOLDER per run:
    `{output_dir}/{model_name}_w{window}_t0{t0str}_run{run_timestamp}/forecast.csv`
    + `.../forecast.geojson`. Riwayat semua run disimpan sbg folder
    terpisah, BUKAN overwrite "latest".

    Return
    ------
    (run_dir, csv_path, geojson_path)
    """
    if output_dir is None:
        output_dir = Config.INFERENCE_DIR
    if run_timestamp is None:
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    t0_str = (
        pd.Timestamp(t0_label).strftime("%Y%m%d_%H%M")
        if t0_label is not None
        else "unknown"
    )

    run_dir = os.path.join(
        output_dir, f"{model_name}_w{window}_t0{t0_str}_run{run_timestamp}"
    )
    os.makedirs(run_dir, exist_ok=True)

    df = detail_df.copy()
    df["model_name"] = model_name
    df["window"] = window
    df = df[CSV_COLUMNS]

    csv_path = os.path.join(run_dir, "forecast.csv")
    geojson_path = os.path.join(run_dir, "forecast.geojson")

    df.to_csv(csv_path, index=False)

    generated_at = datetime.now()
    geo = build_geojson_feature_collection(detail_df, model_name, window, generated_at)
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)

    say_ok(f"Folder forecast disimpan ke: {run_dir}")
    say_ok(f"Forecast CSV disimpan ke: {csv_path}")
    say_ok(f"Forecast GeoJSON disimpan ke: {geojson_path}")
    return run_dir, csv_path, geojson_path


# ============================================================================
# ORKESTRATOR UTAMA
# ============================================================================

def run_inference(
    data_dir=None,
    manifest_file=None,
    model_name=None,
    tail_files=None,
    target_t0=None,
    output_dir=None,
    n_workers=None,
    run_timestamp=None,
    skip_actual=False,
):
    """Fungsi utama Tahap 7: load window terbaru -> pilih model (auto-detect
    atau manual) -> baca window optimalnya dari manifest -> cari anchor per
    pixel -> rollout recursive -> simpan output.

    Return
    ------
    dict berisi: detail_df, anchors_df, run_dir, csv_path, geojson_path,
    n_used, n_total, model_name, model_auto_detected, window.
    """
    if manifest_file is None:
        manifest_file = Config.WINDOW_FINAL_MANIFEST_FILE

    manifest = load_manifest(manifest_file)

    model_auto_detected = model_name is None
    if model_auto_detected:
        model_name, avg_mae, ranking = select_inference_model()
        say_info(
            f"Model AUTO-DETECT (prioritas step "
            f"{Config.INFERENCE_PRIORITY_STEP_RANGE}): {model_name} "
            f"(avg MAE={avg_mae:.4f}K)"
        )
        say_info(
            f"Ranking model (avg MAE step "
            f"{Config.INFERENCE_PRIORITY_STEP_RANGE}):\n"
            f"{ranking.to_string(index=False)}"
        )
    else:
        say_info(f"Model MANUAL (via --model): {model_name}")

    if model_name not in manifest:
        raise ValueError(
            f"Model '{model_name}' tidak ada di manifest {manifest_file}. "
            f"Tersedia: {list(manifest.keys())}. Jalankan "
            "05_train_final_models.py dulu untuk model ini."
        )

    window = manifest[model_name]["window"]
    say_info(f"Window model '{model_name}' (dari manifest): {window}")

    models_dir = os.path.dirname(os.path.abspath(manifest_file))
    model_path = model_path_for(model_name, models_dir)
    model = load_model(model_path)

    say_info("Load window observasi dari data_bandung/...")
    data_matrix, timeline, pixel_meta = load_recent_window_data(
        data_dir=data_dir, tail_files=tail_files, target_t0=target_t0, n_workers=n_workers,
    )

    anchors_df = select_anchor_per_pixel(
        data_matrix, timeline, pixel_meta, window_size=window, target_t0=target_t0,
    )
    n_used = len(anchors_df)
    n_total = data_matrix.shape[1]
    say_info(f"Pixel dgn window observasi valid: {n_used}/{n_total}.")

    initial_windows = build_initial_windows(data_matrix, anchors_df, window_size=window)

    detail_df = run_forecast_rollout(model, anchors_df, initial_windows)

    if skip_actual:
        detail_df["y_true"] = np.nan
        detail_df["abs_error"] = np.nan
    else:
        say_info(
            "Cek observasi asli (kalau target_time-nya sudah ada file "
            ".nc-nya) buat isi y_true/abs_error..."
        )
        actual_lookup = load_actual_values(detail_df, data_dir=data_dir, n_workers=n_workers)
        detail_df["y_true"] = [
            actual_lookup.get((pid, pd.Timestamp(tt)), np.nan)
            for pid, tt in zip(detail_df["pixel_id"], detail_df["target_time"])
        ]
        detail_df["abs_error"] = (detail_df["y_pred"] - detail_df["y_true"]).abs()

    # t0_label buat nama file -- window_end_time TERBARU yang benar2
    # kepakai (biasanya semua pixel sama, kalau ada pixel dgn window lebih
    # stale/gap, max() tetap kasih titik terbaru yang jadi acuan mayoritas).
    t0_label = anchors_df["window_end_time"].max()

    run_dir, csv_path, geojson_path = save_forecast_outputs(
        detail_df, model_name, window, output_dir=output_dir,
        run_timestamp=run_timestamp, t0_label=t0_label,
    )

    return {
        "detail_df": detail_df,
        "anchors_df": anchors_df,
        "run_dir": run_dir,
        "csv_path": csv_path,
        "geojson_path": geojson_path,
        "n_used": n_used,
        "n_total": n_total,
        "model_name": model_name,
        "model_auto_detected": model_auto_detected,
        "window": window,
    }