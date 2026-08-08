# ./scripts/pipeline/inference.py
# Tahap 6: forecast produksi -- baca window observasi TERBARU (atau di
# sekitar --t0 kalau dioverride) dari file .nc yang SUDAH ada di
# data_bandung/, rollout recursive HORIZON_STEPS ke depan pakai model yang
# sudah dilatih, simpan hasil sebagai CSV + GeoJSON. Model production
# dipilih OTOMATIS dari hasil 04_recursive_evaluate.py (bukan hardcode) --
# lihat select_inference_model().
#
# BEDA dari recursive_eval.run_recursive_evaluation(): fungsi itu dipakai
# EVALUASI (anchor & targetnya historis, SAMA-SAMA sudah ada di
# data_matrix, jadi target_idx bisa langsung dipakai index array). Di sini
# target masa depan TIDAK ADA di data_matrix sama sekali -- target_time
# WAJIB dihitung via aritmatika waktu (anchor_t0 + L*FREQ_MINUTES menit,
# L = panjang window SEBELUM extend step ini -- identik secara matematis
# dgn target_idx=starts+L di run_recursive_evaluation, cuma dikonversi ke
# domain waktu bukan index). Yang tetap di-reuse (BUKAN diduplikasi) adalah
# bagian yang berisiko divergen kalau salah: formula damping
# (_apply_damping) & fitur closed-form (compute_window_features_matrix).
# Loop bookkeeping-nya (predict -> damp -> catat -> extend window) ditulis
# baru di run_forecast_rollout() karena shape & tujuannya beda struktural
# (1 anchor per pixel, bukan ribuan test anchor; tanpa y_true).

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
from pipeline.expanding_features import compute_window_features_matrix
from pipeline.recursive_eval import _apply_damping
from ui.terminal_display import say_info, say_ok

MIN_WINDOW_SIZE = Config.MIN_WINDOW_SIZE
HORIZON_STEPS = Config.HORIZON_STEPS
FREQ_MINUTES = Config.FREQ_MINUTES

# Kolom CSV output, urutan tetap.
CSV_COLUMNS = [
    "pixel_id", "lat_idx", "lon_idx", "latitude", "longitude",
    "anchor_t0", "step", "target_time", "y_pred", "y_pred_raw",
    "y_true", "abs_error", "model_name", "damping_factor",
]


def load_recent_window_data(data_dir=None, tail_files=None, target_t0=None, freq_minutes=None, n_workers=None):
    """Baca `tail_files` file .nc TERAKHIR dari `data_dir` -- BUKAN seluruh
    riwayat (data_bandung/ bisa berisi puluhan ribu file historis, baca
    semuanya re-trigger bottleneck I/O berjam-jam, lihat CLAUDE.md §16).

    Parameters
    ----------
    target_t0 : str atau pd.Timestamp, optional. Default None -> pakai file
        PALING AKHIR (data terbaru). Kalau diisi, `tail_files` file diambil
        MUNDUR dari `target_t0` (bukan dari file paling akhir) -- supaya
        --t0 di masa lalu tidak salah baca file yang tidak relevan.

    Return
    ------
    (data_matrix, timeline, pixel_meta) -- sama seperti
    dataset_builder.load_pixel_grid()/build_uniform_timeline().
    """
    if data_dir is None:
        data_dir = Config.FINAL_BASE_DIR
    if tail_files is None:
        tail_files = Config.INFERENCE_TAIL_FILES
    if tail_files < 1:
        # BUG YANG PERNAH KETEMU: Python slicing entries[-tail_files:] TIDAK
        # error buat tail_files<=0 -- entries[-0:] (tail_files=0) balik jadi
        # entries[0:] (SEMUA entries, karena -0==0 di Python), dan
        # entries[-(-5):] (tail_files=-5) jadi entries[5:] (bukan "5
        # terakhir", tapi "semua kecuali 5 pertama"). Kalau tidak di-guard,
        # tail_files<=0 diam-diam baca SELURUH riwayat data_bandung/ (bisa
        # puluhan ribu file) -- re-trigger bottleneck I/O berjam-jam yang
        # justru jadi alasan --tail-files ini ada (lihat CLAUDE.md §16).
        raise ValueError(f"--tail-files harus >= 1, dapat {tail_files}.")

    entries = discover_nc_files(data_dir)
    if not entries:
        raise ValueError(f"Tidak ada file subset_*.nc ditemukan di {data_dir}")

    if target_t0 is not None:
        target_t0 = pd.Timestamp(target_t0)
        entries_before = [e for e in entries if e[0] <= target_t0]
        if not entries_before:
            raise ValueError(
                f"--t0 {target_t0} lebih awal dari seluruh data yang ada di {data_dir} "
                f"(data paling awal: {entries[0][0]})."
            )
        entries = entries_before[-tail_files:]
        say_info(
            f"Mode --t0={target_t0}: pakai {len(entries)} file s/d {entries[-1][0]} "
            f"(dari {len(entries_before)} file yang <= t0)."
        )
    else:
        n_total = len(entries)
        entries = entries[-tail_files:]
        say_info(f"Mode data terbaru: pakai {len(entries)} file terakhir dari {n_total} total (s/d {entries[-1][0]}).")

    timeline = build_uniform_timeline(entries, freq_minutes=freq_minutes)
    data_matrix, pixel_meta = load_pixel_grid(entries, timeline, n_workers=n_workers)
    return data_matrix, timeline, pixel_meta


def select_anchor_per_pixel(data_matrix, timeline, pixel_meta, window_size=None, target_t0=None):
    """Cari window observasi TERBARU per pixel (atau terbaru yang berakhir
    di/sebelum `target_t0` kalau dioverride), panjang `window_size`
    (default Config.MIN_WINDOW_SIZE), bebas-NaN.

    PENTING soal makna `target_t0`: ini titik waktu observasi TERAKHIR
    yang jadi basis window (bukan `anchor_t0` versi codebase yang berarti
    titik AWAL window IS1) -- forecast dimulai dari `target_t0 +
    FREQ_MINUTES menit`. Kolom `anchor_t0` di return DataFrame tetap
    memakai makna codebase (titik AWAL window, dibutuhkan buat hitung
    target_time di run_forecast_rollout) -- `window_end_time` adalah kolom
    yang sesuai makna `--t0` CLI.

    Pixel tanpa window valid (gagal gap-skip ATAU tidak ada yang berakhir
    di/sebelum target_t0) di-skip dgn warning agregat, BUKAN crash -- pola
    sama seperti recursive_eval._map_anchors_to_indices().

    Return
    ------
    pd.DataFrame kolom: pixel_id, lat_idx, lon_idx, latitude, longitude,
    start_idx, end_idx, anchor_t0, window_end_time, pixel_col.
    """
    if window_size is None:
        window_size = MIN_WINDOW_SIZE

    T = len(timeline)
    if target_t0 is not None:
        target_t0 = pd.Timestamp(target_t0)
        eligible_timeline_idx = np.where(timeline <= target_t0)[0]
        if len(eligible_timeline_idx) == 0:
            raise ValueError(
                f"--t0 {target_t0} lebih awal dari seluruh window data yang di-load (mulai {timeline[0]})."
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
            "anchor_t0": timeline[a],
            "window_end_time": timeline[a + window_size - 1],
            "pixel_col": p,
        })

    if skipped_pixel_ids:
        say_info(
            f"PERINGATAN: {len(skipped_pixel_ids)} dari {data_matrix.shape[1]} pixel di-skip "
            f"(tidak ada window {window_size} titik berturut-turut bebas-NaN yang memenuhi batas "
            f"waktu): {skipped_pixel_ids}"
        )

    if not rows:
        raise ValueError(
            f"Tidak ada satupun pixel yang punya window observasi valid ({window_size} titik "
            "berturut-turut tanpa NaN) di rentang data yang di-load -- coba naikkan --tail-files "
            "atau cek kualitas data terbaru (cloud cover ekstensif / gap file besar)."
        )

    return pd.DataFrame(rows)


def select_inference_model(eval_summary_path=None, priority_step_range=None):
    """Pilih model production OTOMATIS berdasar rata-rata MAE di
    `priority_step_range` (default Config.INFERENCE_PRIORITY_STEP_RANGE =
    (12, 18), prioritas horizon panjang -- lihat CLAUDE.md §18.9: metode
    rekursif bikin error terakumulasi paling signifikan di step-step akhir,
    itu ujian utama kualitas model), dibaca dari
    evaluation/recursive_mae_summary.csv (output BASELINE
    04_recursive_evaluate.py, damping_factor=1.0 -- BUKAN file suffix
    _dampXX). TIDAK hardcode nama model di kode -- kalau dataset/training
    berubah dan model terbaik ikut berubah, fungsi ini otomatis mengikuti.

    Return
    ------
    (model_name, avg_mae, ranking_df) -- ranking_df: semua model + rata-rata
    MAE-nya di priority_step_range, urut menaik (transparansi, bukan
    black-box -- dipakai CLI buat nge-print kenapa model itu dipilih).
    """
    if eval_summary_path is None:
        eval_summary_path = Config.RECURSIVE_EVAL_SUMMARY_FILE
    if priority_step_range is None:
        priority_step_range = Config.INFERENCE_PRIORITY_STEP_RANGE

    if not os.path.exists(eval_summary_path):
        raise FileNotFoundError(
            f"File ringkasan evaluasi tidak ditemukan: {eval_summary_path}\n"
            "Jalankan 04_recursive_evaluate.py dulu (baseline, damping_factor=1.0) supaya model "
            "production bisa dipilih otomatis, atau pilih model manual lewat --model."
        )

    df = pd.read_csv(eval_summary_path)
    lo, hi = priority_step_range
    sub = df[df["step"].between(lo, hi)]
    if sub.empty:
        raise ValueError(
            f"Tidak ada baris dengan step di rentang {priority_step_range} di {eval_summary_path} "
            "-- cek apakah file ini hasil run yang HORIZON_STEPS-nya konsisten dgn Config saat ini."
        )

    ranking = sub.groupby("model")["mae"].mean().sort_values().reset_index()
    ranking.columns = ["model", "avg_mae_priority_range"]

    model_name = ranking.iloc[0]["model"]
    avg_mae = float(ranking.iloc[0]["avg_mae_priority_range"])
    return model_name, avg_mae, ranking


def run_forecast_rollout(model, anchors_df, data_matrix, timeline, horizon_steps=None, damping_factor=1.0):
    """Rollout recursive dari SATU window per pixel (bukan ribuan test
    anchor kayak recursive_eval.run_recursive_evaluation()) -- TIDAK ada
    y_true (masa depan asli belum diketahui, ini forecast produksi
    beneran).

    Return
    ------
    pd.DataFrame long-format, kolom: pixel_id, lat_idx, lon_idx, latitude,
    longitude, anchor_t0, step, target_time, y_pred, y_pred_raw.
    """
    if horizon_steps is None:
        horizon_steps = HORIZON_STEPS

    starts = anchors_df["start_idx"].values
    ends = anchors_df["end_idx"].values
    pixel_cols = anchors_df["pixel_col"].values
    window_size = int(ends[0] - starts[0] + 1)

    series = np.stack([
        data_matrix[s : s + window_size, p]
        for s, p in zip(starts, pixel_cols)
    ])  # shape (n_pixel, window_size)

    anchor_t0_series = pd.to_datetime(anchors_df["anchor_t0"].values)

    records = []
    for step in range(1, horizon_steps + 1):
        L = series.shape[1]  # panjang window SEBELUM extend step ini
        features_df = compute_window_features_matrix(series)
        y_pred_raw = model.predict(features_df)
        last_value = series[:, -1]
        y_pred = _apply_damping(y_pred_raw, last_value, step, damping_factor)

        # target_time = anchor_t0 + L*FREQ_MINUTES menit -- identik secara
        # matematis dgn target_idx=starts+L di run_recursive_evaluation
        # (lihat docstring modul), dikonversi ke domain waktu karena index
        # target tidak eksis di data_matrix untuk inference.
        target_time = anchor_t0_series + pd.Timedelta(minutes=L * FREQ_MINUTES)

        records.append(pd.DataFrame({
            "pixel_id": anchors_df["pixel_id"].values,
            "lat_idx": anchors_df["lat_idx"].values,
            "lon_idx": anchors_df["lon_idx"].values,
            "latitude": anchors_df["latitude"].values,
            "longitude": anchors_df["longitude"].values,
            "anchor_t0": anchors_df["anchor_t0"].values,
            "step": step,
            "target_time": target_time,
            "y_pred": y_pred,
            "y_pred_raw": y_pred_raw,
        }))

        # Extend window pakai prediksi (SUDAH didamping) -- konsisten dgn
        # recursive_eval.run_recursive_evaluation().
        series = np.concatenate([series, y_pred.reshape(-1, 1)], axis=1)

    return pd.concat(records, ignore_index=True)


def load_raw_values_lookup(data_dir, min_time, max_time, freq_minutes=None, n_workers=None):
    """Cari nilai tbb_13 ASLI dari file .nc yang SUDAH ADA di `data_dir`
    untuk rentang waktu `[min_time, max_time]` (inklusif) -- fungsi INTI
    yang di-reuse oleh `load_actual_values()` (Stage 05, rentang =
    target_time forecast) DAN `pipeline/visualize.py::load_input_snapshot()`
    (Stage 06, rentang = window_end_time/t0 per pixel) -- SATU tempat buat
    logic "baca beberapa file .nc spesifik & bangun lookup per timestamp",
    supaya kalau ada bug/perubahan di cara baca NetCDF, cuma 1 tempat yang
    perlu diubah, bukan 2 implementasi terpisah yang bisa divergen.

    Kalau tidak ada file yang cocok rentang waktu ini (mis. genuinely masa
    depan yang belum terjadi/didownload), return dict KOSONG, BUKAN error
    -- itu kondisi NORMAL, bukan kegagalan (caller yang tentukan apa
    artinya dict kosong buat use-case masing-masing).

    Return
    ------
    dict {(pixel_id, pd.Timestamp): float} -- cuma berisi entry yang
    filenya ADA & nilainya bukan NaN (cloud-masked tetap dianggap "tidak
    ada" -- sama seperti gap-skip rule di tempat lain).
    """
    min_time = pd.Timestamp(min_time)
    max_time = pd.Timestamp(max_time)

    entries = discover_nc_files(data_dir)
    entries_in_range = [e for e in entries if min_time <= e[0] <= max_time]
    if not entries_in_range:
        say_info(
            f"Tidak ada file .nc di {data_dir} untuk rentang waktu [{min_time}, {max_time}]."
        )
        return {}

    timeline = build_uniform_timeline(entries_in_range, freq_minutes=freq_minutes)
    data_matrix, pixel_meta = load_pixel_grid(entries_in_range, timeline, n_workers=n_workers)

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
    dipakai buat isi `y_true`/`abs_error`. INI CUMA JALAN kalau file-nya
    beneran ada (mis. --t0 di masa lalu dan "masa depan"-nya kebetulan
    sudah terjadi & sudah didownload). Untuk forecast produksi murni (t0 =
    data terbaru, target_time genuinely belum terjadi), tidak ada file
    yang ketemu -- return dict kosong, BUKAN error (masa depan asli
    memang belum ada).

    Wrapper tipis di atas load_raw_values_lookup() (lihat docstring itu
    utk alasan kenapa logic bacanya disentralkan) -- fungsi ini cuma
    nentuin rentang waktu (dari target_time) & nge-log statistik
    n_found/n_possible yang spesifik utk konteks evaluasi forecast.

    TIDAK dipanggil dari run_forecast_rollout() -- sengaja dipisah supaya
    rollout tetap fokus ke prediksi murni, pencarian observasi asli jadi
    langkah terpisah & opsional (hasil kosong tidak menghentikan pipeline).

    Return
    ------
    dict {(pixel_id, pd.Timestamp): float} -- lihat load_raw_values_lookup().
    """
    if data_dir is None:
        data_dir = Config.FINAL_BASE_DIR

    min_time = pd.Timestamp(detail_df["target_time"].min())
    max_time = pd.Timestamp(detail_df["target_time"].max())

    lookup = load_raw_values_lookup(data_dir, min_time, max_time, freq_minutes=freq_minutes, n_workers=n_workers)
    if not lookup:
        say_info(
            "Kemungkinan ini forecast produksi murni (masa depan asli belum terjadi/didownload) -- "
            "kolom y_true/abs_error akan NaN."
        )
        return lookup

    n_possible = len(detail_df)
    n_found = sum(
        1 for _, r in detail_df.iterrows()
        if (r["pixel_id"], pd.Timestamp(r["target_time"])) in lookup
    )
    say_info(f"Observasi asli ketemu untuk {n_found}/{n_possible} baris forecast (sisanya NaN).")
    return lookup


def build_geojson_feature_collection(detail_df, model_name, damping_factor, generated_at):
    """Satu Feature per pixel (Point geometry di lat/lon pixel), forecast
    semua step di-embed di properties (bukan 18 file terpisah)."""
    features = []
    for pixel_id, group in detail_df.groupby("pixel_id", sort=False):
        group = group.sort_values("step")
        first = group.iloc[0]
        forecast = [
            {
                "step": int(r["step"]),
                "target_time": pd.Timestamp(r["target_time"]).isoformat(),
                "y_pred": float(r["y_pred"]),
                "y_pred_raw": float(r["y_pred_raw"]),
                # NaN bukan JSON valid -- None (json null) kalau observasi
                # asli belum ada (forecast masa depan genuine).
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
                "anchor_t0": pd.Timestamp(first["anchor_t0"]).isoformat(),
                "forecast": forecast,
            },
        })

    return {
        "type": "FeatureCollection",
        "model_name": model_name,
        "damping_factor": damping_factor,
        "generated_at": generated_at.isoformat() if hasattr(generated_at, "isoformat") else str(generated_at),
        "features": features,
    }


def save_forecast_outputs(detail_df, model_name, damping_factor, output_dir=None, run_timestamp=None, t0_label=None):
    """Simpan CSV + GeoJSON, nama file include model_name + t0 (titik
    referensi waktu forecast-nya -- window_end_time yang benar2 kepakai,
    BUKAN cuma kapan script-nya dijalankan) + run_timestamp (kapan
    script-nya dijalankan). Retensi: riwayat semua run disimpan, BUKAN
    overwrite "latest".

    `t0_label` idealnya SELALU diisi caller (run_inference() isi ini dari
    anchors_df, representatif walau --t0 tidak dioverride -- default-nya
    ambil window_end_time terbaru yang benar2 dipakai). Kalau None
    (dipanggil manual di luar run_inference()), nama file fallback cuma
    pakai run_timestamp (t0 tidak diketahui, ditandai "t0unknown").

    Return
    ------
    (csv_path, geojson_path)
    """
    if output_dir is None:
        output_dir = Config.INFERENCE_DIR
    if run_timestamp is None:
        # Sampai menit saja (bukan detik) -- cukup buat retensi timestamped
        # per run batch terjadwal, nama file lebih ringkas. Trade-off: kalau
        # ada 2 run untuk model+t0 yang sama dalam menit yang sama, file
        # kedua akan menimpa yang pertama (jarang terjadi buat batch
        # terjadwal).
        run_timestamp = datetime.now().strftime("%Y%m%d_%H%M")

    t0_str = pd.Timestamp(t0_label).strftime("%Y%m%d_%H%M") if t0_label is not None else "unknown"

    os.makedirs(output_dir, exist_ok=True)

    df = detail_df.copy()
    df["model_name"] = model_name
    df["damping_factor"] = damping_factor
    df = df[CSV_COLUMNS]

    base_name = f"forecast_{model_name}_t0{t0_str}_run{run_timestamp}"
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    geojson_path = os.path.join(output_dir, f"{base_name}.geojson")

    df.to_csv(csv_path, index=False)

    generated_at = datetime.now()
    geo = build_geojson_feature_collection(detail_df, model_name, damping_factor, generated_at)
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(geo, f, indent=2)

    say_ok(f"Forecast CSV disimpan ke: {csv_path}")
    say_ok(f"Forecast GeoJSON disimpan ke: {geojson_path}")
    return csv_path, geojson_path


def run_inference(data_dir=None, models_dir=None, model_name=None, tail_files=None, target_t0=None,
                   damping_factor=1.0, output_dir=None, n_workers=None, run_timestamp=None):
    """Fungsi utama Tahap 6: load window terbaru -> cari anchor per pixel ->
    pilih model (auto-detect atau manual) -> rollout recursive -> simpan
    output.

    Return
    ------
    dict berisi: detail_df, anchors_df, csv_path, geojson_path, n_used,
    n_total, model_name, model_auto_detected.
    """
    import joblib

    if models_dir is None:
        models_dir = Config.EXPANDING_MODELS_DIR

    say_info("Load window observasi dari data_bandung/...")
    data_matrix, timeline, pixel_meta = load_recent_window_data(
        data_dir=data_dir, tail_files=tail_files, target_t0=target_t0, n_workers=n_workers,
    )

    anchors_df = select_anchor_per_pixel(data_matrix, timeline, pixel_meta, target_t0=target_t0)
    n_used = len(anchors_df)
    n_total = data_matrix.shape[1]
    say_info(f"Pixel dgn window observasi valid: {n_used}/{n_total}.")

    model_auto_detected = model_name is None
    if model_auto_detected:
        model_name, avg_mae, ranking = select_inference_model()
        say_info(
            f"Model AUTO-DETECT (prioritas step {Config.INFERENCE_PRIORITY_STEP_RANGE}): "
            f"{model_name} (avg MAE={avg_mae:.4f}K)"
        )
        say_info(f"Ranking model (avg MAE step {Config.INFERENCE_PRIORITY_STEP_RANGE}):\n"
                  f"{ranking.to_string(index=False)}")
    else:
        say_info(f"Model MANUAL (via --model): {model_name}")

    model_path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}\nJalankan 03_train_models.py dulu.")
    model = joblib.load(model_path)

    detail_df = run_forecast_rollout(model, anchors_df, data_matrix, timeline, damping_factor=damping_factor)

    say_info("Cek observasi asli (kalau target_time-nya sudah ada file .nc-nya) buat isi y_true/abs_error...")
    actual_lookup = load_actual_values(detail_df, data_dir=data_dir, n_workers=n_workers)
    detail_df["y_true"] = [
        actual_lookup.get((pid, pd.Timestamp(tt)), np.nan)
        for pid, tt in zip(detail_df["pixel_id"], detail_df["target_time"])
    ]
    detail_df["abs_error"] = (detail_df["y_pred"] - detail_df["y_true"]).abs()

    # t0_label buat nama file -- window_end_time TERBARU yang benar2 kepakai
    # (bukan `target_t0` mentah dari argumen, karena kalau --t0 tidak diisi
    # `target_t0` itu None; ambil dari anchors_df supaya SELALU ada label
    # yang representatif, baik --t0 dioverride maupun pakai data terbaru).
    # Biasanya semua pixel sama -- kalau ada pixel yang window-nya lebih
    # stale (gap), max() tetap kasih titik terbaru yang jadi acuan mayoritas.
    t0_label = anchors_df["window_end_time"].max()

    csv_path, geojson_path = save_forecast_outputs(
        detail_df, model_name, damping_factor, output_dir=output_dir, run_timestamp=run_timestamp,
        t0_label=t0_label,
    )

    return {
        "detail_df": detail_df,
        "anchors_df": anchors_df,
        "csv_path": csv_path,
        "geojson_path": geojson_path,
        "n_used": n_used,
        "n_total": n_total,
        "model_name": model_name,
        "model_auto_detected": model_auto_detected,
    }
