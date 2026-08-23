# ./scripts/pipeline/window_features.py
#
# Fixed-size sliding window sample builder untuk eksperimen window search.
#
# BEDA FUNDAMENTAL dari pipeline/expanding_features.py (metode lama):
#   - Window sekarang FIXED-SIZE (bukan expanding/tumbuh per step).
#   - Fitur adalah RAW LAG VALUES (lag_1..lag_w), BUKAN fitur statistik
#     ringkas (mean/std/min/max/slope/dst). Karena window fixed, jumlah
#     kolom otomatis konstan untuk satu kandidat window -- gak butuh
#     ringkasan statistik lagi.
#   - Window CANDIDATE (Config.WINDOW_CANDIDATES) dievaluasi terpisah per
#     model untuk cari window terbaik (lihat 04_search_window.py).
#
# Konvensi kolom:
#   lag_1 = observasi PALING BARU (index t0, ujung window)
#   lag_2 = t0 - 1
#   ...
#   lag_w = t0 - w + 1 (observasi PALING LAMA di window)
#
# Alasan urutan ini (bukan sebaliknya): saat recursive rollout, geser
# window ke step berikutnya tinggal:
#   lag_w, lag_(w-1), ..., lag_2 <- lag_(w-1), ..., lag_1
#   lag_1 <- prediksi baru
# yaitu shift-kanan sederhana, gak perlu mikir index absolut ulang.
#
# Model dilatih SATU STEP ke depan (window -> target t0+1), recursive
# single-model -- SAMA seperti desain metode lama (bukan 18 model
# terpisah per step). Multi-step horizon (18 step) disimulasikan saat
# recursive rollout (evaluasi & inference), bukan saat pembuatan
# training sample.
#
# Implementasi vectorized pakai np.lib.stride_tricks.sliding_window_view
# per pixel -- O(T) per pixel, gak ada loop Python per baris/anchor.

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.time_features import (
    time_features_from_timestamps,
    TIME_FEATURE_COLUMNS,
)


TARGET_COLUMN = "target_tbb"


def lag_column_names(window):
    """Nama kolom lag_1..lag_w, konsisten dipakai di training & inference."""

    return [f"lag_{k}" for k in range(1, window + 1)]


def feature_column_names(window):
    """Nama SEMUA kolom fitur (lag + waktu), urutan tetap: lag_1..lag_w
    dulu baru TIME_FEATURE_COLUMNS.

    SATU-SATUNYA sumber urutan kolom fitur -- dipakai di training
    (04_search_window.py/05_train_final_models.py, X = df[feature_column_names(window)])
    DAN di rollout (window_model_training.recursive_rollout_predict,
    np.concatenate([window_state, time_feats])). Kalau urutan ini beda
    antara training & rollout, model salah interpretasi kolom TANPA
    error apapun (silent bug) -- makanya jangan bangun list ini manual
    di tempat lain, selalu panggil fungsi ini.
    """

    return lag_column_names(window) + list(TIME_FEATURE_COLUMNS)


def compute_window_samples_single_pixel(y, window):
    """
    Bangun sample (window -> target 1 step ke depan) untuk SATU pixel,
    fully vectorized.

    Parameters
    ----------
    y : np.ndarray, shape (T,)
        Time series satu pixel (boleh mengandung NaN untuk gap).
    window : int
        Ukuran window fixed (jumlah lag).

    Returns
    -------
    features : np.ndarray, shape (N, window)
        features[i, 0] = lag_1 (paling baru) ... features[i, -1] = lag_w.
    targets : np.ndarray, shape (N,)
    anchor_idx : np.ndarray, shape (N,)
        Index t0 (posisi observasi paling baru di window) pada `y`.
    target_idx : np.ndarray, shape (N,)
        Index target (anchor_idx + 1).

    Hanya baris yang SELURUH window + target-nya valid (non-NaN) yang
    dikembalikan -- satu NaN di window ATAU di target membuat baris itu
    di-skip seluruhnya (konsisten dengan aturan gap-handling metode lama,
    lihat CLAUDE.md bagian 4).
    """

    T = len(y)

    if T <= window:
        empty_feat = np.empty((0, window), dtype=y.dtype)
        empty_1d = np.empty((0,), dtype=np.int64)
        return empty_feat, empty_1d.astype(y.dtype), empty_1d, empty_1d

    valid = ~np.isnan(y)

    # windows_all[i] = y[i : i+window], berakhir di index i+window-1
    windows_all = np.lib.stride_tricks.sliding_window_view(y, window)
    valid_windows_all = np.lib.stride_tricks.sliding_window_view(
        valid, window
    )

    # buang baris terakhir (i = T-window) karena gak ada target setelahnya
    windows = windows_all[:-1]
    valid_windows = valid_windows_all[:-1]

    targets = y[window:]
    valid_targets = valid[window:]

    row_valid = valid_windows.all(axis=1) & valid_targets

    anchor_idx_all = np.arange(window - 1, T - 1)
    target_idx_all = anchor_idx_all + 1

    windows = windows[row_valid]
    targets = targets[row_valid]
    anchor_idx = anchor_idx_all[row_valid]
    target_idx = target_idx_all[row_valid]

    # windows[i] urutannya [t0-w+1 .. t0] (menaik).
    # Kolom lag_1 harus t0 (paling baru) -> balik urutan kolom.
    features = windows[:, ::-1]

    return features, targets, anchor_idx, target_idx


def build_window_dataset(
    data_matrix,
    timeline,
    pixel_meta,
    window,
):
    """
    Bangun training dataset fixed-window untuk SEMUA pixel sekaligus.

    Parameters
    ----------
    data_matrix : np.ndarray, shape (T, P)
    timeline : pd.DatetimeIndex, shape (T,)
    pixel_meta : dict dengan key pixel_id/lat_idx/lon_idx/latitude/longitude,
        masing-masing array shape (P,)
    window : int

    Returns
    -------
    pd.DataFrame dengan kolom:
        lag_1 .. lag_w, hour_sin, hour_cos, pixel_id, latitude, longitude,
        anchor_time, target_time, target_tbb

    hour_sin/hour_cos dihitung dari target_time (lihat
    pipeline/time_features.py) -- SENGAJA dari waktu TARGET yang
    diprediksi, bukan waktu anchor (t0). Alasan: ini fitur yang harus
    tetap valid saat recursive rollout, di mana yang diketahui pasti di
    tiap step adalah "jam berapa titik yang sedang diprediksi", bukan
    "jam berapa observasi terakhir" (itu sudah terkandung di lag_1).
    """

    T, P = data_matrix.shape

    if len(timeline) != T:
        raise ValueError(
            "Panjang timeline tidak sama dengan jumlah timestep "
            "data_matrix."
        )

    frames = []

    pixel_id = pixel_meta["pixel_id"]
    latitude = pixel_meta["latitude"]
    longitude = pixel_meta["longitude"]

    timeline_values = timeline.values

    for p in range(P):
        y = data_matrix[:, p]

        features, targets, anchor_idx, target_idx = (
            compute_window_samples_single_pixel(y, window)
        )

        if len(targets) == 0:
            continue

        df = pd.DataFrame(
            features,
            columns=lag_column_names(window),
        )

        time_feats = time_features_from_timestamps(
            timeline_values[target_idx]
        )
        for col_idx, col_name in enumerate(TIME_FEATURE_COLUMNS):
            df[col_name] = time_feats[:, col_idx]

        df["pixel_id"] = pixel_id[p]
        df["latitude"] = latitude[p]
        df["longitude"] = longitude[p]
        df["anchor_time"] = timeline_values[anchor_idx]
        df["target_time"] = timeline_values[target_idx]
        df[TARGET_COLUMN] = targets

        frames.append(df)

    if not frames:
        return pd.DataFrame(
            columns=(
                lag_column_names(window)
                + TIME_FEATURE_COLUMNS
                + [
                    "pixel_id",
                    "latitude",
                    "longitude",
                    "anchor_time",
                    "target_time",
                    TARGET_COLUMN,
                ]
            )
        )

    return pd.concat(frames, ignore_index=True)


def find_full_horizon_anchors_single_pixel(y, window, horizon_steps):
    """
    Cari index t0 (anchor) di mana window [t0-w+1, t0] DAN seluruh
    horizon_steps target ke depan [t0+1, t0+horizon_steps] valid
    (non-NaN).

    Dipakai untuk evaluasi RECURSIVE (bukan training single-step) --
    setiap anchor di sini punya observasi asli lengkap sepanjang horizon
    untuk dibandingkan terhadap hasil rollout model
    (window search validation & evaluasi akhir di TEST).

    Returns
    -------
    anchor_idx : np.ndarray
        Index t0 yang valid.
    """

    T = len(y)

    span = window + horizon_steps

    if T < span:
        return np.array([], dtype=np.int64)

    valid = ~np.isnan(y)

    valid_span_all = np.lib.stride_tricks.sliding_window_view(valid, span)
    span_valid = valid_span_all.all(axis=1)

    # start index dari span = t0 - window + 1
    start_idx = np.where(span_valid)[0]
    anchor_idx = start_idx + window - 1

    return anchor_idx


def get_window_and_targets_at_anchor(y, anchor_idx, window, horizon_steps):
    """
    Ambil window observasi (lag_1..lag_w, urutan lag_1=paling baru) dan
    target asli sepanjang horizon_steps untuk satu anchor tertentu.

    Dipakai saat recursive rollout evaluation -- window awal diambil dari
    observasi ASLI, lalu di-update dengan prediksi model di setiap step.
    """

    start = anchor_idx - window + 1
    window_vals = y[start:anchor_idx + 1][::-1]  # lag_1 dulu

    true_future = y[anchor_idx + 1:anchor_idx + 1 + horizon_steps]

    return window_vals, true_future


def build_rollout_arrays_single_pixel(
    y,
    window,
    horizon_steps,
    anchor_stride=1,
):
    """
    Versi VECTORIZED dari get_window_and_targets_at_anchor() untuk SEMUA
    anchor valid sekaligus di satu pixel (fancy indexing, tanpa loop
    Python per anchor) -- dipakai buat window search & evaluasi recursive
    rollout di validation/test set.

    Returns
    -------
    windows : np.ndarray, shape (N, window)
        windows[:, 0] = lag_1 (paling baru) ... windows[:, -1] = lag_w.
    true_future : np.ndarray, shape (N, horizon_steps)
        true_future[:, k] = observasi asli k+1 step setelah anchor.
    anchors : np.ndarray, shape (N,)
        Index t0 yang dipakai (setelah anchor_stride).
    """

    anchors = find_full_horizon_anchors_single_pixel(y, window, horizon_steps)

    if anchor_stride > 1:
        anchors = anchors[::anchor_stride]

    if len(anchors) == 0:
        return (
            np.empty((0, window), dtype=y.dtype),
            np.empty((0, horizon_steps), dtype=y.dtype),
            np.empty((0,), dtype=np.int64),
        )

    lag_offsets = np.arange(window)  # 0, 1, ..., w-1
    window_idx = anchors[:, None] - lag_offsets[None, :]  # (N, w)
    windows = y[window_idx]  # kolom 0 = anchor itu sendiri = lag_1

    future_offsets = np.arange(1, horizon_steps + 1)
    future_idx = anchors[:, None] + future_offsets[None, :]  # (N, horizon)
    true_future = y[future_idx]

    return windows, true_future, anchors


def build_rollout_arrays(
    data_matrix,
    timeline,
    pixel_meta,
    window,
    horizon_steps,
    anchor_stride=1,
):
    """
    Gabungan build_rollout_arrays_single_pixel() untuk SEMUA pixel.

    Returns
    -------
    windows : np.ndarray, shape (N_total, window)
    true_future : np.ndarray, shape (N_total, horizon_steps)
    pixel_ids : np.ndarray, shape (N_total,)
        pixel_id tiap baris (buat breakdown per-pixel kalau dibutuhkan).
    """

    P = data_matrix.shape[1]
    pixel_id = pixel_meta["pixel_id"]

    all_windows = []
    all_future = []
    all_pixel_ids = []

    for p in range(P):
        windows, true_future, anchors = build_rollout_arrays_single_pixel(
            data_matrix[:, p],
            window=window,
            horizon_steps=horizon_steps,
            anchor_stride=anchor_stride,
        )

        if len(anchors) == 0:
            continue

        all_windows.append(windows)
        all_future.append(true_future)
        all_pixel_ids.append(np.full(len(anchors), pixel_id[p]))

    if not all_windows:
        return (
            np.empty((0, window)),
            np.empty((0, horizon_steps)),
            np.empty((0,), dtype=object),
        )

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_future, axis=0),
        np.concatenate(all_pixel_ids, axis=0),
    )


def build_rollout_arrays_with_anchors(
    data_matrix,
    timeline,
    pixel_meta,
    window,
    horizon_steps,
    anchor_stride=1,
):
    """
    Sama seperti build_rollout_arrays(), TAPI juga mengembalikan
    anchor_time (timestamp asli t0 per baris) -- dibutuhkan buat
    grouping spatial metrics (spatial_collapse_ratio, spatial_correlation)
    lintas pixel: baris-baris dengan anchor_time & step yang sama berasal
    dari "kejadian" spasial yang sama (semua pixel grid pada waktu itu),
    jadi harus bisa di-group balik.

    build_rollout_arrays() yang lama SENGAJA tidak diubah/dipakai ulang
    di sini -- fungsi ini duplikat logikanya secara sengaja. CATATAN
    (update fitur waktu): 04_search_window.py SEKARANG pakai fungsi
    `_with_anchors` ini juga (bukan lagi build_rollout_arrays() versi
    tanpa anchor_time), karena recursive_rollout_predict() butuh
    anchor_time buat hitung fitur waktu (hour_sin/hour_cos) tiap step
    rollout. build_rollout_arrays() versi lama dibiarkan ada (tidak
    dihapus) tapi sudah tidak dipanggil di manapun di pipeline aktif.

    Returns
    -------
    windows : np.ndarray, shape (N_total, window)
    true_future : np.ndarray, shape (N_total, horizon_steps)
    pixel_ids : np.ndarray, shape (N_total,)
        pixel_id tiap baris.
    anchor_time : np.ndarray, shape (N_total,), dtype datetime64[ns]
        Timestamp asli t0 (anchor) tiap baris -- sama untuk baris-baris
        dari pixel berbeda yang anchor-nya jatuh di titik waktu yang sama.
    """

    P = data_matrix.shape[1]
    pixel_id = pixel_meta["pixel_id"]
    timeline_values = np.asarray(timeline.values, dtype="datetime64[ns]")

    all_windows = []
    all_future = []
    all_pixel_ids = []
    all_anchor_time = []

    for p in range(P):
        windows, true_future, anchors = build_rollout_arrays_single_pixel(
            data_matrix[:, p],
            window=window,
            horizon_steps=horizon_steps,
            anchor_stride=anchor_stride,
        )

        if len(anchors) == 0:
            continue

        all_windows.append(windows)
        all_future.append(true_future)
        all_pixel_ids.append(np.full(len(anchors), pixel_id[p]))
        all_anchor_time.append(timeline_values[anchors])

    if not all_windows:
        return (
            np.empty((0, window)),
            np.empty((0, horizon_steps)),
            np.empty((0,), dtype=object),
            np.empty((0,), dtype="datetime64[ns]"),
        )

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_future, axis=0),
        np.concatenate(all_pixel_ids, axis=0),
        np.concatenate(all_anchor_time, axis=0),
    )