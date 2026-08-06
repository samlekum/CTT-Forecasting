# ./scripts/pipeline/dataset_builder.py
# Membangun dataset training dari kumpulan file NetCDF hasil 01_download_data.py.
# Alur: scan semua file subset_*.nc -> susun timeline 10-menit uniform per
# pixel -> deteksi gap (file hilang ATAU pixel NaN karena cloud mask) ->
# generate anchor & window expanding per pixel (gap-skip rule, CLAUDE.md §4)
# -> hitung 9 fitur closed-form (expanding_features.py) -> gabung jadi satu
# dataset training dengan kolom target_tbb_13.
#
# CATATAN DESAIN (belum ditulis eksplisit di CLAUDE.md, didiskusikan di sesi
# ini -- lihat juga update di CLAUDE.md §10/§11):
# - Anchor stride = 1 (setiap posisi start valid dipakai). Bisa diubah lewat
#   parameter `anchor_stride` di build_dataset() kalau dataset kegedean.
# - Target channel HANYA tbb_13 (sesuai CLAUDE.md §8), bukan multi-channel.
# - Grid pixel diasumsikan konsisten bentuknya antar file (diverifikasi
#   terhadap grid canonical dari file valid pertama). File dengan shape
#   nyimpang dianggap gap penuh di semua pixel pada timestamp itu.

import os
import logging

import numpy as np
import pandas as pd
import xarray as xr

from pipeline.config import Config
from pipeline.netcdf_tools import extract_time_from_filename
from pipeline.expanding_features import (
    compute_cumsum_stats,
    compute_expanding_window_features,
    FEATURE_COLUMNS,
)

TARGET_CHANNEL = Config.TARGET_CHANNEL
TARGET_COLUMN = f"target_{TARGET_CHANNEL}"

# Ukuran window: IS1 = 6 titik (indeks anchor .. anchor+5), IS18 = 23 titik.
# Total rentang yang dibutuhkan satu anchor (dari titik pertama window
# sampai target OS18) = 24 titik berturut-turut tanpa gap. Lihat CLAUDE.md §2 & §4.
MIN_WINDOW_SIZE = Config.MIN_WINDOW_SIZE
HORIZON_STEPS = Config.HORIZON_STEPS
ANCHOR_SPAN = MIN_WINDOW_SIZE - 1 + HORIZON_STEPS + 1  # = 24


def discover_nc_files(base_dir, filename_pattern="subset_"):
    """Scan `base_dir` secara rekursif, cari semua file subset_*.nc, dan
    kembalikan list (timestamp, filepath) terurut kronologis.

    File yang namanya nggak cocok pola timestamp (extract_time_from_filename
    return None) di-skip dengan warning -- bukan bikin crash seluruh proses.
    """
    entries = []
    for root, _dirs, files in os.walk(base_dir):
        for fname in files:
            if not fname.startswith(filename_pattern) or not fname.endswith(".nc"):
                continue
            ts = extract_time_from_filename(fname)
            if ts is None:
                logging.warning(f"Nama file tidak cocok pola timestamp, dilewati: {fname}")
                continue
            entries.append((ts, os.path.join(root, fname)))

    entries.sort(key=lambda x: x[0])
    return entries


def build_uniform_timeline(entries, freq_minutes=10):
    """Bangun index waktu uniform per `freq_minutes` dari timestamp
    pertama sampai terakhir yang ditemukan. Timestamp yang nggak punya file
    (gap file hilang) tetap ada slot-nya di timeline ini, cuma nanti nilainya
    NaN di semua pixel.
    """
    t_min = entries[0][0]
    t_max = entries[-1][0]
    timeline = pd.date_range(t_min, t_max, freq=f"{freq_minutes}min")
    return timeline


def load_pixel_grid(entries, timeline):
    """Baca semua file NetCDF, ekstrak nilai tbb_13 tiap pixel, susun jadi
    matrix (T, P) di mana T = panjang timeline, P = jumlah pixel (flatten
    grid lat x lon).

    Return
    ------
    data_matrix : np.ndarray, shape (T, P), NaN = gap (file hilang, pixel
        cloud-masked, atau shape file menyimpang dari grid canonical).
    pixel_meta : pd.DataFrame, shape (P, ...), kolom lat_idx/lon_idx/latitude/
        longitude untuk tiap pixel (urutan sejajar dengan kolom data_matrix).
    """
    timeline_index = {ts: i for i, ts in enumerate(timeline)}
    T = len(timeline)

    canonical_shape = None
    canonical_lat = None
    canonical_lon = None
    data_matrix = None
    n_mismatched = 0
    n_missing_channel = 0

    for ts, path in entries:
        t_idx = timeline_index.get(ts)
        if t_idx is None:
            # Timestamp di luar rentang timeline (seharusnya nggak terjadi
            # karena timeline dibangun dari min/max entries, tapi dijaga).
            continue

        try:
            with xr.open_dataset(path) as ds:
                if TARGET_CHANNEL not in ds.variables:
                    n_missing_channel += 1
                    continue

                values = ds[TARGET_CHANNEL].values

                if canonical_shape is None:
                    canonical_shape = values.shape
                    # latitude/longitude di dataset adalah koordinat 1D
                    # (panjang = jumlah baris/kolom grid), bukan grid penuh.
                    # Meshgrid dulu supaya sejajar 1:1 dengan pixel hasil
                    # ravel() dari `values` (row-major: lat berubah lambat,
                    # lon berubah cepat).
                    lat_1d = ds["latitude"].values
                    lon_1d = ds["longitude"].values
                    lat_mesh, lon_mesh = np.meshgrid(lat_1d, lon_1d, indexing="ij")
                    canonical_lat = lat_mesh.ravel()
                    canonical_lon = lon_mesh.ravel()
                    P = int(np.prod(canonical_shape))
                    data_matrix = np.full((T, P), np.nan, dtype=np.float64)

                if values.shape != canonical_shape:
                    # Grid menyimpang dari canonical -> anggap gap penuh
                    # untuk timestamp ini (biarkan NaN, jangan dipaksa reshape).
                    n_mismatched += 1
                    continue

                data_matrix[t_idx, :] = values.ravel()

        except Exception as e:
            logging.warning(f"Gagal baca {path}: {e}")
            continue

    if data_matrix is None:
        raise ValueError(
            f"Tidak ada satupun file yang punya variabel '{TARGET_CHANNEL}' -- "
            "cek nama channel atau isi data_bandung/."
        )

    if n_mismatched > 0:
        logging.warning(
            f"{n_mismatched} file punya shape grid beda dari canonical "
            f"{canonical_shape}, dianggap gap penuh."
        )
    if n_missing_channel > 0:
        logging.warning(
            f"{n_missing_channel} file tidak punya variabel '{TARGET_CHANNEL}', dilewati."
        )

    lat_idx_grid, lon_idx_grid = np.meshgrid(
        np.arange(canonical_shape[0]), np.arange(canonical_shape[1]), indexing="ij"
    )
    pixel_meta = pd.DataFrame({
        "pixel_id": [f"{i}_{j}" for i, j in zip(lat_idx_grid.ravel(), lon_idx_grid.ravel())],
        "lat_idx": lat_idx_grid.ravel(),
        "lon_idx": lon_idx_grid.ravel(),
        "latitude": canonical_lat,
        "longitude": canonical_lon,
    })

    return data_matrix, pixel_meta


def find_valid_anchors(valid_mask, span=ANCHOR_SPAN, stride=1):
    """Cari semua posisi start `a` di mana valid_mask[a : a+span] semuanya
    True (gap-skip rule CLAUDE.md §4: satu gap saja di rentang anchor ->
    seluruh anchor di-skip).

    Vectorized pakai cumulative sum -- bukan loop cek satu-satu per posisi.
    """
    T = len(valid_mask)
    if T < span:
        return np.array([], dtype=np.int64)

    valid_int = valid_mask.astype(np.int64)
    cumsum = np.concatenate(([0], np.cumsum(valid_int)))
    # jumlah valid dalam window [a, a+span-1] = cumsum[a+span] - cumsum[a]
    window_valid_count = cumsum[span:] - cumsum[:-span]
    candidate_starts = np.where(window_valid_count == span)[0]

    if stride > 1:
        candidate_starts = candidate_starts[::stride]

    return candidate_starts


def build_pixel_samples(y, anchors, pixel_id, pixel_lat, pixel_lon, timeline):
    """Untuk satu pixel, generate semua training sample (18 step per anchor)
    dan hitung 9 fitur closed-form sekaligus (vectorized lintas semua anchor
    & step pixel ini).
    """
    if len(anchors) == 0:
        return None

    # Bangun starts/ends untuk SEMUA (anchor x step) sekaligus -> satu kali
    # panggil compute_expanding_window_features, bukan per-anchor.
    steps = np.arange(1, HORIZON_STEPS + 1)
    # starts: tiap anchor diulang HORIZON_STEPS kali
    starts = np.repeat(anchors, HORIZON_STEPS)
    # ends = a + 4 + k, untuk k=1..18 (k dari `steps`, di-tile per anchor)
    ends = starts + 4 + np.tile(steps, len(anchors))
    target_idx = ends + 1

    # PENTING: np.cumsum mem-propagate NaN -- begitu ada satu NaN di `y`,
    # semua cumsum SETELAH posisi itu ikut NaN, walau window yang di-query
    # nggak menyentuh posisi NaN tsb sama sekali. Ganti NaN jadi 0 sebelum
    # cumsum. Ini AMAN karena find_valid_anchors() sudah menjamin setiap
    # window dari anchor yang lolos gap-skip rule nggak pernah menyentuh
    # posisi NaN -- jadi nilai pengganti 0 itu nggak pernah ikut kehitung
    # di sum manapun yang benar-benar dipakai.
    y_filled = np.nan_to_num(y, nan=0.0)
    cumsum_stats = compute_cumsum_stats(y_filled)
    features = compute_expanding_window_features(y, starts, ends, cumsum_stats=cumsum_stats)

    df = pd.DataFrame(features)
    df["pixel_id"] = pixel_id
    df["latitude"] = pixel_lat
    df["longitude"] = pixel_lon
    df["anchor_t0"] = timeline[starts].values
    df["step"] = np.tile(steps, len(anchors))
    df["target_time"] = timeline[target_idx].values
    df[TARGET_COLUMN] = y[target_idx]

    return df


def build_dataset(data_dir=None, output_path=None, anchor_stride=None, freq_minutes=10):
    """Fungsi utama: baca semua file NetCDF di `data_dir`, generate dataset
    training expanding window untuk semua pixel, gabung jadi satu DataFrame.

    Parameters
    ----------
    data_dir : str, optional
        Folder berisi file subset_*.nc. Default None -> pakai
        `Config.FINAL_BASE_DIR` (data_bandung/), sesuai konvensi path
        terpusat yang dipakai 01_download_data.py.
    output_path : str, optional
        Kalau diisi, hasil disimpan sebagai CSV di path ini. None -> tidak
        disimpan, cuma di-return (dipakai testing/eksplorasi tanpa nulis
        file). Untuk full run produksi, pass eksplisit `Config.EXPANDING_DATASET_FILE`.
    anchor_stride : int, optional
        Jarak antar anchor yang dipakai. Default None -> pakai
        `Config.ANCHOR_STRIDE_DEFAULT`.
    freq_minutes : int
        Resolusi timeline (default 10 menit, sesuai Himawari).

    Return
    ------
    pd.DataFrame gabungan semua pixel, kolom: pixel_id, latitude, longitude,
    anchor_t0, step, target_time, 9 kolom fitur (FEATURE_COLUMNS), target_tbb_13.
    """
    if data_dir is None:
        data_dir = Config.FINAL_BASE_DIR
    if anchor_stride is None:
        anchor_stride = Config.ANCHOR_STRIDE_DEFAULT

    entries = discover_nc_files(data_dir)
    if not entries:
        raise ValueError(f"Tidak ada file subset_*.nc ditemukan di {data_dir}")

    timeline = build_uniform_timeline(entries, freq_minutes=freq_minutes)
    data_matrix, pixel_meta = load_pixel_grid(entries, timeline)

    all_samples = []
    for p in range(data_matrix.shape[1]):
        y = data_matrix[:, p]
        valid_mask = ~np.isnan(y)
        anchors = find_valid_anchors(valid_mask, span=ANCHOR_SPAN, stride=anchor_stride)

        row = pixel_meta.iloc[p]
        df_pixel = build_pixel_samples(
            y, anchors, row["pixel_id"], row["latitude"], row["longitude"], timeline
        )
        if df_pixel is not None:
            all_samples.append(df_pixel)

    if not all_samples:
        raise ValueError(
            "Tidak ada anchor valid di pixel manapun -- cek kelengkapan data "
            "(kemungkinan terlalu banyak gap)."
        )

    dataset = pd.concat(all_samples, ignore_index=True)
    ordered_cols = (
        ["pixel_id", "latitude", "longitude", "anchor_t0", "step", "target_time"]
        + FEATURE_COLUMNS
        + [TARGET_COLUMN]
    )
    dataset = dataset[ordered_cols]

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        dataset.to_csv(output_path, index=False)

    return dataset