# scripts/pipeline/dataset_builder.py
#
# Membangun dataset training dari kumpulan file NetCDF hasil
# 01_download_data.py.
#
# Alur utama:
#   scan file NetCDF
#       ↓
#   susun timeline 10-menit uniform
#       ↓
#   baca tbb_13 menjadi temporal matrix (T, P)
#       ↓
#   simpan raw temporal cache
#       ↓
#   generate anchor & expanding window
#       ↓
#   hitung fitur
#       ↓
#   gabung menjadi dataset training
#
# CATATAN:
# Tahap pembacaan NetCDF menggunakan netCDF4.Dataset secara langsung
# untuk mengurangi overhead xarray pada setiap file.
#
# Tidak ada perubahan terhadap logika anchor/window/feature engineering.
#

import os
import logging
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
from netCDF4 import Dataset
from tqdm import tqdm

from pipeline.config import Config
from pipeline.netcdf_tools import extract_time_from_filename
from pipeline.expanding_features import (
    compute_cumsum_stats,
    compute_expanding_window_features,
    FEATURE_COLUMNS,
)
from ui.terminal_display import make_progress_bar, say_info, say_error


TARGET_CHANNEL = Config.TARGET_CHANNEL
TARGET_COLUMN = f"target_{TARGET_CHANNEL}"


# ============================================================
# CONFIG WINDOW
# ============================================================

MIN_WINDOW_SIZE = Config.LEGACY_MIN_WINDOW_SIZE
HORIZON_STEPS = Config.HORIZON_STEPS

# ANCHOR_SPAN sengaja TIDAK lagi disediakan Config (dihapus saat config.py
# dirombak ke metode chronological, lihat komentar
# Config.LEGACY_MIN_WINDOW_SIZE) -- direkonstruksi lokal di sini persis
# formula lama: window + horizon (mis. MIN_WINDOW_SIZE=6, HORIZON_STEPS=18
# -> ANCHOR_SPAN=24, cocok dengan komentar lama di model_training.py
# "230 menit = ANCHOR_SPAN-1 titik" = 23*10menit, ANCHOR_SPAN=24).
ANCHOR_SPAN = MIN_WINDOW_SIZE + HORIZON_STEPS

WINDOW_OFFSET = MIN_WINDOW_SIZE - 2


# ============================================================
# DISCOVER FILES
# ============================================================

def discover_nc_files(base_dir, filename_pattern="subset_"):
    """
    Scan base_dir secara rekursif dan mencari file subset_*.nc.

    Return:
        list[(timestamp, filepath)]
    """

    entries = []

    for root, _dirs, files in os.walk(base_dir):

        for fname in files:

            if not fname.startswith(filename_pattern):
                continue

            if not fname.endswith(".nc"):
                continue

            ts = extract_time_from_filename(fname)

            if ts is None:
                logging.warning(
                    f"Nama file tidak cocok pola timestamp, "
                    f"dilewati: {fname}"
                )
                continue

            entries.append(
                (
                    ts,
                    os.path.join(root, fname)
                )
            )

    entries.sort(key=lambda x: x[0])

    return entries


# ============================================================
# UNIFORM TIMELINE
# ============================================================

def build_uniform_timeline(entries, freq_minutes=None):
    """
    Bangun timeline uniform dari timestamp pertama sampai terakhir.

    File yang hilang tetap mendapatkan slot waktu.
    Nilainya nanti akan menjadi NaN.
    """

    if freq_minutes is None:
        freq_minutes = Config.FREQ_MINUTES

    t_min = entries[0][0]
    t_max = entries[-1][0]

    timeline = pd.date_range(
        t_min,
        t_max,
        freq=f"{freq_minutes}min"
    )

    return timeline


# ============================================================
# FAST NETCDF WORKER
# ============================================================

def _read_netcdf_worker(
    path,
    target_channel,
    canonical_shape,
):
    """
    Worker untuk membaca SATU file NetCDF.

    Menggunakan netCDF4.Dataset secara langsung, bukan
    xarray.open_dataset(), karena tahap ini hanya membutuhkan
    satu variabel target.

    Return:
        (values_flat, status)

    values_flat:
        np.ndarray 1D jika berhasil.

    status:
        None
        "missing_channel"
        "shape_mismatch"
        "error:<message>"
    """

    try:

        with Dataset(path, "r") as ds:

            if target_channel not in ds.variables:

                return None, "missing_channel"

            values = np.asarray(
                ds.variables[target_channel][:]
            )

            if (
                canonical_shape is not None
                and values.shape != canonical_shape
            ):

                return None, "shape_mismatch"

            return values.ravel(), None

    except Exception as e:

        return None, f"error:{e}"


# ============================================================
# LOAD PIXEL GRID
# ============================================================

def load_pixel_grid(
    entries,
    timeline,
    n_workers=None,
):
    """
    Baca seluruh file NetCDF dan susun temporal matrix:

        data_matrix.shape = (T, P)

    T = jumlah timestep
    P = jumlah pixel

    Missing file / NaN / shape mismatch:
        → NaN

    Pembacaan file dilakukan paralel menggunakan
    ProcessPoolExecutor.
    """

    if n_workers is None:
        n_workers = Config.NETCDF_READ_WORKERS

    timeline_index = {
        ts: i
        for i, ts in enumerate(timeline)
    }

    T = len(timeline)

    # ========================================================
    # PHASE 1
    # Cari file valid pertama
    # ========================================================

    canonical_shape = None
    canonical_lat = None
    canonical_lon = None

    data_matrix = None

    n_mismatched = 0
    n_missing_channel = 0
    n_errors = 0

    first_valid_pos = None

    for pos, (ts, path) in enumerate(entries):

        t_idx = timeline_index.get(ts)

        if t_idx is None:
            continue

        try:

            with Dataset(path, "r") as ds:

                if TARGET_CHANNEL not in ds.variables:

                    n_missing_channel += 1

                    continue

                values = np.asarray(
                    ds.variables[TARGET_CHANNEL][:]
                )

                canonical_shape = values.shape

                # ------------------------------------------------
                # Latitude / longitude
                # ------------------------------------------------

                if "latitude" not in ds.variables:
                    raise ValueError(
                        f"Variabel latitude tidak ditemukan: {path}"
                    )

                if "longitude" not in ds.variables:
                    raise ValueError(
                        f"Variabel longitude tidak ditemukan: {path}"
                    )

                lat_1d = np.asarray(
                    ds.variables["latitude"][:]
                )

                lon_1d = np.asarray(
                    ds.variables["longitude"][:]
                )

                # ------------------------------------------------
                # Flatten grid
                # ------------------------------------------------

                lat_mesh, lon_mesh = np.meshgrid(
                    lat_1d,
                    lon_1d,
                    indexing="ij"
                )

                canonical_lat = lat_mesh.ravel()
                canonical_lon = lon_mesh.ravel()

                P = int(
                    np.prod(canonical_shape)
                )

                # ------------------------------------------------
                # Allocate temporal matrix
                #
                # float32 cukup untuk TBB Kelvin dan menghemat
                # memory dibanding float64.
                # ------------------------------------------------

                data_matrix = np.full(
                    (T, P),
                    np.nan,
                    dtype=np.float32
                )

                data_matrix[t_idx, :] = (
                    values.ravel().astype(
                        np.float32,
                        copy=False
                    )
                )

                first_valid_pos = pos

        except Exception as e:

            n_errors += 1

            logging.warning(
                f"Gagal baca {path}: {e}"
            )

            continue

        break

    # ========================================================
    # Tidak ada file valid
    # ========================================================

    if data_matrix is None:

        raise ValueError(
            f"Tidak ada satupun file yang punya variabel "
            f"'{TARGET_CHANNEL}' -- cek nama channel atau "
            f"isi data_bandung/."
        )

    # ========================================================
    # PHASE 2
    # Semua file setelah file valid pertama
    # ========================================================

    remaining = []

    for pos in range(
        first_valid_pos + 1,
        len(entries)
    ):

        ts, path = entries[pos]

        t_idx = timeline_index.get(ts)

        if t_idx is None:
            continue

        remaining.append(
            (t_idx, path)
        )

    # ========================================================
    # READ REMAINING FILES
    # ========================================================

    if remaining:

        # ----------------------------------------------------
        # Sequential fallback
        # ----------------------------------------------------

        if n_workers <= 1:

            progress = make_progress_bar(
                remaining,
                desc="Baca NetCDF",
                unit="file"
            )

            for t_idx, path in progress:

                values_flat, status = (
                    _read_netcdf_worker(
                        path,
                        TARGET_CHANNEL,
                        canonical_shape,
                    )
                )

                if status == "missing_channel":

                    n_missing_channel += 1

                elif status == "shape_mismatch":

                    n_mismatched += 1

                elif status is not None:

                    n_errors += 1

                    logging.warning(
                        f"Gagal baca {path}: {status}"
                    )

                else:

                    data_matrix[
                        t_idx, :
                    ] = values_flat.astype(
                        np.float32,
                        copy=False
                    )

        # ----------------------------------------------------
        # Parallel
        # ----------------------------------------------------

        else:

            # Jangan gunakan chunksize yang bergantung pada
            # jumlah total file.
            #
            # Dengan 34.419 file dan 7 worker, formula lama:
            #
            #   len(remaining) // (n_workers * 4)
            #
            # menghasilkan:
            #
            #   34.419 // 28 = 1.229
            #
            # Chunk sebesar ini membuat progress/scheduling
            # menjadi sangat kasar.
            #
            # Gunakan chunk kecil dan stabil.

            chunksize = 8

            say_info(
                f"Baca NetCDF paralel: "
                f"{len(remaining)} file tersisa, "
                f"{n_workers} proses, "
                f"chunksize={chunksize}"
            )

            paths = [
                path
                for _t_idx, path in remaining
            ]

            target_channels = [
                TARGET_CHANNEL
            ] * len(remaining)

            canonical_shapes = [
                canonical_shape
            ] * len(remaining)

            with ProcessPoolExecutor(
                max_workers=n_workers
            ) as executor:

                results_iter = executor.map(
                    _read_netcdf_worker,
                    paths,
                    target_channels,
                    canonical_shapes,
                    chunksize=chunksize,
                )

                progress = tqdm(
                    zip(
                        remaining,
                        results_iter
                    ),
                    total=len(remaining),
                    desc="Baca NetCDF (paralel)",
                    bar_format=(
                        "{desc} |{bar}| "
                        "{n_fmt}/{total_fmt} file "
                        "[{elapsed}<{remaining}]"
                    ),
                    ncols=80,
                )

                for (
                    (t_idx, path),
                    (values_flat, status),
                ) in progress:

                    if status == "missing_channel":

                        n_missing_channel += 1

                    elif status == "shape_mismatch":

                        n_mismatched += 1

                    elif status is not None:

                        n_errors += 1

                        logging.warning(
                            f"Gagal baca {path}: {status}"
                        )

                    else:

                        data_matrix[
                            t_idx, :
                        ] = values_flat.astype(
                            np.float32,
                            copy=False
                        )

    # ========================================================
    # REPORT ERRORS
    # ========================================================

    if n_mismatched > 0:

        logging.warning(
            f"{n_mismatched} file punya shape grid beda "
            f"dari canonical {canonical_shape}, "
            f"dianggap gap penuh."
        )

    if n_missing_channel > 0:

        logging.warning(
            f"{n_missing_channel} file tidak punya "
            f"variabel '{TARGET_CHANNEL}', dilewati."
        )

    if n_errors > 0:

        say_error(
            f"{n_errors} file gagal dibaca "
            f"(exception saat buka/parse) -- "
            f"lihat log warning untuk detail per file, "
            f"dianggap gap."
        )

    # ========================================================
    # PIXEL METADATA
    # ========================================================

    lat_idx_grid, lon_idx_grid = np.meshgrid(
        np.arange(canonical_shape[0]),
        np.arange(canonical_shape[1]),
        indexing="ij",
    )

    pixel_meta = pd.DataFrame(
        {
            "pixel_id": [
                f"{i}_{j}"
                for i, j in zip(
                    lat_idx_grid.ravel(),
                    lon_idx_grid.ravel(),
                )
            ],
            "lat_idx": lat_idx_grid.ravel(),
            "lon_idx": lon_idx_grid.ravel(),
            "latitude": canonical_lat,
            "longitude": canonical_lon,
        }
    )

    return data_matrix, pixel_meta


# ============================================================
# FIND VALID ANCHORS
# ============================================================

def find_valid_anchors(
    valid_mask,
    span=ANCHOR_SPAN,
    stride=1,
):
    """
    Cari posisi start anchor di mana seluruh rentang span
    valid.

    Satu NaN dalam span menyebabkan anchor tersebut di-skip.
    """

    T = len(valid_mask)

    if T < span:

        return np.array(
            [],
            dtype=np.int64
        )

    valid_int = valid_mask.astype(
        np.int64
    )

    cumsum = np.concatenate(
        (
            [0],
            np.cumsum(valid_int)
        )
    )

    window_valid_count = (
        cumsum[span:]
        -
        cumsum[:-span]
    )

    candidate_starts = np.where(
        window_valid_count == span
    )[0]

    if stride > 1:

        candidate_starts = (
            candidate_starts[::stride]
        )

    return candidate_starts


# ============================================================
# BUILD PIXEL SAMPLES
# ============================================================

def build_pixel_samples(
    y,
    anchors,
    pixel_id,
    pixel_lat,
    pixel_lon,
    timeline,
):
    """
    Generate seluruh training sample untuk satu pixel.

    Logika expanding window TIDAK diubah.
    """

    if len(anchors) == 0:

        return None

    steps = np.arange(
        1,
        HORIZON_STEPS + 1
    )

    starts = np.repeat(
        anchors,
        HORIZON_STEPS
    )

    ends = (
        starts
        +
        WINDOW_OFFSET
        +
        np.tile(
            steps,
            len(anchors)
        )
    )

    target_idx = ends + 1

    # NaN → 0 hanya untuk cumsum.
    # Anchor sudah dijamin tidak menyentuh NaN.
    y_filled = np.nan_to_num(
        y,
        nan=0.0
    )

    cumsum_stats = (
        compute_cumsum_stats(
            y_filled
        )
    )

    features = (
        compute_expanding_window_features(
            y,
            starts,
            ends,
            cumsum_stats=cumsum_stats,
        )
    )

    df = pd.DataFrame(
        features
    )

    df["pixel_id"] = pixel_id

    df["latitude"] = pixel_lat

    df["longitude"] = pixel_lon

    df["anchor_t0"] = (
        timeline[starts].values
    )

    df["step"] = np.tile(
        steps,
        len(anchors)
    )

    df["target_time"] = (
        timeline[target_idx].values
    )

    df[TARGET_COLUMN] = (
        y[target_idx]
    )

    return df


# ============================================================
# SAVE RAW CACHE
# ============================================================

def save_raw_cache(
    cache_path,
    data_matrix,
    timeline,
    pixel_meta,
):
    """
    Simpan raw temporal cache ke NPZ.

    Cache berisi:
        data_matrix
        timeline_ns
        pixel_id
        lat_idx
        lon_idx
        latitude
        longitude
    """

    os.makedirs(
        os.path.dirname(cache_path),
        exist_ok=True
    )

    pixel_id_arr = np.array(
        pixel_meta["pixel_id"].values,
        dtype="<U32"
    )

    np.savez_compressed(
        cache_path,

        data_matrix=data_matrix.astype(
            np.float32
        ),

        timeline_ns=(
            timeline.values
            .astype("datetime64[ns]")
            .astype(np.int64)
        ),

        pixel_id=pixel_id_arr,

        lat_idx=pixel_meta[
            "lat_idx"
        ].values,

        lon_idx=pixel_meta[
            "lon_idx"
        ].values,

        latitude=pixel_meta[
            "latitude"
        ].values,

        longitude=pixel_meta[
            "longitude"
        ].values,
    )


# ============================================================
# LOAD RAW CACHE
# ============================================================

def load_raw_cache(cache_path):
    """
    Load kembali raw temporal cache.

    Perilaku dtype dipertahankan:
    data_matrix dikembalikan sebagai float64.
    """

    npz = np.load(
        cache_path,
        allow_pickle=False
    )

    data_matrix = (
        npz["data_matrix"]
        .astype(np.float64)
    )

    timeline = pd.to_datetime(
        npz["timeline_ns"]
    )

    pixel_meta = pd.DataFrame(
        {
            "pixel_id": (
                npz["pixel_id"]
                .astype("<U32")
                .astype(str)
            ),

            "lat_idx": npz["lat_idx"],

            "lon_idx": npz["lon_idx"],

            "latitude": npz["latitude"],

            "longitude": npz["longitude"],
        }
    )

    return (
        data_matrix,
        timeline,
        pixel_meta,
    )


# ============================================================
# BUILD DATASET
# ============================================================

def build_dataset(
    data_dir=None,
    output_path=None,
    anchor_stride=None,
    freq_minutes=None,
    max_files=None,
    cache_path=None,
    n_workers=None,
):
    """
    Fungsi utama pipeline dataset.

    Tahapan:

        NetCDF
          ↓
        temporal matrix
          ↓
        raw cache
          ↓
        anchor
          ↓
        expanding features
          ↓
        training dataset
    """

    if data_dir is None:

        data_dir = (
            Config.FINAL_BASE_DIR
        )

    if anchor_stride is None:

        anchor_stride = (
            Config.ANCHOR_STRIDE_DEFAULT
        )

    if freq_minutes is None:

        freq_minutes = (
            Config.FREQ_MINUTES
        )

    if cache_path is None:

        cache_path = (
            Config.EXPANDING_RAW_CACHE_FILE
        )

    # ========================================================
    # DISCOVER FILES
    # ========================================================

    entries = discover_nc_files(
        data_dir
    )

    if not entries:

        raise ValueError(
            f"Tidak ada file subset_*.nc "
            f"ditemukan di {data_dir}"
        )

    # ========================================================
    # SMOKE TEST
    # ========================================================

    if max_files is not None:

        entries = entries[
            :max_files
        ]

        logging.info(
            "Mode smoke-test: hanya "
            f"pakai {len(entries)} file pertama "
            f"(dari max_files={max_files})."
        )

        if not entries:

            raise ValueError(
                f"--max-files={max_files} "
                f"menghasilkan 0 file."
            )

    say_info(
        f"Total file .nc yang akan dibaca: "
        f"{len(entries)}"
    )

    # ========================================================
    # TIMELINE
    # ========================================================

    timeline = (
        build_uniform_timeline(
            entries,
            freq_minutes=freq_minutes,
        )
    )

    # ========================================================
    # LOAD TEMPORAL MATRIX
    # ========================================================

    data_matrix, pixel_meta = (
        load_pixel_grid(
            entries,
            timeline,
            n_workers=n_workers,
        )
    )

    # ========================================================
    # SAVE RAW CACHE
    # ========================================================

    if cache_path:

        save_raw_cache(
            cache_path,
            data_matrix,
            timeline,
            pixel_meta,
        )

        say_info(
            f"Cache raw time series disimpan ke: "
            f"{cache_path}"
        )

    # ========================================================
    # BUILD TRAINING SAMPLES
    # ========================================================

    all_samples = []

    total_anchors = 0

    n_pixels = (
        data_matrix.shape[1]
    )

    pixel_progress = (
        make_progress_bar(
            range(n_pixels),
            desc="Proses pixel",
            unit="pixel",
        )
    )

    for p in pixel_progress:

        y = data_matrix[:, p]

        valid_mask = ~np.isnan(y)

        anchors = (
            find_valid_anchors(
                valid_mask,
                span=ANCHOR_SPAN,
                stride=anchor_stride,
            )
        )

        total_anchors += len(
            anchors
        )

        pixel_progress.set_postfix_str(
            f"anchor={total_anchors}"
        )

        row = pixel_meta.iloc[p]

        df_pixel = (
            build_pixel_samples(
                y,
                anchors,
                row["pixel_id"],
                row["latitude"],
                row["longitude"],
                timeline,
            )
        )

        if df_pixel is not None:

            all_samples.append(
                df_pixel
            )

    # ========================================================
    # VALIDATE
    # ========================================================

    if not all_samples:

        raise ValueError(
            "Tidak ada anchor valid "
            "di pixel manapun -- cek "
            "kelengkapan data."
        )

    # ========================================================
    # CONCAT DATASET
    # ========================================================

    dataset = pd.concat(
        all_samples,
        ignore_index=True
    )

    ordered_cols = (
        [
            "pixel_id",
            "latitude",
            "longitude",
            "anchor_t0",
            "step",
            "target_time",
        ]
        +
        FEATURE_COLUMNS
        +
        [
            TARGET_COLUMN
        ]
    )

    dataset = dataset[
        ordered_cols
    ]

    # ========================================================
    # SAVE CSV
    # ========================================================

    if output_path:

        os.makedirs(
            os.path.dirname(
                output_path
            ),
            exist_ok=True
        )

        dataset.to_csv(
            output_path,
            index=False
        )

    return dataset