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
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import xarray as xr
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

# Ukuran window: IS1 = 6 titik (indeks anchor .. anchor+5), IS18 = 23 titik.
# Total rentang yang dibutuhkan satu anchor (dari titik pertama window
# sampai target OS18) = 24 titik berturut-turut tanpa gap. Lihat CLAUDE.md §2 & §4.
MIN_WINDOW_SIZE = Config.MIN_WINDOW_SIZE
HORIZON_STEPS = Config.HORIZON_STEPS
# ANCHOR_SPAN disentralkan ke Config (satu sumber kebenaran) karena sekarang
# juga dipakai model_training.py buat purge/embargo stratified_monthly_split()
# -- lihat Config.ANCHOR_SPAN / Config.PURGE_STEPS di config.py.
ANCHOR_SPAN = Config.ANCHOR_SPAN  # = MIN_WINDOW_SIZE + HORIZON_STEPS = 24

# Offset dari `anchor` (start) ke `end` window IS1 (step=1), relatif ke
# titik pertama window. IS1 punya MIN_WINDOW_SIZE titik (indeks 0-based
# anchor..anchor+MIN_WINDOW_SIZE-1), jadi end IS1 = anchor + MIN_WINDOW_SIZE - 1.
# Untuk step k, window tumbuh (k-1) titik dari IS1, jadi:
#   end(step=k) = anchor + (MIN_WINDOW_SIZE - 1) + (k - 1)
#               = anchor + (MIN_WINDOW_SIZE - 2) + k
# WINDOW_OFFSET = MIN_WINDOW_SIZE - 2 dipakai di build_pixel_samples().
# SEBELUMNYA ini hardcoded jadi angka 4 (yang cuma kebetulan benar untuk
# MIN_WINDOW_SIZE=6 default) -- kalau MIN_WINDOW_SIZE diubah di config.py,
# offset di sini WAJIB ikut berubah otomatis, makanya diturunkan eksplisit.
WINDOW_OFFSET = MIN_WINDOW_SIZE - 2


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


def build_uniform_timeline(entries, freq_minutes=None):
    """Bangun index waktu uniform per `freq_minutes` dari timestamp
    pertama sampai terakhir yang ditemukan. Timestamp yang nggak punya file
    (gap file hilang) tetap ada slot-nya di timeline ini, cuma nanti nilainya
    NaN di semua pixel.

    freq_minutes : int, optional
        Default None -> pakai `Config.FREQ_MINUTES` (10, sesuai resolusi
        Himawari). Diturunkan dari config, bukan hardcoded, supaya konsisten
        kalau sumber data berubah resolusi.
    """
    if freq_minutes is None:
        freq_minutes = Config.FREQ_MINUTES
    t_min = entries[0][0]
    t_max = entries[-1][0]
    timeline = pd.date_range(t_min, t_max, freq=f"{freq_minutes}min")
    return timeline


def _read_netcdf_worker(path, target_channel, canonical_shape):
    """Worker TOP-LEVEL (bukan closure) dipanggil di proses terpisah lewat
    ProcessPoolExecutor untuk baca SATU file NetCDF. Harus di top-level
    module supaya bisa di-pickle dan dikirim ke worker process lain.

    KENAPA ProcessPoolExecutor, BUKAN ThreadPoolExecutor: library HDF5/netCDF4
    C di balik `xr.open_dataset()` pada build default TIDAK thread-safe
    (thread-safety HDF5 cuma aktif kalau library-nya dikompilasi eksplisit
    dengan --enable-threadsafe, jarang jadi default paket binary). Baca
    paralel via banyak thread Python di satu proses BERISIKO crash/korupsi
    diam-diam. Proses terpisah aman karena tiap worker punya instance
    HDF5/netCDF4 sendiri, tidak share state C library apapun.

    Return
    ------
    (values_flat, status) -- values_flat: np.ndarray 1D hasil ravel() kalau
    sukses, None kalau gagal. status: None (sukses), "missing_channel",
    "shape_mismatch", atau "error:<pesan>" (exception apapun saat baca/buka).
    """
    try:
        with xr.open_dataset(path) as ds:
            if target_channel not in ds.variables:
                return None, "missing_channel"
            values = ds[target_channel].values
            if canonical_shape is not None and values.shape != canonical_shape:
                return None, "shape_mismatch"
            return values.ravel(), None
    except Exception as e:
        return None, f"error:{e}"


def load_pixel_grid(entries, timeline, n_workers=None):
    """Baca semua file NetCDF, ekstrak nilai tbb_13 tiap pixel, susun jadi
    matrix (T, P) di mana T = panjang timeline, P = jumlah pixel (flatten
    grid lat x lon).

    DIBACA PARALEL (ProcessPoolExecutor, lihat _read_netcdf_worker untuk
    alasan proses bukan thread) -- ini bottleneck I/O paling berat di
    seluruh Tahap 2 (12.5 jam untuk 34.420 file kalau sekuensial murni,
    lihat CLAUDE.md §10 poin 4; bottleneck-nya di overhead buka/parse tiap
    file, BUKAN compute, jadi paralelisasi lintas proses langsung nyerang
    akar masalahnya).

    Alur 2 fase:
    1. SEKUENSIAL, cepat: scan entries satu-satu sampai ketemu file valid
       PERTAMA (punya target_channel, berhasil dibuka) -- dipakai nentuin
       canonical_shape & grid lat/lon, yang dibutuhkan SEBELUM data_matrix
       bisa dialokasikan dan sebelum file lain bisa divalidasi shape-nya.
       File yang di-skip di fase ini (missing channel / gagal baca) TIDAK
       dibaca ulang di fase 2.
    2. PARALEL: sisa file (setelah file valid pertama) dibaca lintas proses
       via ProcessPoolExecutor.map(). Hasil di-assign balik ke data_matrix
       di proses utama pakai t_idx -- aman tanpa lock karena tiap file
       nulis ke baris berbeda (t_idx unik per timestamp).

    Parameters
    ----------
    n_workers : int, optional
        Jumlah proses paralel. Default None -> Config.NETCDF_READ_WORKERS.
        n_workers <= 1 -> fallback sekuensial murni (skip overhead spawn
        ProcessPoolExecutor sama sekali) -- berguna untuk smoke-test file
        sedikit atau debugging.

    Return
    ------
    data_matrix : np.ndarray, shape (T, P), NaN = gap (file hilang, pixel
        cloud-masked, atau shape file menyimpang dari grid canonical).
    pixel_meta : pd.DataFrame, shape (P, ...), kolom lat_idx/lon_idx/latitude/
        longitude untuk tiap pixel (urutan sejajar dengan kolom data_matrix).
    """
    if n_workers is None:
        n_workers = Config.NETCDF_READ_WORKERS

    timeline_index = {ts: i for i, ts in enumerate(timeline)}
    T = len(timeline)

    # --- Fase 1 (sekuensial): cari file valid PERTAMA buat nentuin
    # canonical_shape & grid lat/lon. WAJIB sekuensial -- semua file lain
    # butuh nilai ini buat dialokasikan/divalidasi. Biasanya cuma butuh
    # baca 1 file (kecuali file-file awal korup/nggak punya channel), jadi
    # murah dibanding total keseluruhan.
    canonical_shape = None
    canonical_lat = None
    canonical_lon = None
    data_matrix = None
    n_mismatched = 0
    n_missing_channel = 0
    n_errors = 0
    first_valid_pos = None  # posisi di `entries`, BUKAN t_idx

    for pos, (ts, path) in enumerate(entries):
        t_idx = timeline_index.get(ts)
        if t_idx is None:
            continue
        try:
            with xr.open_dataset(path) as ds:
                if TARGET_CHANNEL not in ds.variables:
                    n_missing_channel += 1
                    continue

                values = ds[TARGET_CHANNEL].values
                canonical_shape = values.shape
                # latitude/longitude di dataset adalah koordinat 1D (panjang
                # = jumlah baris/kolom grid), bukan grid penuh. Meshgrid dulu
                # supaya sejajar 1:1 dengan pixel hasil ravel() dari `values`
                # (row-major: lat berubah lambat, lon berubah cepat).
                lat_1d = ds["latitude"].values
                lon_1d = ds["longitude"].values
                lat_mesh, lon_mesh = np.meshgrid(lat_1d, lon_1d, indexing="ij")
                canonical_lat = lat_mesh.ravel()
                canonical_lon = lon_mesh.ravel()
                P = int(np.prod(canonical_shape))
                data_matrix = np.full((T, P), np.nan, dtype=np.float64)
                data_matrix[t_idx, :] = values.ravel()
                first_valid_pos = pos
        except Exception as e:
            n_errors += 1
            logging.warning(f"Gagal baca {path}: {e}")
            continue
        # Hanya sampai sini kalau try-block sukses TANPA `continue` di
        # tengah (channel ketemu, data_matrix ke-set) -- baru berhenti scan.
        break

    if data_matrix is None:
        raise ValueError(
            f"Tidak ada satupun file yang punya variabel '{TARGET_CHANNEL}' -- "
            "cek nama channel atau isi data_bandung/."
        )

    # --- Fase 2 (paralel): sisa file setelah file valid pertama. ---
    remaining = []
    for pos in range(first_valid_pos + 1, len(entries)):
        ts, path = entries[pos]
        t_idx = timeline_index.get(ts)
        if t_idx is None:
            continue
        remaining.append((t_idx, path))

    if remaining:
        if n_workers <= 1:
            # Fallback sekuensial murni -- tanpa overhead spawn proses.
            progress = make_progress_bar(remaining, desc="Baca NetCDF", unit="file")
            for t_idx, path in progress:
                values_flat, status = _read_netcdf_worker(path, TARGET_CHANNEL, canonical_shape)
                if status == "missing_channel":
                    n_missing_channel += 1
                elif status == "shape_mismatch":
                    n_mismatched += 1
                elif status is not None:
                    n_errors += 1
                    logging.warning(f"Gagal baca {path}: {status}")
                else:
                    data_matrix[t_idx, :] = values_flat
        else:
            # Chunksize: task-nya kecil (buka 1 file) tapi jumlahnya bisa
            # puluhan ribu -- submit satu future per file itu mahal karena
            # overhead IPC (pickle/unpickle) per-task. Kelompokkan beberapa
            # file per chunk supaya overhead itu diamortisasi.
            chunksize = max(1, len(remaining) // (n_workers * 4))
            say_info(
                f"Baca NetCDF paralel: {len(remaining)} file tersisa, "
                f"{n_workers} proses, chunksize={chunksize}"
            )
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                results_iter = executor.map(
                    _read_netcdf_worker,
                    [path for _t_idx, path in remaining],
                    [TARGET_CHANNEL] * len(remaining),
                    [canonical_shape] * len(remaining),
                    chunksize=chunksize,
                )
                progress = tqdm(
                    zip(remaining, results_iter),
                    total=len(remaining),
                    desc="Baca NetCDF (paralel)",
                    bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} file [{elapsed}<{remaining}]",
                    ncols=80,
                )
                for (t_idx, path), (values_flat, status) in progress:
                    if status == "missing_channel":
                        n_missing_channel += 1
                    elif status == "shape_mismatch":
                        n_mismatched += 1
                    elif status is not None:
                        n_errors += 1
                        logging.warning(f"Gagal baca {path}: {status}")
                    else:
                        data_matrix[t_idx, :] = values_flat

    if n_mismatched > 0:
        logging.warning(
            f"{n_mismatched} file punya shape grid beda dari canonical "
            f"{canonical_shape}, dianggap gap penuh."
        )
    if n_missing_channel > 0:
        logging.warning(
            f"{n_missing_channel} file tidak punya variabel '{TARGET_CHANNEL}', dilewati."
        )
    if n_errors > 0:
        say_error(
            f"{n_errors} file gagal dibaca (exception saat buka/parse) -- "
            "lihat log warning untuk detail per file, dianggap gap."
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
    # ends = a + WINDOW_OFFSET + k, untuk k=1..HORIZON_STEPS (k dari `steps`,
    # di-tile per anchor). WINDOW_OFFSET diturunkan dari Config.MIN_WINDOW_SIZE
    # (lihat definisi di atas) -- BUKAN angka hardcoded lagi.
    ends = starts + WINDOW_OFFSET + np.tile(steps, len(anchors))
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


def save_raw_cache(cache_path, data_matrix, timeline, pixel_meta):
    """Simpan data_matrix + timeline + pixel_meta ke .npz supaya tahap lain
    (04_recursive_evaluate.py) bisa pakai nilai mentah tbb_13 tanpa baca
    ulang ribuan file .nc (I/O paling berat di pipeline ini -- lihat catatan
    performa sesi ini, ~12.5 jam untuk 34rb file).

    timeline (DatetimeIndex) disimpan sebagai int64 nanoseconds supaya aman
    di npz (npz tidak native support datetime64), dikonversi balik pas load.
    pixel_meta (DataFrame kecil, <=beberapa ratus baris) disimpan per-kolom.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    # PENTING: pakai np.array(..., dtype="<U32") eksplisit, BUKAN
    # arr.astype(str). Di numpy>=2.0, .astype(str) pada array object bisa
    # menghasilkan StringDType (variable-length) yang disimpan sebagai
    # object array di .npz -- gagal di-load dengan allow_pickle=False.
    # Fixed-width unicode ("<U32") aman untuk npz tanpa pickle.
    pixel_id_arr = np.array(pixel_meta["pixel_id"].values, dtype="<U32")
    np.savez_compressed(
        cache_path,
        data_matrix=data_matrix.astype(np.float32),  # float32 cukup untuk TBB (Kelvin), hemat ~separuh ukuran file
        timeline_ns=timeline.values.astype("datetime64[ns]").astype(np.int64),
        pixel_id=pixel_id_arr,
        lat_idx=pixel_meta["lat_idx"].values,
        lon_idx=pixel_meta["lon_idx"].values,
        latitude=pixel_meta["latitude"].values,
        longitude=pixel_meta["longitude"].values,
    )


def load_raw_cache(cache_path):
    """Kebalikan dari save_raw_cache(). Return (data_matrix, timeline, pixel_meta)
    dengan tipe data sama seperti yang di-return load_pixel_grid()/build_uniform_timeline()."""
    npz = np.load(cache_path, allow_pickle=False)
    data_matrix = npz["data_matrix"].astype(np.float64)
    timeline = pd.to_datetime(npz["timeline_ns"])
    pixel_meta = pd.DataFrame({
        "pixel_id": npz["pixel_id"].astype("<U32").astype(str),
        "lat_idx": npz["lat_idx"],
        "lon_idx": npz["lon_idx"],
        "latitude": npz["latitude"],
        "longitude": npz["longitude"],
    })
    return data_matrix, timeline, pixel_meta


def build_dataset(data_dir=None, output_path=None, anchor_stride=None, freq_minutes=None,
                   max_files=None, cache_path=None, n_workers=None):
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
    freq_minutes : int, optional
        Resolusi timeline. Default None -> pakai `Config.FREQ_MINUTES`
        (10 menit, sesuai Himawari).
    max_files : int, optional
        Kalau diisi, cuma pakai `max_files` file PERTAMA secara kronologis
        (bukan random) -- buat smoke-test cepat di subset kecil sebelum
        full run, sesuai kebiasaan kerja CLAUDE.md §11 (`03a_build_features.py`
        di repo lama makan 10 jam+ karena langsung full-run tanpa validasi
        subset dulu). Default None = pakai semua file (perilaku produksi).
        CATATAN: karena dipotong secara kronologis dari awal, anchor dekat
        ujung potongan bakal ke-skip otomatis oleh gap-skip rule (§4) --
        ini WAJAR untuk smoke-test, bukan bug.
    cache_path : str, optional
        Kalau diisi (atau default None -> Config.EXPANDING_RAW_CACHE_FILE),
        simpan data_matrix + timeline + pixel_meta mentah ke .npz di path
        ini setelah selesai baca NetCDF, dipakai 04_recursive_evaluate.py.
        Pass False eksplisit untuk skip caching (mis. smoke-test cepat yang
        nggak perlu cache).
    n_workers : int, optional
        Jumlah proses paralel untuk baca file NetCDF (lihat load_pixel_grid).
        Default None -> Config.NETCDF_READ_WORKERS. Set 1 untuk fallback
        sekuensial murni.

    Return
    ------
    pd.DataFrame gabungan semua pixel, kolom: pixel_id, latitude, longitude,
    anchor_t0, step, target_time, 9 kolom fitur (FEATURE_COLUMNS), target_tbb_13.
    """
    if data_dir is None:
        data_dir = Config.FINAL_BASE_DIR
    if anchor_stride is None:
        anchor_stride = Config.ANCHOR_STRIDE_DEFAULT
    if freq_minutes is None:
        freq_minutes = Config.FREQ_MINUTES
    if cache_path is None:
        cache_path = Config.EXPANDING_RAW_CACHE_FILE

    entries = discover_nc_files(data_dir)
    if not entries:
        raise ValueError(f"Tidak ada file subset_*.nc ditemukan di {data_dir}")

    if max_files is not None:
        entries = entries[:max_files]
        logging.info(f"Mode smoke-test: hanya pakai {len(entries)} file pertama (dari max_files={max_files}).")
        if not entries:
            raise ValueError(f"--max-files={max_files} menghasilkan 0 file untuk diproses (butuh >= 1).")

    say_info(f"Total file .nc yang akan dibaca: {len(entries)}")

    timeline = build_uniform_timeline(entries, freq_minutes=freq_minutes)
    data_matrix, pixel_meta = load_pixel_grid(entries, timeline, n_workers=n_workers)

    if cache_path:
        save_raw_cache(cache_path, data_matrix, timeline, pixel_meta)
        say_info(f"Cache raw time series disimpan ke: {cache_path}")

    all_samples = []
    total_anchors = 0
    n_pixels = data_matrix.shape[1]
    pixel_progress = make_progress_bar(range(n_pixels), desc="Proses pixel", unit="pixel")
    for p in pixel_progress:
        y = data_matrix[:, p]
        valid_mask = ~np.isnan(y)
        anchors = find_valid_anchors(valid_mask, span=ANCHOR_SPAN, stride=anchor_stride)
        total_anchors += len(anchors)
        pixel_progress.set_postfix_str(f"anchor={total_anchors}")

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