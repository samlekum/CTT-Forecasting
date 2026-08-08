# ./scripts/pipeline/visualize.py
# Tahap 7: render hasil forecast Stage 05 (CSV di forecast_output/) jadi
# animasi GIF -- 6 panel PETA (semua spatial plot pakai lat/lon asli,
# TIDAK ADA panel teks polos) per step: Input@t0, Prediksi, Aktual, Kelas
# Awan, Risiko Banjir, Error Map. Metadata (t0/target_time/model/MAE/dll)
# jadi suptitle figure, bukan panel tersendiri -- keputusan desain
# eksplisit dari sesi diskusi (lihat CLAUDE.md §20).
#
# Sumbernya CSV forecast produksi (pipeline/inference.py::CSV_COLUMNS),
# BUKAN hasil evaluasi Stage 04. Skala warna (TBB & MAE) dihitung ULANG
# PER STEP (bukan global lintas 18 step) -- vmax tiap frame selalu =
# nilai tertinggi yang benar2 muncul di step itu sendiri, biar tiap
# frame pakai kontras/dynamic range penuh (trade-off: warna tidak lagi
# apple-to-apple lintas step, lihat compute_value_range()/compute_mae_range()).

import os
import re
import sys
import json
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless -- script CLI, tidak ada display interaktif
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FormatStrFormatter
from PIL import Image

from pipeline.config import Config
from pipeline.inference import CSV_COLUMNS, load_raw_values_lookup
from ui.terminal_display import say_info, say_ok, say_error, banner, make_progress_bar, _c

FREQ_MINUTES = Config.FREQ_MINUTES

# Regex nama FOLDER (bukan nama file lagi -- Stage 05 sekarang simpan
# forecast.csv/forecast.geojson dalam 1 folder per run, pola sama persis
# dgn folder output Stage 06/visualizations/, lihat CLAUDE.md §21). Folder
# hasil Stage 05 versi LAMA (file flat forecast_{model}_t0..._run....csv
# langsung di forecast_output/, dari sebelum restrukturisasi ini) otomatis
# TIDAK match apa pun di sini (bukan folder), jadi otomatis ke-skip di
# discover_forecast_files() -- sama filosofi "pola lama diabaikan diam2"
# yang sudah ada sejak fitur t0-di-nama ditambahkan.
FORECAST_FOLDER_RE = re.compile(
    r"^(?P<model>[A-Za-z0-9]+)_t0(?P<t0>\d{8}_\d{4})_run(?P<run>\d{8}_\d{4})$"
)

# --- Palet warna (skill dataviz, lihat references/palette.md) --------------
# Sequential blue (hue default), 13 step persis dari palette.md, dipakai
# utk panel kontinu Input/Prediksi/Aktual. DIBALIK (reversed) dari
# orientasi biasa skill ("terang=dekat nol") -- utk TBB, nilai RENDAH itu
# yang berbahaya/bermakna (puncak awan tinggi/dingin -> konveksi kuat),
# jadi harus menonjol GELAP, bukan memudar ke background seperti asumsi
# skill "nol = tidak penting" yang tidak berlaku di domain TBB.
_TBB_BLUE_STEPS = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
    "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
TBB_CMAP = LinearSegmentedColormap.from_list("tbb_blue_r", list(reversed(_TBB_BLUE_STEPS)))

# Sequential kedua (slot kategorikal-2 = orange) utk Error Map -- palette.md
# cuma kasih 1 hex orange (bukan ramp 13-step spt biru), jadi ramp
# sintetis 2-stop dari surface terang ke oranye itu. Orientasi NORMAL
# (0 error = terang/pudar, error besar = pekat) -- beda dari TBB, 0 error
# memang berarti "tidak ada apa-apa" secara semantik.
MAE_CMAP = LinearSegmentedColormap.from_list("error_orange", ["#fcfcfb", "#eb6834"])

# Status palette (fixed, never themed -- palette.md) dipakai utk KEDUA
# panel kategorikal (Kelas Awan & Risiko Banjir) -- breakpoint SAMA persis
# (classify_tbb_grid), jadi warna otomatis konsisten antar 2 panel itu.
STATUS_COLORS = ("#0ca30c", "#fab219", "#d03b3b")  # good/warning/critical

NODATA_COLOR = "#c8c7c2"     # abu netral -- pixel hilang dari CSV / nilai NaN
BOUNDARY_COLOR = "#52514e"   # secondary ink -- garis batas kecamatan
TEXT_SECONDARY = "#52514e"


# ============================================================================
# Discovery & CLI selection
# ============================================================================

def discover_forecast_files(forecast_dir=None):
    """Scan forecast_dir, kembalikan HANYA folder berpola BARU Stage 05
    (`{model}_t0{...}_run{...}/`, lihat FORECAST_FOLDER_RE) yang beneran
    punya `forecast.csv` di dalamnya. File FLAT lama (hasil Stage 05
    SEBELUM restrukturisasi jadi folder-per-run -- `forecast_{model}_t0
    {...}_run{...}.csv` langsung di forecast_dir) diabaikan diam-diam
    (satu baris ringkasan jumlahnya, bukan error/warning per-file --
    riwayat lama TETAP ADA di disk, cuma tidak lagi muncul di menu).

    Return
    ------
    pd.DataFrame kolom: filename (nama folder, dipakai buat label
    tampilan), path (path LENGKAP ke forecast.csv di dalam folder itu),
    model_name, t0_str, run_str, t0_ts, run_ts -- terurut run_ts
    DESCENDING (run terbaru duluan).
    """
    if forecast_dir is None:
        forecast_dir = Config.INFERENCE_DIR
    if not os.path.isdir(forecast_dir):
        raise FileNotFoundError(
            f"Folder forecast tidak ditemukan: {forecast_dir}\n"
            "Jalankan 05_run_inference.py dulu."
        )

    rows = []
    n_old_pattern = 0
    for entry in sorted(os.listdir(forecast_dir)):
        entry_path = os.path.join(forecast_dir, entry)
        if not os.path.isdir(entry_path):
            if entry.startswith("forecast_") and entry.endswith(".csv"):
                n_old_pattern += 1
            continue
        m = FORECAST_FOLDER_RE.match(entry)
        if not m:
            continue
        csv_path = os.path.join(entry_path, "forecast.csv")
        if not os.path.exists(csv_path):
            continue  # folder cocok pola tapi forecast.csv tidak ada -- skip diam2, bukan crash
        rows.append({
            "filename": entry,
            "path": csv_path,
            "model_name": m.group("model"),
            "t0_str": m.group("t0"),
            "run_str": m.group("run"),
            "t0_ts": pd.to_datetime(m.group("t0"), format="%Y%m%d_%H%M"),
            "run_ts": pd.to_datetime(m.group("run"), format="%Y%m%d_%H%M"),
        })

    if n_old_pattern:
        say_info(
            f"{n_old_pattern} file forecast_*.csv pola LAMA (flat, dari sebelum Stage 05 "
            "direstruktur jadi folder-per-run) diabaikan -- tetap ada di disk, cuma tidak "
            "muncul di menu ini lagi."
        )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("run_ts", ascending=False).reset_index(drop=True)
    return df


def _numbered_input_select(labels):
    """Fallback menu (ketik angka + Enter) -- dipakai kalau terminal
    TIDAK interaktif (stdin/stdout bukan TTY, mis. dijalankan non-
    interaktif/piped/CI) atau bukan Windows (menu panah pakai `msvcrt`,
    stdlib khusus Windows -- lihat _interactive_menu_select())."""
    for i, label in enumerate(labels):
        say_info(f"[{i + 1}] {label}")
    n = len(labels)
    while True:
        raw = input(f"Pilih nomor (1-{n}): ").strip()
        if raw.isdigit() and 1 <= int(raw) <= n:
            return int(raw) - 1
        say_error(f"Input tidak valid: '{raw}'. Masukkan angka 1-{n}.")


def _interactive_menu_select(labels):
    """Menu navigasi panah atas/bawah / PageUp/PageDown, Enter buat pilih
    -- dipakai buat pilih file forecast tanpa ketik angka (diminta Dhika).
    Windows-only (`msvcrt`, stdlib bawaan Windows, TANPA dependency baru)
    -- fallback otomatis ke `_numbered_input_select()` kalau bukan Windows
    atau stdin/stdout bukan TTY interaktif (mis. dijalankan lewat pipe).

    Return
    ------
    int -- index (0-based) label yang dipilih.
    """
    if os.name != "nt" or not sys.stdin.isatty() or not sys.stdout.isatty():
        return _numbered_input_select(labels)

    import msvcrt

    idx = 0
    n = len(labels)

    def render(first=False):
        if not first:
            sys.stdout.write(f"\033[{n}A")  # cursor naik n baris, redraw di tempat sama
        for i, label in enumerate(labels):
            marker = "> " if i == idx else "  "
            line = f"{marker}{label}"
            if i == idx:
                line = _c("1;36", line)
            sys.stdout.write("\033[K" + line + "\n")  # \033[K = clear sisa baris lama
        sys.stdout.flush()

    say_info("Navigasi: panah atas/bawah atau PageUp/PageDown, Enter buat pilih.")
    render(first=True)
    while True:
        ch = msvcrt.getch()
        if ch in (b"\xe0", b"\x00"):  # prefix byte tombol khusus (panah/PgUp/PgDn)
            ch2 = msvcrt.getch()
            if ch2 == b"H":       # Up
                idx = (idx - 1) % n
            elif ch2 == b"P":     # Down
                idx = (idx + 1) % n
            elif ch2 == b"I":     # Page Up
                idx = max(0, idx - 5)
            elif ch2 == b"Q":     # Page Down
                idx = min(n - 1, idx + 5)
            else:
                continue
            render()
        elif ch in (b"\r", b"\n"):  # Enter
            sys.stdout.write("\n")
            sys.stdout.flush()
            return idx
        elif ch in (b"\x1b", b"\x03"):  # Esc / Ctrl+C -- batalkan
            sys.stdout.write("\n")
            sys.stdout.flush()
            raise KeyboardInterrupt("Dibatalkan user.")
        # tombol lain diabaikan


def prompt_select_forecast_file(candidates_df):
    """Pilih 1 baris dari discover_forecast_files(). 0 hasil -> raise
    ValueError (caller/06_visualize.py tangkap & say_error, berhenti
    otomatis, TIDAK traceback, TIDAK nunggu input). 1 hasil -> langsung
    dipakai, skip prompt (tapi tetap say_info file mana yang dipakai --
    transparansi). >1 hasil -> menu navigasi panah/PageUp-PageDown + Enter
    (lihat _interactive_menu_select(), fallback ketik-angka kalau tidak
    memungkinkan), terurut run_ts DESCENDING (run terbaru di atas)."""
    if candidates_df.empty:
        raise ValueError(
            "Tidak ada file forecast berpola baru "
            "(forecast_{model}_t0{...}_run{...}.csv) ditemukan. "
            "Jalankan 05_run_inference.py dulu."
        )
    if len(candidates_df) == 1:
        row = candidates_df.iloc[0]
        say_info(f"Hanya 1 hasil forecast ditemukan, langsung dipakai: {row['filename']}")
        return row

    banner("PILIH HASIL FORECAST")
    labels = [
        f"model={row['model_name']}  t0={row['t0_ts']}  run={row['run_ts']}  ({row['filename']})"
        for _, row in candidates_df.iterrows()
    ]
    idx = _interactive_menu_select(labels)
    selected = candidates_df.iloc[idx]
    say_ok(f"Dipilih: {selected['filename']}")
    return selected


# ============================================================================
# Load & validasi
# ============================================================================

def load_forecast_csv(csv_path):
    """Baca CSV forecast Stage 05, validasi skema kolom cocok
    inference.CSV_COLUMNS (reuse, bukan duplikasi daftar kolom).

    Ditolak SEBELUM dilempar ke render pipeline kalau 0 baris (skema
    kolom valid tapi kosong -- bisa kejadian kalau Stage 05 gagal
    forecast SEMUA pixel, mis. 0/35 krn window semuanya invalid/gap) --
    tanpa guard ini, error baru muncul jauh di dalam (`iloc[0]` di
    `visualize_forecast()`, atau `steps[0]`/`frames[0]` di
    `render_all_frames()`/`assemble_gif()`) sbg IndexError mentah yang
    TIDAK ketangkep `except (FileNotFoundError, ValueError)` di
    `06_visualize.py` -- user lihat traceback penuh, bukan pesan jelas.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File CSV forecast tidak ditemukan: {csv_path}")
    df = pd.read_csv(csv_path)
    missing_cols = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"CSV forecast {csv_path} kehilangan kolom {missing_cols} -- pastikan ini "
            "hasil 05_run_inference.py versi terbaru."
        )
    if df.empty:
        raise ValueError(
            f"CSV forecast {csv_path} kosong (0 baris) -- kemungkinan Stage 05 gagal "
            "forecast semua pixel (0/35). Tidak ada yang bisa divisualisasikan."
        )
    df["anchor_t0"] = pd.to_datetime(df["anchor_t0"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    return df


def warn_missing_pixels(detail_df, grid_shape=None):
    """Cek 35 pixel lengkap atau tidak. Set pixel yang di-skip Stage 05
    SAMA di semua step (ditentukan sekali oleh select_anchor_per_pixel()
    sebelum rollout mulai) -- cukup dicek sekali di sini, bukan per-frame."""
    if grid_shape is None:
        grid_shape = Config.PIXEL_GRID_SHAPE
    n_lat, n_lon = grid_shape
    expected = {f"{i}_{j}" for i in range(n_lat) for j in range(n_lon)}
    present = set(detail_df["pixel_id"].unique())
    missing = expected - present
    if missing:
        say_info(
            f"PERINGATAN: {len(missing)}/{len(expected)} pixel tidak ada di CSV forecast "
            f"(di-skip Stage 05, lihat select_anchor_per_pixel()): {sorted(missing)}"
        )
    return missing


# ============================================================================
# "Input @ t0" -- observasi asli TEPAT SEBELUM rollout mulai (BUKAN ada di
# CSV Stage 05, dibaca ulang dari data_bandung/ lewat load_raw_values_lookup()
# yang di-reuse dari pipeline/inference.py)
# ============================================================================

def compute_t0_per_pixel(detail_df):
    """window_end_time (=t0, titik observasi TERAKHIR sebelum forecast
    mulai) per pixel = target_time(step=1) - FREQ_MINUTES -- aritmatika
    sederhana dari kolom target_time yang SUDAH ADA di CSV (lihat
    inference.py: target_time(step=1) = anchor_t0 + MIN_WINDOW_SIZE*FREQ_MINUTES,
    dan window_end_time = anchor_t0 + (MIN_WINDOW_SIZE-1)*FREQ_MINUTES,
    jadi window_end_time = target_time(step=1) - FREQ_MINUTES).

    Dihitung PER ROW (bukan diasumsikan sama semua pixel) -- robust kalau
    ada pixel yang window-nya lebih stale (gap) drpd pixel lain.

    Return
    ------
    pd.Series, index=pixel_id, value=t0 (pd.Timestamp) pixel itu.
    """
    step1 = detail_df[detail_df["step"] == 1][["pixel_id", "target_time"]].copy()
    step1["t0"] = step1["target_time"] - pd.Timedelta(minutes=FREQ_MINUTES)
    return step1.set_index("pixel_id")["t0"]


def load_input_snapshot(detail_df, data_dir=None, n_workers=None):
    """Baca nilai tbb_13 ASLI di titik t0 (window_end_time) per pixel dari
    data_bandung/ -- reuse inference.load_raw_values_lookup() (fungsi
    INTI yang sama dipakai Stage 05 utk y_true, cuma rentang waktunya
    beda: t0 per pixel, bukan target_time forecast).

    Return
    ------
    dict {pixel_id: float} -- cuma berisi pixel yang file .nc-nya ADA di
    t0-nya (seharusnya SELALU ada, karena Stage 05 cuma generate forecast
    dari window yang sudah divalidasi valid) & nilainya bukan NaN.
    """
    if data_dir is None:
        data_dir = Config.FINAL_BASE_DIR

    t0_per_pixel = compute_t0_per_pixel(detail_df)
    if t0_per_pixel.empty:
        return {}

    raw_lookup = load_raw_values_lookup(
        data_dir, t0_per_pixel.min(), t0_per_pixel.max(), n_workers=n_workers
    )

    input_lookup = {}
    for pixel_id, t0 in t0_per_pixel.items():
        val = raw_lookup.get((pixel_id, pd.Timestamp(t0)))
        if val is not None:
            input_lookup[pixel_id] = val

    say_info(f"Data input @ t0 ketemu untuk {len(input_lookup)}/{len(t0_per_pixel)} pixel.")
    return input_lookup


# ============================================================================
# Reshape & klasifikasi (fungsi murni, gampang diuji terisolasi)
# ============================================================================

def pixel_rows_to_grid(step_df, value_col, grid_shape=None):
    """Reshape long-format (1 baris per pixel_id) jadi array 2D siap
    imshow. lat_idx -> axis 0 (baris), lon_idx -> axis 1 (kolom) --
    KONSISTEN dgn urutan meshgrid(indexing="ij") di
    dataset_builder.load_pixel_grid(), supaya orientasi peta match urutan
    pixel_id/lat_idx/lon_idx di semua tahap lain.

    Pixel yang HILANG dari step_df (di-skip Stage 05) -> NaN, BUKAN 0 (0
    Kelvin salah tafsir data-wise) dan BUKAN crash.
    """
    if grid_shape is None:
        grid_shape = Config.PIXEL_GRID_SHAPE
    grid = np.full(grid_shape, np.nan, dtype=float)
    for _, row in step_df.iterrows():
        lat_i, lon_i = int(row["lat_idx"]), int(row["lon_idx"])
        if not (0 <= lat_i < grid_shape[0] and 0 <= lon_i < grid_shape[1]):
            raise ValueError(
                f"lat_idx/lon_idx ({lat_i},{lon_i}) di luar Config.PIXEL_GRID_SHAPE "
                f"{grid_shape} -- config mungkin tidak sesuai lagi dgn domain data grid."
            )
        val = row[value_col]
        grid[lat_i, lon_i] = val if pd.notna(val) else np.nan
    return grid


def pixel_rows_to_grid_from_lookup(step_df, lookup, grid_shape=None):
    """Sama seperti pixel_rows_to_grid(), tapi nilai diambil dari dict
    lookup {pixel_id: value} (dipakai utk panel Input, yang nilainya
    BUKAN kolom di step_df -- lihat load_input_snapshot())."""
    if grid_shape is None:
        grid_shape = Config.PIXEL_GRID_SHAPE
    grid = np.full(grid_shape, np.nan, dtype=float)
    for _, row in step_df.iterrows():
        lat_i, lon_i = int(row["lat_idx"]), int(row["lon_idx"])
        val = lookup.get(row["pixel_id"])
        grid[lat_i, lon_i] = val if val is not None else np.nan
    return grid


def classify_tbb_grid(value_grid, thresholds=None):
    """Klasifikasi grid TBB jadi 3 kode kategorikal {0,1,2}, breakpoint
    SAMA PERSIS dipakai panel Kelas Awan & Risiko Banjir -- cuma label
    beda (dilewatkan terpisah saat render, BUKAN 2 fungsi klasifikasi
    terpisah). Catatan domain: TBB RENDAH = puncak awan tinggi/dingin =
    konveksi kuat = hujan lebat/risiko banjir (BAHAYA), TBB TINGGI =
    langit cerah/awan tipis mendekati suhu permukaan (AMAN) -- arah ini
    SEMPAT KEBALIK di versi awal (breakpoint sama, tapi code 0/2 tertukar,
    lihat CLAUDE.md §20.13), ketahuan krn semua forecast asli (~276-300K,
    kondisi cerah) selalu ke-klasifikasi "bahaya" yang harusnya "aman".

    thresholds default Config.TBB_RISK_THRESHOLDS = (200.0, 270.0):
      code 2 : value <  lo        (bahaya / hujan)
      code 1 : lo <= value <= hi  (waspada / mendung)
      code 0 : value >  hi        (aman / tidak hujan)

    NaN di value_grid -> NaN di output (propagate, BUKAN default ke kelas
    tertentu yang bisa disalahartikan mis. "aman").
    """
    if thresholds is None:
        thresholds = Config.TBB_RISK_THRESHOLDS
    lo, hi = thresholds
    code_grid = np.full(value_grid.shape, np.nan, dtype=float)
    valid = ~np.isnan(value_grid)
    code_grid[valid & (value_grid < lo)] = 2.0
    code_grid[valid & (value_grid >= lo) & (value_grid <= hi)] = 1.0
    code_grid[valid & (value_grid > hi)] = 0.0
    return code_grid


# ============================================================================
# Geometri peta (extent, aspect ratio) & overlay batas kecamatan
# ============================================================================

def compute_grid_geometry(grid_shape=None):
    """Hitung extent [lon_min,lon_max,lat_min,lat_max] (dgn padding
    setengah-sel, supaya sel grid align rapi sbg "cell" bukan cuma titik)
    & latitude tengah (utk koreksi aspect ratio) dari BOUNDING BOX
    DOWNLOAD tetap (Config.BANDUNG_LAT_MIN/MAX, LON_MIN/MAX) -- BUKAN
    dari lat/lon pixel yang HADIR di CSV, supaya extent peta konsisten
    walau ada pixel tepi yang di-skip Stage 05 (kalau dihitung dari data
    yang hadir, grid bisa "menyusut" diam-diam kalau kebetulan pixel tepi
    hilang).

    Return
    ------
    (extent, center_lat) -- extent: list [lon_min,lon_max,lat_min,lat_max]
    siap dilempar ke imshow(extent=...).
    """
    if grid_shape is None:
        grid_shape = Config.PIXEL_GRID_SHAPE
    n_lat, n_lon = grid_shape
    lat_min, lat_max = Config.BANDUNG_LAT_MIN, Config.BANDUNG_LAT_MAX
    lon_min, lon_max = Config.BANDUNG_LON_MIN, Config.BANDUNG_LON_MAX
    lat_step = (lat_max - lat_min) / (n_lat - 1) if n_lat > 1 else 0.0
    lon_step = (lon_max - lon_min) / (n_lon - 1) if n_lon > 1 else 0.0
    extent = [lon_min - lon_step / 2, lon_max + lon_step / 2,
              lat_min - lat_step / 2, lat_max + lat_step / 2]
    center_lat = (lat_min + lat_max) / 2
    return extent, center_lat


def load_kecamatan_boundaries(geojson_path=None):
    """Parse scripts/geojson/KotaBandung.geojson pakai stdlib json (BUKAN
    geopandas -- hindari dependency berat baru). Ekstrak ring LUAR tiap
    polygon (abaikan lubang -- cukup buat garis referensi visual, bukan
    analisis geometri presisi). Sama CRS dgn pixel grid (lat/lon derajat
    WGS84 polos), jadi align langsung tanpa transformasi.

    Gagal parse/file tidak ada -> say_info + return [] (overlay di-skip,
    BUKAN crash -- ini fitur pelengkap, bukan wajib).

    Return
    ------
    list of list[(lon, lat), ...] -- satu ring per elemen list.
    """
    if geojson_path is None:
        geojson_path = Config.KOTA_BANDUNG_GEOJSON
    if not os.path.exists(geojson_path):
        say_info(f"File batas kecamatan tidak ditemukan ({geojson_path}) -- overlay dilewati.")
        return []
    try:
        with open(geojson_path, encoding="utf-8") as f:
            geo = json.load(f)
        rings = []
        for feat in geo.get("features", []):
            geom = feat.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            if gtype == "Polygon":
                polys = [coords]
            elif gtype == "MultiPolygon":
                polys = coords
            else:
                continue
            for poly in polys:
                if poly:
                    rings.append(poly[0])  # ring luar (index 0) saja
        return rings
    except Exception as e:
        say_info(f"Gagal parse {geojson_path} ({e}) -- overlay batas kecamatan dilewati.")
        return []


def _draw_boundaries(ax, boundaries):
    if not boundaries:
        return
    for ring in boundaries:
        lons = [p[0] for p in ring]
        lats = [p[1] for p in ring]
        ax.plot(lons, lats, color=BOUNDARY_COLOR, linewidth=0.6, alpha=0.7, zorder=5)


# ============================================================================
# Mask pixel di luar batas administratif Kota Bandung (opsional, murni
# layer visualisasi -- TIDAK menyentuh data_bandung/, dataset training,
# atau CSV forecast Stage 05 sama sekali)
# ============================================================================

def compute_boundary_mask(boundaries, grid_shape=None, n_samples=7):
    """True = sel pixel (bukan cuma titik tengahnya) beririsan dgn UNION
    seluruh polygon kecamatan -- dites dgn sampling grid n_samples x
    n_samples titik di dalam cell (bukan cuma 1 titik tengah, supaya
    pixel yang SEBAGIAN masuk wilayah kota tetap ketahuan, bukan cuma
    yang titik tengahnya persis di dalam polygon -- dgn spacing grid
    ~0.05 derajat vs kota yang sempit/tidak beraturan, tes titik-tengah-
    saja terlalu agresif, cuma 6/35 pixel lolos vs 13/35 dgn cell-overlap).

    boundaries kosong ([], gagal load geojson) -> return all-True (TIDAK
    masking apa pun, biar tetap tampil semua pixel spt sebelum fitur ini
    ada -- fail-open, bukan fail-closed, karena ini fitur pelengkap).
    """
    if grid_shape is None:
        grid_shape = Config.PIXEL_GRID_SHAPE
    if not boundaries:
        return np.ones(grid_shape, dtype=bool)

    paths = [MplPath(ring) for ring in boundaries]
    n_lat, n_lon = grid_shape
    lat_min, lat_max = Config.BANDUNG_LAT_MIN, Config.BANDUNG_LAT_MAX
    lon_min, lon_max = Config.BANDUNG_LON_MIN, Config.BANDUNG_LON_MAX
    lat_step = (lat_max - lat_min) / (n_lat - 1) if n_lat > 1 else 0.0
    lon_step = (lon_max - lon_min) / (n_lon - 1) if n_lon > 1 else 0.0

    mask = np.zeros(grid_shape, dtype=bool)
    for lat_idx in range(n_lat):
        lat_c = lat_max - lat_idx * lat_step
        lats = np.linspace(lat_c - lat_step / 2, lat_c + lat_step / 2, n_samples)
        for lon_idx in range(n_lon):
            lon_c = lon_min + lon_idx * lon_step
            lons = np.linspace(lon_c - lon_step / 2, lon_c + lon_step / 2, n_samples)
            pts = np.array([(lo, la) for la in lats for lo in lons])
            mask[lat_idx, lon_idx] = any(p.contains_points(pts).any() for p in paths)
    return mask


def apply_boundary_mask(grid, boundary_mask):
    """Set NaN di posisi yang di luar boundary_mask -- pixel itu jadi
    render abu-abu (NODATA_COLOR) via jalur masked_invalid yang sudah
    ada, TANPA kode render terpisah."""
    if boundary_mask is None:
        return grid
    masked = grid.copy()
    masked[~boundary_mask] = np.nan
    return masked


def boundary_mask_to_pixel_ids(boundary_mask):
    """Konversi grid boolean (lat_idx,lon_idx) -> set pixel_id string
    "{lat_idx}_{lon_idx}" yang TERMASUK (True) -- dipakai buat filter
    detail_df/input_lookup sebelum hitung value_range/mae_range GLOBAL,
    supaya skala warna tidak lagi dipengaruhi pixel yang di-mask."""
    n_lat, n_lon = boundary_mask.shape
    return {
        f"{lat_idx}_{lon_idx}"
        for lat_idx in range(n_lat)
        for lon_idx in range(n_lon)
        if boundary_mask[lat_idx, lon_idx]
    }


# ============================================================================
# Render 1 panel
# ============================================================================

def render_map_panel(ax, grid, extent, vmin, vmax, title, cmap, cbar_label,
                      boundaries=None, placeholder_text=None, aspect=None):
    """Render 1 panel peta kontinu (Input/Prediksi/Aktual/Error). Kalau
    grid TIDAK punya nilai valid SAMA SEKALI (np.all(isnan)) DAN
    placeholder_text diisi -- render TEKS placeholder DI ATAS frame peta
    kosong (tetap extent/aspect/boundary sama, supaya "tetap map" sesuai
    requirement -- BUKAN axis blank tanpa konteks geografis), bukan
    imshow kosong yang ambigu (bisa disalahartikan "semua nilai rendah").
    """
    if aspect is not None:
        ax.set_aspect(aspect)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    _draw_boundaries(ax, boundaries)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)

    if placeholder_text is not None and np.all(np.isnan(grid)):
        ax.text(0.5, 0.5, placeholder_text, ha="center", va="center",
                 transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, wrap=True)
        return None

    masked = np.ma.masked_invalid(grid)
    cmap_obj = cmap.copy()
    cmap_obj.set_bad(NODATA_COLOR)
    im = ax.imshow(masked, extent=extent, vmin=vmin, vmax=vmax, cmap=cmap_obj,
                    origin="upper", aspect="auto" if aspect is not None else None)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    return im


def render_categorical_map_panel(ax, code_grid, extent, class_labels, colors, title,
                                  boundaries=None, aspect=None):
    """Render 1 panel peta kategorikal (Kelas Awan / Risiko Banjir) --
    generik, terima label & warna sbg parameter (dipanggil 2x dari
    code_grid yang SAMA, cuma label/warna beda -- lihat classify_tbb_grid()).
    Legend selalu ada (icon+label, bukan warna doang -- syarat skill
    dataviz utk status color)."""
    if aspect is not None:
        ax.set_aspect(aspect)

    cmap_obj = ListedColormap(list(colors))
    cmap_obj.set_bad(NODATA_COLOR)
    masked = np.ma.masked_invalid(code_grid)
    ax.imshow(masked, extent=extent, vmin=-0.5, vmax=len(class_labels) - 0.5,
              cmap=cmap_obj, origin="upper", aspect="auto" if aspect is not None else None)
    _draw_boundaries(ax, boundaries)
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Longitude", fontsize=8)
    ax.set_ylabel("Latitude", fontsize=8)
    ax.tick_params(labelsize=7)

    patches = [mpatches.Patch(color=colors[i], label=class_labels[i]) for i in range(len(class_labels))]
    patches.append(mpatches.Patch(color=NODATA_COLOR, label="tidak ada data"))
    ax.legend(handles=patches, loc="upper right", fontsize=6.5, framealpha=0.85)


# ============================================================================
# Render 1 figure (6 panel) & seluruh frame
# ============================================================================

def build_step_figure(step_df, grid_input, t0_repr, step, horizon_steps, extent, aspect,
                       value_range, mae_range, boundaries, model_name, damping_factor,
                       boundary_mask=None, figsize=None):
    """1 figure, layout 2x3, SEMUA 6 panel berupa peta (lihat header modul
    utk tata letak final). Metadata jadi suptitle, bukan panel.

    boundary_mask (opsional): grid boolean (lat_idx,lon_idx), True=pixel
    dipertahankan. Diterapkan ke SEMUA grid (termasuk grid_input yang
    sudah di-mask sebelum masuk sini) supaya pixel di luar batas Kota
    Bandung konsisten jadi abu-abu (NODATA_COLOR) di ke-6 panel -- lihat
    compute_boundary_mask()."""
    if figsize is None:
        figsize = (16, 10)

    fig, axes = plt.subplots(2, 3, figsize=figsize, dpi=110)

    grid_pred = pixel_rows_to_grid(step_df, "y_pred")
    grid_true = pixel_rows_to_grid(step_df, "y_true")
    grid_mae = pixel_rows_to_grid(step_df, "abs_error")
    if boundary_mask is not None:
        grid_pred = apply_boundary_mask(grid_pred, boundary_mask)
        grid_true = apply_boundary_mask(grid_true, boundary_mask)
        grid_mae = apply_boundary_mask(grid_mae, boundary_mask)

    vmin, vmax = value_range
    render_map_panel(axes[0, 0], grid_input, extent, vmin, vmax, "Input",
                      TBB_CMAP, "TBB (K)", boundaries, placeholder_text="Data input t0\ntidak ditemukan", aspect=aspect)
    render_map_panel(axes[0, 1], grid_pred, extent, vmin, vmax, "Prediksi",
                      TBB_CMAP, "TBB (K)", boundaries, aspect=aspect)
    render_map_panel(axes[0, 2], grid_true, extent, vmin, vmax, "Aktual",
                      TBB_CMAP, "TBB (K)", boundaries,
                      placeholder_text="Tidak ada observasi asli\n(target_time belum terjadi/didownload)",
                      aspect=aspect)

    code_grid = classify_tbb_grid(grid_pred)
    render_categorical_map_panel(axes[1, 0], code_grid, extent, ("tidak hujan", "mendung", "hujan"),
                                  STATUS_COLORS, "Kelas Awan", boundaries, aspect=aspect)
    render_categorical_map_panel(axes[1, 1], code_grid, extent, ("aman", "waspada", "bahaya"),
                                  STATUS_COLORS, "Risiko Banjir", boundaries, aspect=aspect)

    mae_vmin, mae_vmax = mae_range if mae_range is not None else (0.0, 1.0)
    render_map_panel(axes[1, 2], grid_mae, extent, mae_vmin, mae_vmax, "Error Map",
                      MAE_CMAP, "MAE (K)", boundaries,
                      placeholder_text="Tidak ada observasi asli\n-> error tidak bisa dihitung",
                      aspect=aspect)

    target_time = step_df["target_time"].iloc[0]
    step_abs_error = step_df["abs_error"].dropna()
    mae_step = step_abs_error.mean() if not step_abs_error.empty else None
    valid_codes = code_grid[~np.isnan(code_grid)]
    n_bahaya = int(np.sum(valid_codes == 2))
    n_waspada = int(np.sum(valid_codes == 1))
    n_aman = int(np.sum(valid_codes == 0))

    subtitle = (
        f"Model: {model_name} (damping={damping_factor:.2f})  |  t0: {t0_repr}  |  "
        f"target_time: {target_time}  |  step {step:02d}/{horizon_steps:02d}\n"
        f"MAE step ini: {'%.2fK' % mae_step if mae_step is not None else 'n/a (tidak ada observasi asli)'}"
        f"  |  Status: {n_aman} aman, {n_waspada} waspada, {n_bahaya} bahaya"
    )
    fig.suptitle(subtitle, fontsize=11, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


def render_all_frames(detail_df, input_lookup, frames_dir, model_name, damping_factor,
                       mask_outside_bandung=False):
    """Render 1 PNG per step (loop step 1..HORIZON_STEPS), simpan ke
    frames_dir. `plt.close(fig)` WAJIB tiap step -- 18 figure per run
    kalau tidak ditutup bisa leak memori.

    mask_outside_bandung (default False -- tampilkan semua 35 pixel,
    dipilih Dhika stlh coba versi ter-mask): kalau True, pixel yang
    cell-nya TIDAK beririsan dgn batas administratif Kota Bandung
    (compute_boundary_mask()) di-mask NaN (abu-abu) di ke-6 panel &
    DIKELUARKAN dari perhitungan vmin/vmax global -- murni tampilan,
    TIDAK mengubah detail_df/CSV sumber. Opt-in via `--mask-outside-bandung`
    di CLI kalau nanti dibutuhkan lagi."""
    os.makedirs(frames_dir, exist_ok=True)
    extent, center_lat = compute_grid_geometry()
    aspect = 1.0 / max(np.cos(np.radians(center_lat)), 1e-6)
    boundaries = load_kecamatan_boundaries()

    boundary_mask = None
    if mask_outside_bandung:
        boundary_mask = compute_boundary_mask(boundaries)
        n_kept = int(boundary_mask.sum())
        say_info(
            f"Mask di luar batas Kota Bandung aktif: {n_kept}/{boundary_mask.size} pixel "
            "dipertahankan (cell pixel beririsan dgn polygon kecamatan), sisanya jadi abu-abu."
        )

    t0_per_pixel = compute_t0_per_pixel(detail_df)
    t0_repr = t0_per_pixel.max() if not t0_per_pixel.empty else None

    steps = sorted(detail_df["step"].unique())
    horizon_steps = len(steps)

    first_step_df = detail_df[detail_df["step"] == steps[0]]
    grid_input = pixel_rows_to_grid_from_lookup(first_step_df, input_lookup)
    if boundary_mask is not None:
        grid_input = apply_boundary_mask(grid_input, boundary_mask)

    frame_paths = []
    for step in make_progress_bar(steps, desc="Render frame", unit="step"):
        step_df = detail_df[detail_df["step"] == step]
        # vmin/vmax dihitung ULANG per step (bukan global lintas 18 step)
        # -- warna paling pekat/tinggi di tiap frame selalu merepresentasikan
        # nilai tertinggi PADA STEP ITU SENDIRI (TBB & MAE), sesuai diminta
        # Dhika. Trade-off: warna TIDAK lagi apple-to-apple lintas step
        # (mis. "biru gelap" di step 1 & step 18 bisa berarti suhu beda),
        # tapi tiap frame individual memakai kontras/dynamic range penuh.
        value_range = compute_value_range(step_df, input_lookup, boundary_mask)
        mae_range = compute_mae_range(step_df, boundary_mask)
        fig = build_step_figure(
            step_df, grid_input, t0_repr, int(step), horizon_steps, extent, aspect,
            value_range, mae_range, boundaries, model_name, damping_factor,
            boundary_mask=boundary_mask,
        )
        frame_path = os.path.join(frames_dir, f"step{int(step):02d}.png")
        fig.savefig(frame_path, dpi=110)
        plt.close(fig)
        frame_paths.append(frame_path)

    return frame_paths


def compute_value_range(step_df, input_lookup, boundary_mask=None):
    """vmin/vmax gabungan Input + y_pred + y_true UNTUK 1 STEP (dipanggil
    ULANG tiap step dari render_all_frames(), BUKAN sekali secara global)
    -- vmax selalu = nilai TBB tertinggi yang benar2 muncul di step itu
    (Input+Prediksi+Aktual), supaya warna paling pekat di tiap frame
    merepresentasikan nilai tertinggi pada step tsb sendiri.

    boundary_mask (opsional): kalau diisi, pixel di luar mask DIKELUARKAN
    dari perhitungan range -- supaya skala warna tidak lagi dipengaruhi
    nilai pixel yang toh akan di-mask abu-abu saat render."""
    detail_df = step_df
    if boundary_mask is not None:
        inside_ids = boundary_mask_to_pixel_ids(boundary_mask)
        detail_df = detail_df[detail_df["pixel_id"].isin(inside_ids)]
        input_lookup = {k: v for k, v in input_lookup.items() if k in inside_ids}

    values = [v for v in detail_df["y_pred"].values if not np.isnan(v)]
    values.extend(v for v in detail_df["y_true"].values if not np.isnan(v))
    values.extend(v for v in input_lookup.values() if v is not None and not np.isnan(v))
    if not values:
        return (0.0, 1.0)  # fallback ekstrem, seharusnya tidak pernah kejadian (y_pred selalu ada)
    return (float(min(values)), float(max(values)))


def compute_mae_range(step_df, boundary_mask=None):
    """vmax MAE UNTUK 1 STEP (0 selalu jadi vmin), dipanggil ULANG tiap
    step -- vmax = abs_error tertinggi yang benar2 muncul di step itu
    sendiri (bukan max lintas 18 step). None kalau abs_error NaN semua
    di step ini (mis. belum ada observasi asli) -- caller pakai fallback
    (0,1) yang tidak akan pernah benar2 dipakai buat imshow (step itu
    lewat jalur placeholder kalau ini None). boundary_mask: lihat
    compute_value_range()."""
    detail_df = step_df
    if boundary_mask is not None:
        inside_ids = boundary_mask_to_pixel_ids(boundary_mask)
        detail_df = detail_df[detail_df["pixel_id"].isin(inside_ids)]

    values = [v for v in detail_df["abs_error"].values if not np.isnan(v)]
    if not values:
        return None
    return (0.0, float(max(values)))


# ============================================================================
# GIF assembly
# ============================================================================

def assemble_gif(frame_paths, gif_path, duration_ms=None, loop=0):
    """Gabung frame_paths (urut step 1..N) jadi 1 GIF via Pillow. GIF
    otomatis kuantisasi ke 256 warna (palette-based) -- untuk heatmap
    kontinu bisa muncul sedikit banding, tapi ini dampak yang bisa
    diterima untuk tool diagnostic internal (bukan output publikasi)."""
    if duration_ms is None:
        duration_ms = Config.VISUALIZATION_FRAME_DURATION_MS
    frames = [Image.open(p).convert("RGB") for p in frame_paths]
    frames[0].save(gif_path, save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=loop, optimize=False)
    for im in frames:
        im.close()
    say_ok(f"GIF disimpan ke: {gif_path}")
    return gif_path


# ============================================================================
# Struktur folder output
# ============================================================================

def build_output_paths(model_name, t0_str, run_str, output_dir=None):
    """visualizations/{model}_t0{t0str}_run{runstr}/frames/step{01-N}.png
    + .../animation.gif. `t0_str`/`run_str` diambil LANGSUNG dari nama
    file CSV sumber (regex), bukan dihitung ulang -- identitas folder =
    identitas file sumber persis.

    SENGAJA idempotent (deterministik dari CSV sumber, BEDA dari Stage 05
    yang selalu unique tiap run krn data baru) -- run 06_visualize.py 2x
    pada CSV yang sama akan OVERWRITE folder yang sama, bukan bikin folder
    baru. Kalau nanti mau tiap render immutable, tinggal tambah suffix
    `_render{timestamp}` -- sengaja tidak dilakukan skrg, lihat CLAUDE.md §20.
    """
    if output_dir is None:
        output_dir = Config.VISUALIZATION_DIR
    run_dir = os.path.join(output_dir, f"{model_name}_t0{t0_str}_run{run_str}")
    frames_dir = os.path.join(run_dir, "frames")
    gif_path = os.path.join(run_dir, "animation.gif")
    os.makedirs(frames_dir, exist_ok=True)
    return run_dir, frames_dir, gif_path


# ============================================================================
# Orchestrator
# ============================================================================

def visualize_forecast(csv_path=None, forecast_dir=None, output_dir=None, frame_duration_ms=None,
                        mask_outside_bandung=False):
    """Fungsi utama Tahap 7: pilih CSV forecast (menu/override) -> load &
    validasi -> cari input snapshot @ t0 -> render 18 frame -> gabung GIF.

    mask_outside_bandung (default False): lihat render_all_frames().

    Return
    ------
    dict berisi: csv_path, model_name, t0_str, run_str, run_dir,
    frames_dir, gif_path, n_frames, missing_pixels, has_actual_values.
    """
    if csv_path is None:
        candidates = discover_forecast_files(forecast_dir)
        selected = prompt_select_forecast_file(candidates)
        csv_path = selected["path"]
        model_name = selected["model_name"]
        t0_str = selected["t0_str"]
        run_str = selected["run_str"]
    else:
        # Model/t0/run sekarang diparse dari NAMA FOLDER INDUK
        # (forecast.csv sendiri namanya generic, identitas ada di nama
        # folder -- lihat FORECAST_FOLDER_RE / CLAUDE.md §21).
        folder_name = os.path.basename(os.path.dirname(csv_path))
        m = FORECAST_FOLDER_RE.match(folder_name)
        if not m:
            raise ValueError(
                f"Nama folder induk '{folder_name}' tidak cocok pola {{model}}_t0{{...}}_run{{...}} "
                "-- tidak bisa diparse otomatis. Pastikan --csv menunjuk ke forecast.csv di dalam "
                "folder output 05_run_inference.py (forecast_output/{model}_t0..._run.../forecast.csv)."
            )
        model_name = m.group("model")
        t0_str = m.group("t0")
        run_str = m.group("run")

    say_info(f"CSV forecast: {csv_path}")
    detail_df = load_forecast_csv(csv_path)
    missing_pixels = warn_missing_pixels(detail_df)

    damping_factor = float(detail_df["damping_factor"].iloc[0])
    has_actual_values = bool(detail_df["y_true"].notna().any())

    say_info("Cari data input @ t0 dari data_bandung/...")
    input_lookup = load_input_snapshot(detail_df)

    run_dir, frames_dir, gif_path = build_output_paths(model_name, t0_str, run_str, output_dir)
    frame_paths = render_all_frames(
        detail_df, input_lookup, frames_dir, model_name, damping_factor,
        mask_outside_bandung=mask_outside_bandung,
    )
    assemble_gif(frame_paths, gif_path, duration_ms=frame_duration_ms)

    return {
        "csv_path": csv_path,
        "model_name": model_name,
        "t0_str": t0_str,
        "run_str": run_str,
        "run_dir": run_dir,
        "frames_dir": frames_dir,
        "gif_path": gif_path,
        "n_frames": len(frame_paths),
        "missing_pixels": missing_pixels,
        "has_actual_values": has_actual_values,
    }
