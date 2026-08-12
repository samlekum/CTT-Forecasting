# ./scripts/pipeline/config.py
#
# Konfigurasi terpusat untuk seluruh pipeline CTT Forecasting.
#
# Perubahan desain eksperimen 2026-08:
# - Stage 02 menghasilkan temporal cache (.npz)
# - Stage 03 menggunakan chronological train/test split
# - Train : Desember 2025 - Mei 2026
# - Test  : Juni 2026 - Juli 2026
# - Tidak menggunakan purge/embargo untuk split utama
# - Window tidak lagi dikunci ke satu nilai.
#   Kandidat window diuji per model pada tahap eksperimen.
#
# Catatan:
# Konfigurasi lama yang masih diperlukan oleh download/inference/
# visualization tetap dipertahankan agar pipeline lama tidak rusak.

import os

from dotenv import load_dotenv


# ============================================================================
# PROJECT ROOT
# ============================================================================

SCRIPT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(SCRIPT_DIR)
)

load_dotenv(
    os.path.join(
        PROJECT_ROOT,
        ".env",
    )
)


class Config:
    """Konfigurasi terpusat untuk pipeline."""

    PROJECT_ROOT = PROJECT_ROOT

    # =========================================================================
    # FTP / DOWNLOAD
    # =========================================================================

    FTP_HOST = os.environ.get(
        "FTP_HOST"
    )

    FTP_USER = os.environ.get(
        "FTP_USER"
    )

    FTP_PASS = os.environ.get(
        "FTP_PASS"
    )

    FTP_REMOTE_BASE = os.environ.get(
        "FTP_REMOTE_DIR"
    )

    FTP_TIMEOUT = 60

    # =========================================================================
    # TELEGRAM
    # =========================================================================

    TELEGRAM_BOT_TOKEN = os.environ.get(
        "TELEGRAM_BOT_TOKEN"
    )

    TELEGRAM_CHAT_ID = os.environ.get(
        "TELEGRAM_CHAT_ID"
    )

    # =========================================================================
    # DOWNLOAD FILTER
    # =========================================================================

    FILENAME_MUST_CONTAIN = os.environ.get(
        "FILENAME_MUST_CONTAIN",
        "R21_FLDK.02801_02401",
    ).strip()

    # =========================================================================
    # PATHS
    # =========================================================================

    TEMP_FILE = os.path.join(
        PROJECT_ROOT,
        "temp_download.nc",
    )

    FINAL_BASE_DIR = os.path.join(
        PROJECT_ROOT,
        "data_bandung",
    )

    METADATA_BASE_DIR = os.path.join(
        PROJECT_ROOT,
        "_metadata",
    )

    LOG_FILE = os.path.join(
        PROJECT_ROOT,
        "download_activity.log",
    )

    # =========================================================================
    # RETRY
    # =========================================================================

    MAX_RETRY = 3

    # =========================================================================
    # BANDUNG SUBSET
    # =========================================================================

    BANDUNG_LAT_MIN = -7.0
    BANDUNG_LAT_MAX = -6.8

    BANDUNG_LON_MIN = 107.5
    BANDUNG_LON_MAX = 107.8

    # =========================================================================
    # HIMAWARI CHANNELS
    # =========================================================================

    TBB_CHANNELS = [
        f"tbb_{i:02d}"
        for i in range(7, 17)
    ]

    # =========================================================================
    # MACHINE LEARNING MODELS
    # =========================================================================
    #
    # Model yang digunakan dalam eksperimen:
    #   XGBoost
    #   LightGBM
    #   CatBoost
    #
    # Ubah daftar ini di satu tempat agar training/evaluasi konsisten.

    MODEL_NAMES = [
        "xgboost",
        "lightgbm",
        "catboost",
    ]

    # =========================================================================
    # LEGACY INTERVAL CONFIGURATION
    # =========================================================================
    #
    # Dipertahankan untuk kompatibilitas dengan script lama.
    #
    # Pipeline eksperimen temporal baru menggunakan timeline 10 menit
    # dan horizon 18 step.

    INTERVALS_MINUTES = [
        10,
        30,
        60,
    ]

    # =========================================================================
    # LEGACY LAG CONFIGURATION
    # =========================================================================
    #
    # Ini bukan konfigurasi utama untuk eksperimen temporal baru.
    # Dipertahankan agar modul lama yang masih menggunakan LAG_COUNT
    # tidak langsung rusak.
    #
    # Stage 02 baru TIDAK membangun CSV lag dari konfigurasi ini.

    LAG_COUNT = 6

    assert LAG_COUNT >= 3, (
        "LAG_COUNT minimal 3 untuk kompatibilitas "
        "dengan fitur delta/acceleration pada pipeline lama."
    )

    # =========================================================================
    # EXPERIMENTAL TEMPORAL CONFIGURATION
    # =========================================================================

    # -------------------------------------------------------------------------
    # Target
    # -------------------------------------------------------------------------

    # Target utama adalah Brightness Temperature channel 13.
    TARGET_CHANNEL = "tbb_13"

    # -------------------------------------------------------------------------
    # Temporal resolution
    # -------------------------------------------------------------------------

    # Himawari data yang digunakan memiliki interval 10 menit.
    FREQ_MINUTES = 10

    # -------------------------------------------------------------------------
    # Forecast horizon
    # -------------------------------------------------------------------------
    #
    # 18 step × 10 menit = 180 menit = 3 jam.

    HORIZON_STEPS = 18

    assert HORIZON_STEPS >= 1, (
        "HORIZON_STEPS minimal 1."
    )

    # -------------------------------------------------------------------------
    # Candidate observation windows
    # -------------------------------------------------------------------------
    #
    # Sebelumnya seluruh model dipaksa menggunakan:
    #
    #     window = 6
    #
    # Sekarang window menjadi hyperparameter eksperimen.
    #
    # Setiap model akan diuji dengan kandidat window yang sama:
    #
    #     3 step  = 30 menit
    #     6 step  = 60 menit
    #     12 step = 120 menit
    #     18 step = 180 menit
    #
    # Window terbaik ditentukan berdasarkan performa validasi/training
    # yang sesuai dengan desain eksperimen.
    #
    # Nilai ini TIDAK berarti semua window dipakai bersamaan.

    WINDOW_CANDIDATES = [
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18,
    ]

    assert len(WINDOW_CANDIDATES) > 0, (
        "WINDOW_CANDIDATES tidak boleh kosong."
    )

    assert len(
        set(WINDOW_CANDIDATES)
    ) == len(WINDOW_CANDIDATES), (
        "WINDOW_CANDIDATES tidak boleh memiliki duplikat."
    )

    assert all(
        isinstance(w, int) and w >= 1
        for w in WINDOW_CANDIDATES
    ), (
        "Setiap window harus integer >= 1."
    )

    assert HORIZON_STEPS >= max(
        WINDOW_CANDIDATES
    ) or HORIZON_STEPS > 0, (
        "Konfigurasi horizon/window tidak valid."
    )

    # -------------------------------------------------------------------------
    # Chronological train/test split
    # -------------------------------------------------------------------------
    #
    # Sesuai desain eksperimen berdasarkan PPT:
    #
    # TRAIN:
    #   2025-12-01 -> 2026-05-31
    #
    # TEST:
    #   2026-06-01 -> 2026-07-31
    #
    # Tidak menggunakan stratified monthly split.
    # Tidak menggunakan purge.
    # Tidak menggunakan embargo.
    #
    # Alasan metodologis:
    # seluruh informasi temporal yang digunakan model berasal dari periode
    # sebelum test. Test benar-benar merupakan periode masa depan relatif
    # terhadap training.

    TRAIN_START = "2025-12-01 00:00:00"
    TRAIN_END = "2026-05-31 23:59:59"

    TEST_START = "2026-06-01 00:00:00"
    TEST_END = "2026-07-31 23:59:59"

    # -------------------------------------------------------------------------
    # Temporal split output
    # -------------------------------------------------------------------------

    TEMPORAL_SPLIT_DIR = os.path.join(
        PROJECT_ROOT,
        "dataset",
        "temporal_split",
    )

    TEMPORAL_TRAIN_FILE = os.path.join(
        TEMPORAL_SPLIT_DIR,
        "train_temporal.npz",
    )

    TEMPORAL_TEST_FILE = os.path.join(
        TEMPORAL_SPLIT_DIR,
        "test_temporal.npz",
    )

    TEMPORAL_SPLIT_METADATA_FILE = os.path.join(
        TEMPORAL_SPLIT_DIR,
        "split_metadata.json",
    )

    # -------------------------------------------------------------------------
    # Window search: fit / validation split (SUBSET dari TRAIN)
    # -------------------------------------------------------------------------
    #
    # Window terbaik per model dipilih pakai potongan EKOR dari TRAIN
    # (bukan TEST asli) -- TEST (Jun-Jul) sama sekali tidak disentuh
    # sampai evaluasi akhir, biar pemilihan window tidak bias/leak.
    #
    # FIT       : 2025-12-01 -> 2026-03-31 (dilatih single-step per window)
    # VALIDATION: 2026-04-01 -> 2026-05-31 (dievaluasi recursive rollout
    #             18 step, MAE rata-rata semua step per model per window)

    WINDOW_SEARCH_FIT_END = "2026-03-31 23:59:59"
    WINDOW_SEARCH_VAL_START = "2026-04-01 00:00:00"
    WINDOW_SEARCH_VAL_END = "2026-05-31 23:59:59"

    # Subsample anchor validation (ambil 1 dari N anchor secara berurutan)
    # supaya rollout evaluation tidak usah pakai SEMUA anchor -- MAE rata-
    # rata tetap representatif tanpa perlu itung jutaan anchor x horizon.
    WINDOW_SEARCH_ANCHOR_STRIDE = 6

    WINDOW_SEARCH_DIR = os.path.join(
        PROJECT_ROOT,
        "window_search",
    )

    WINDOW_SEARCH_RESULTS_FILE = os.path.join(
        WINDOW_SEARCH_DIR,
        "window_search_results.csv",
    )

    WINDOW_SEARCH_BEST_FILE = os.path.join(
        WINDOW_SEARCH_DIR,
        "best_window_per_model.csv",
    )

    # =========================================================================
    # LEGACY EXPANDING DATASET PATHS
    # =========================================================================
    #
    # Dipertahankan sementara karena beberapa script lama masih merujuk
    # ke path ini.
    #
    # Dataset CSV expanding window TIDAK lagi menjadi output Stage 02 baru.

    EXPANDING_DATASET_DIR = os.path.join(
        PROJECT_ROOT,
        "dataset",
    )

    EXPANDING_DATASET_FILE = os.path.join(
        EXPANDING_DATASET_DIR,
        "expanding_features.csv",
    )

    # =========================================================================
    # TEMPORAL RAW CACHE
    # =========================================================================
    #
    # Output utama Stage 02:
    #
    #   data_matrix
    #   timeline_ns
    #   pixel_id
    #   lat_idx
    #   lon_idx
    #   latitude
    #   longitude
    #
    # Shape aktual full run lu:
    #
    #   (34989, 35)
    #
    # Cache ini digunakan agar tahap berikutnya tidak perlu membaca
    # 34.420 file NetCDF lagi.

    EXPANDING_RAW_CACHE_FILE = os.path.join(
        EXPANDING_DATASET_DIR,
        "expanding_raw_cache.npz",
    )

    # =========================================================================
    # EXPANDING / DATASET LEGACY COMPATIBILITY
    # =========================================================================
    #
    # Jangan digunakan oleh chronological experiment baru.
    #
    # Dibiarkan sebagai compatibility layer karena dataset_builder.py
    # lama masih mengakses parameter ini.
    #
    # Setelah seluruh pipeline baru selesai dirombak, parameter ini dapat
    # dihapus bersama dataset_builder.py legacy.

    ANCHOR_STRIDE_DEFAULT = 1

    # -------------------------------------------------------------------------
    # LEGACY MODEL DATASET PARAMETERS
    # -------------------------------------------------------------------------

    # TEST_FRAC tidak digunakan pada chronological split baru.
    #
    # Tetap dipertahankan agar script lama tidak error jika masih mengakses
    # Config.TEST_FRAC.

    TEST_FRAC = 0.15

    # -------------------------------------------------------------------------
    # Legacy anchor values
    # -------------------------------------------------------------------------
    #
    # JANGAN digunakan oleh eksperimen baru.
    #
    # Hanya compatibility sementara dengan dataset_builder.py dan modul
    # lama yang belum kita hapus.
    #
    # Penting:
    # nilai ini TIDAK menentukan WINDOW_CANDIDATES.

    LEGACY_MIN_WINDOW_SIZE = 6

    # Tidak lagi digunakan untuk split baru.
    #
    # Sengaja TIDAK membuat:
    #
    #     ANCHOR_SPAN
    #     PURGE_STEPS
    #
    # sebagai parameter eksperimen baru.
    #
    # Kalau modul legacy masih mencoba mengakses Config.MIN_WINDOW_SIZE
    # atau Config.PURGE_STEPS, modul tersebut harus diperbaiki/dihapus
    # pada tahap refactor berikutnya.

    # =========================================================================
    # RAW NETCDF READING
    # =========================================================================

    # Jumlah worker untuk Stage 02.
    #
    # os.cpu_count() - 1 dipakai agar satu core tetap tersedia untuk
    # sistem operasi / proses lain.
    #
    # Pada mesin lu sebelumnya 7 worker memberikan hasil:
    #
    #   34,419 file -> ~142.9 detik
    #
    # sehingga konfigurasi ini sudah terbukti bekerja dengan baik.

    NETCDF_READ_WORKERS = max(
        1,
        (os.cpu_count() or 4) - 1,
    )

    # =========================================================================
    # MODEL OUTPUT
    # =========================================================================

    EXPANDING_MODELS_DIR = os.path.join(
        PROJECT_ROOT,
        "models",
    )

    # =========================================================================
    # NOISE SWEEP
    # =========================================================================
    #
    # Masih dipertahankan karena belum kita rombak.
    # Jangan digunakan dalam eksperimen utama sampai desain training baru
    # selesai.

    NOISE_SWEEP_MODELS_DIR = os.path.join(
        PROJECT_ROOT,
        "models_noise_sweep",
    )

    STEP_NOISE_SWEEP_MODELS_DIR = os.path.join(
        PROJECT_ROOT,
        "models_step_noise_sweep",
    )

    # =========================================================================
    # EVALUATION
    # =========================================================================

    EXPANDING_EVAL_DIR = os.path.join(
        PROJECT_ROOT,
        "evaluation",
    )

    RECURSIVE_EVAL_DETAIL_FILE = os.path.join(
        EXPANDING_EVAL_DIR,
        "recursive_evaluation.csv",
    )

    RECURSIVE_EVAL_SUMMARY_FILE = os.path.join(
        EXPANDING_EVAL_DIR,
        "recursive_mae_summary.csv",
    )

    # =========================================================================
    # EXPERIMENTAL SWEEP OUTPUTS
    # =========================================================================

    DAMPING_SWEEP_DIR = os.path.join(
        EXPANDING_EVAL_DIR,
        "sweep_damping",
    )

    NOISE_STD_SWEEP_DIR = os.path.join(
        EXPANDING_EVAL_DIR,
        "sweep_noise_std",
    )

    STEP_NOISE_SWEEP_DIR = os.path.join(
        EXPANDING_EVAL_DIR,
        "sweep_step_noise",
    )

    # =========================================================================
    # INFERENCE
    # =========================================================================

    # Jumlah file NetCDF terakhir yang dibaca saat inference.
    #
    # Ini sengaja tetap kecil agar inference tidak membaca seluruh
    # 34.420 file historis.

    INFERENCE_TAIL_FILES = 100

    # Prioritas evaluasi untuk pemilihan model production.
    INFERENCE_PRIORITY_STEP_RANGE = (
        12,
        18,
    )

    # Output forecast.
    INFERENCE_DIR = os.path.join(
        PROJECT_ROOT,
        "forecast_output",
    )

    # =========================================================================
    # VISUALIZATION
    # =========================================================================

    # Grid Bandung:
    #
    # latitude  = 5
    # longitude = 7
    # total     = 35 pixel

    PIXEL_GRID_SHAPE = (
        5,
        7,
    )

    # Threshold TBB untuk klasifikasi visual.
    TBB_RISK_THRESHOLDS = (
        200.0,
        270.0,
    )

    VISUALIZATION_DIR = os.path.join(
        PROJECT_ROOT,
        "visualizations",
    )

    VISUALIZATION_FRAME_DURATION_MS = 600

    KOTA_BANDUNG_GEOJSON = os.path.join(
        PROJECT_ROOT,
        "scripts",
        "geojson",
        "KotaBandung.geojson",
    )

    # =========================================================================
    # SUMMARY REPORT
    # =========================================================================

    SUMMARY_REPORT_DIR = os.path.join(
        PROJECT_ROOT,
        "summary_report",
    )

    # =========================================================================
    # INFERENCE / MODEL SELECTION
    # =========================================================================

    # Nilai ini digunakan oleh pipeline inference lama.
    # Dipertahankan supaya modul inference tidak rusak.

    INFERENCE_MODEL_SELECTION = (
        "evaluation"
    )

    # =========================================================================
    # VALIDATION
    # =========================================================================

    @classmethod
    def validate(cls):
        """
        Validasi environment variable yang diperlukan oleh pipeline
        download.

        Tidak memvalidasi FTP untuk tahap lokal seperti Stage 02/03 karena
        tahap tersebut tidak membutuhkan koneksi FTP.
        """

        required = [
            (
                "FTP_HOST",
                cls.FTP_HOST,
            ),
            (
                "FTP_USER",
                cls.FTP_USER,
            ),
            (
                "FTP_PASS",
                cls.FTP_PASS,
            ),
            (
                "FTP_REMOTE_DIR",
                cls.FTP_REMOTE_BASE,
            ),
        ]

        missing = [
            name
            for name, value in required
            if not value
        ]

        if missing:
            raise SystemExit(
                "ERROR: variabel .env berikut belum diset: "
                + ", ".join(missing)
                + ".\n"
                "Salin .env.example menjadi .env "
                "lalu isi kredensialnya."
            )


def load_config():
    """
    Load and validate configuration.
    """
    Config.validate()
    return Config