# ./scripts/pipeline/config.py
# Menyediakan konfigurasi terpusat untuk seluruh pipeline, meliputi pengaturan FTP, direktori proyek, parameter pemrosesan, dan validasi variabel lingkungan (.env).

import os
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class Config:
    PROJECT_ROOT = PROJECT_ROOT

    """Konfigurasi terpusat untuk pipeline."""
    # FTP Credentials
    FTP_HOST = os.environ.get("FTP_HOST")
    FTP_USER = os.environ.get("FTP_USER")
    FTP_PASS = os.environ.get("FTP_PASS")
    FTP_REMOTE_BASE = os.environ.get("FTP_REMOTE_DIR")
    FTP_TIMEOUT = 60

    # Telegram notifikasi (opsional -- kalau kosong, notifikasi otomatis nonaktif)
    TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
    TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

    # Filters
    FILENAME_MUST_CONTAIN = os.environ.get(
        "FILENAME_MUST_CONTAIN", "R21_FLDK.02801_02401"
    ).strip()

    # Paths
    TEMP_FILE = os.path.join(PROJECT_ROOT, "temp_download.nc")
    FINAL_BASE_DIR = os.path.join(PROJECT_ROOT, "data_bandung")
    METADATA_BASE_DIR = os.path.join(PROJECT_ROOT, "_metadata")
    LOG_FILE = os.path.join(PROJECT_ROOT, "download_activity.log")

    # Retry
    MAX_RETRY = 3

    # Subset bounds (Bandung area)
    BANDUNG_LAT_MIN = -7.0
    BANDUNG_LAT_MAX = -6.8
    BANDUNG_LON_MIN = 107.5
    BANDUNG_LON_MAX = 107.8

    # Kanal infrared Himawari
    TBB_CHANNELS = [f"tbb_{i:02d}" for i in range(7, 17)]  # tbb_07 ... tbb_16

    # Model ML yang dipakai di Tahap 4 (training), 5 (evaluasi recursive), 6 (inference).
    # Ubah di sini saja kalau mau menambah/mengurangi model -- otomatis konsisten di ketiganya.
    MODEL_NAMES = ["xgboost", "lightgbm", "catboost"]

    # Interval prediksi (menit) yang dipakai di Tahap 3a, 3b, 4, 5.
    INTERVALS_MINUTES = [10, 30, 60]

    # Jumlah lag (window observasi) yang dipakai sebagai fitur -- SEBELUMNYA 3
    # (t, tm1, tm2). Dipakai di Tahap 3a (bangun kolom lag), model_training.py
    # (FEATURE_COLUMNS) dan inference.py (window recursive forecast). Ubah di
    # sini SAJA -- otomatis konsisten di semua tahap. PENTING: mengubah nilai
    # ini mengubah skema kolom CSV di dataset/features_*min.csv (kolom
    # tbb_13_tm3, tm4, ... baru), jadi WAJIB jalankan
    # `03a_build_features.py --rebuild` (bukan mode incremental) setelah
    # mengubah nilai ini, atau tahap-tahap berikutnya akan gagal/tidak
    # konsisten dengan CSV lama.
    LAG_COUNT = 6
    assert LAG_COUNT >= 3, "LAG_COUNT minimal 3 -- fitur delta/accel butuh tbb_13_t/tm1/tm2."

    # ------------------------------------------------------------------
    # Konfigurasi khusus EXPANDING WINDOW (project ini -- lihat CLAUDE.md
    # §2-3, §12). Dipakai oleh expanding_features.py & dataset_builder.py.
    # Beda dari LAG_COUNT/INTERVALS_MINUTES di atas yang merupakan skema
    # fixed sliding window punya repo lama (dibiarkan ada di sini kalau ada
    # script yang di-reuse apa adanya, tapi TIDAK dipakai expanding window).
    # ------------------------------------------------------------------

    # Channel target satu-satunya yang dipakai sebagai time series `y`
    # untuk window expanding. TIDAK multi-channel -- lihat CLAUDE.md §8.
    TARGET_CHANNEL = "tbb_13"

    # Ukuran window awal (IS1 = 6 titik) & jumlah step horizon (OS1..OS18).
    # Ubah di sini SAJA kalau mau ubah skema window -- otomatis konsisten
    # di dataset_builder.py. PENTING: mengubah nilai ini mengubah definisi
    # anchor span (rentang wajib bebas-gap untuk gap-skip rule §4) dan
    # jumlah sample per anchor, jadi dataset hasil build_dataset() yang
    # lama harus di-rebuild ulang setelah diubah.
    MIN_WINDOW_SIZE = 6
    HORIZON_STEPS = 18

    # Resolusi timeline data Himawari (menit). Dipakai dataset_builder.py
    # buat bangun timeline uniform. Ubah di sini SAJA kalau sumber data
    # berubah resolusi -- jangan hardcode di fungsi manapun.
    FREQ_MINUTES = 10

    # Direktori & nama file default output dataset training expanding window.
    EXPANDING_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    EXPANDING_DATASET_FILE = os.path.join(EXPANDING_DATASET_DIR, "expanding_features.csv")

    # Cache raw time series (data_matrix + timeline + pixel_meta) hasil baca
    # NetCDF di 02_build_expanding_features.py. Disimpan sekali di sini biar
    # 04_recursive_evaluate.py TIDAK perlu baca ulang 34rb+ file .nc (yang
    # makan ~12.5 jam I/O -- lihat catatan performa sesi ini) buat dapetin
    # nilai mentah tbb_13 per titik. Format .npz (numpy compressed).
    EXPANDING_RAW_CACHE_FILE = os.path.join(EXPANDING_DATASET_DIR, "expanding_raw_cache.npz")

    # Direktori & file output evaluasi recursive (Tahap 5 / 04_recursive_evaluate.py).
    # recursive_evaluation.csv = detail per (model, pixel_id, anchor_t0, step).
    # recursive_mae_summary.csv = ringkasan MAE (+ std prediksi vs std observasi
    # asli, buat ngecek spatial collapse) per (model, step). Format ini SENGAJA
    # dibuat gampang di-extend kolom (mis. reliability calibration) belakangan
    # tanpa re-arsitektur, sesuai CLAUDE.md §7.
    EXPANDING_EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
    RECURSIVE_EVAL_DETAIL_FILE = os.path.join(EXPANDING_EVAL_DIR, "recursive_evaluation.csv")
    RECURSIVE_EVAL_SUMMARY_FILE = os.path.join(EXPANDING_EVAL_DIR, "recursive_mae_summary.csv")

    # Direktori output model hasil training (Tahap 3) & ringkasan metrik.
    # Dipakai model_training.py / 03_train_models.py. Path model per-model:
    # {EXPANDING_MODELS_DIR}/{model_name}.joblib. Ringkasan metrik gabungan:
    # {EXPANDING_MODELS_DIR}/training_summary.csv
    EXPANDING_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

    # Fraksi test default untuk stratified_monthly_split() di Tahap 3.
    TEST_FRAC = 0.15

    # Jarak antar anchor (stride) default di dataset_builder.build_dataset().
    # 1 = semua posisi start valid dipakai (data maksimal, tapi overlap
    # tinggi antar anchor bersebelahan). Naikkan kalau dataset full run
    # ternyata kegedean -- lihat CLAUDE.md §12.
    ANCHOR_STRIDE_DEFAULT = 1

    @classmethod
    def validate(cls):
        """Validasi semua env var wajib terisi, dan laporkan nama-nama yang belum diisi."""
        required = [
            ("FTP_HOST", cls.FTP_HOST),
            ("FTP_USER", cls.FTP_USER),
            ("FTP_PASS", cls.FTP_PASS),
            ("FTP_REMOTE_DIR", cls.FTP_REMOTE_BASE),
        ]
        missing = [name for name, val in required if not val]
        if missing:
            raise SystemExit(
                f"ERROR: variabel .env berikut belum diset: {', '.join(missing)}.\n"
                "Salin .env.example menjadi .env lalu isi kredensialnya."
            )


def load_config():
    """Load and validate configuration."""
    Config.validate()
    return Config