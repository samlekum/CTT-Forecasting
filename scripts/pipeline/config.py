# ./scripts/pipeline/config.py
# Menyediakan konfigurasi terpusat untuk seluruh pipeline, meliputi pengaturan FTP, direktori proyek, parameter pemrosesan, dan validasi variabel lingkungan (.env).

import os
from dotenv import load_dotenv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))


class Config:
    """Konfigurasi terpusat untuk pipeline."""

    PROJECT_ROOT = PROJECT_ROOT

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
    assert MIN_WINDOW_SIZE >= 2, (
        "MIN_WINDOW_SIZE minimal 2 -- window IS1 butuh >=2 titik supaya slope "
        "(regresi linear y~t) punya varians waktu (t) yang tidak nol. Kalau "
        "cuma 1 titik, denom OLS = 0 dan slope diam-diam fallback ke 0 "
        "(lihat expanding_features.compute_expanding_window_features)."
    )
    assert HORIZON_STEPS >= 1, "HORIZON_STEPS minimal 1 -- expanding window recursive butuh minimal 1 step horizon."

    # Rentang total 1 anchor: dari titik pertama window (anchor_t0) sampai
    # target step HORIZON_STEPS, inklusif. Definisi identik dengan yang
    # sebelumnya cuma dihitung lokal di dataset_builder.py -- disentralkan ke
    # sini (satu sumber kebenaran) karena sekarang DIPAKAI JUGA oleh
    # model_training.py (lihat PURGE_STEPS di bawah, untuk purge/embargo
    # stratified_monthly_split() -- lihat investigasi temporal leakage sesi
    # ini). dataset_builder.py tetap jadi tempat definisi ANCHOR_SPAN dipakai
    # (gap-skip rule §4), tapi nilainya sekarang diturunkan dari sini.
    ANCHOR_SPAN = MIN_WINDOW_SIZE + HORIZON_STEPS

    # Purge/embargo (satuan tick FREQ_MINUTES) yang WAJIB di-drop dari train,
    # di stratified_monthly_split(). KENAPA: window training untuk step
    # manapun (sampai HORIZON_STEPS) bisa menjangkau sampai ANCHOR_SPAN-1
    # titik ke depan dari anchor_t0-nya sendiri -- kalau anchor_t0 train
    # terlalu dekat ke cutoff, window/target row itu bisa menyentuh raw
    # observasi yang SECARA WAKTU sudah masuk wilayah test (bahkan bisa jadi
    # nilai yang sama persis dipakai sebagai target test row lain -- temporal
    # leakage). PURGE_STEPS = ANCHOR_SPAN-1 adalah batas konservatif.
    #
    # DUA SISI diterapkan (lihat docstring lengkap stratified_monthly_split()
    # di model_training.py untuk detail & alasan tiap sisi ditemukan):
    # 1. SEBELUM cutoff tiap bulan: anchor dgn `anchor_t0 < cutoff_time -
    #    PURGE_STEPS*FREQ_MINUTES` di-drop dari train bulan itu.
    # 2. SETELAH anchor test terakhir tiap bulan: anchor MANAPUN (bisa bulan
    #    berikutnya) dgn `anchor_t0` dalam PURGE_STEPS*FREQ_MINUTES menit
    #    setelah anchor test terakhir bulan sebelumnya JUGA di-drop --
    #    ditemukan lewat validasi (bukan cuma teori) bahwa anchor test akhir
    #    bulan bisa punya target yang menjorok ke bulan berikutnya, dan
    #    window training di awal bulan berikutnya bisa berbagi raw value
    #    dengan target itu kalau sisi ini tidak ada.
    #
    # Kedua sisi menjamin window/target train (step manapun) tidak pernah
    # menyentuh index >= cutoff bulan manapun. Test set TIDAK terpengaruh oleh
    # ini (tetap anchor_t0 >= cutoff_time, lihat model_training.py).
    PURGE_STEPS = ANCHOR_SPAN - 1

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

    # Direktori dasar buat scripts/tools/sweep_noise_std.py -- tiap nilai
    # noise_std yang di-sweep dapet subfolder SENDIRI (noise{XXX}/) berisi
    # model .joblib + training_summary.csv, TERPISAH dari EXPANDING_MODELS_DIR
    # produksi (sweep TIDAK PERNAH menimpa model produksi yang lagi dipakai).
    NOISE_SWEEP_MODELS_DIR = os.path.join(PROJECT_ROOT, "models_noise_sweep")

    # ------------------------------------------------------------------
    # Konfigurasi khusus INFERENCE (Tahap 6 / 05_run_inference.py). Baca
    # file .nc TERBARU di FINAL_BASE_DIR (data_bandung/) langsung -- BUKAN
    # fetch FTP baru, BUKAN pakai EXPANDING_RAW_CACHE_FILE (cache itu
    # snapshot dataset training, bukan data real-time terbaru). Lihat
    # pipeline/inference.py.
    # ------------------------------------------------------------------

    # Jumlah file .nc TERAKHIR (kronologis, bukan random) yang dibaca dari
    # FINAL_BASE_DIR untuk cari window observasi terbaru per pixel (atau
    # window di sekitar --t0 kalau dioverride, lihat inference.py). WAJIB
    # slice SEBELUM dilempar ke build_uniform_timeline()/load_pixel_grid()
    # -- 2 fungsi itu baca SEMUA entries yang dikasih tanpa filter internal
    # apapun, dan data_bandung/ berisi puluhan ribu file historis (lupa
    # slice = re-trigger bottleneck I/O berjam-jam, lihat CLAUDE.md §16).
    # 100 file = ~16,7 jam cakupan (10 menit/file) -- cukup toleran untuk
    # pixel yang cloud-covered berjam-jam (MIN_WINDOW_SIZE=6 titik/50 menit
    # bebas-NaN dicari mundur dari file terbaru/--t0), sementara biaya baca
    # tetap murah (puluhan detik, jauh di bawah baca seluruh riwayat).
    # Naikkan lewat --tail-files kalau operasional nunjukkin banyak pixel
    # ke-skip.
    INFERENCE_TAIL_FILES = 100

    # Rentang step (inklusif) yang dipakai select_inference_model() buat
    # milih model production OTOMATIS dari evaluation/recursive_mae_summary.csv
    # (hasil 04_recursive_evaluate.py) -- BUKAN dihardcode nama modelnya di
    # sini, karena model terbaik bisa berubah tiap kali dataset/training
    # berubah. Range (12, 18) = prioritas horizon panjang, sesuai kesimpulan
    # final CLAUDE.md §18.9: metode rekursif bikin error terakumulasi paling
    # signifikan di step-step akhir -- itu ujian utama kualitas model,
    # bukan step awal yang secara inheren lebih gampang diprediksi (window
    # masih didominasi observasi real).
    INFERENCE_PRIORITY_STEP_RANGE = (12, 18)

    # Direktori output forecast (Tahap 6). SUDAH ada di .gitignore
    # (forecast_output/) -- riwayat run TIDAK di-commit, tapi TETAP
    # disimpan lokal sbg 1 FOLDER per run (nama folder timestamped + nama
    # model, pola sama dgn visualizations/ di Tahap 7, BUKAN overwrite
    # "latest") supaya bisa diaudit/dibandingkan lintas run.
    INFERENCE_DIR = os.path.join(PROJECT_ROOT, "forecast_output")

    # ------------------------------------------------------------------
    # Konfigurasi khusus VISUALISASI (Tahap 7 / 06_visualize.py). Render
    # CSV forecast Tahap 6 jadi animasi GIF 6-panel (semua panel = peta
    # spasial pakai lat/lon asli) per step. Lihat pipeline/visualize.py.
    # ------------------------------------------------------------------

    # Dimensi grid pixel Bandung (lat_idx 0-4, lon_idx 0-6) -- FIXED,
    # sesuai subset area download (lihat CLAUDE.md §12 "Definisi pixel":
    # grid ~5x7=35 pixel). Dipakai visualize.py buat reshape long-format
    # CSV -> array 2D siap imshow. TIDAK diturunkan otomatis dari data CSV
    # (max lat_idx/lon_idx yang HADIR) karena Stage 05 bisa skip pixel
    # tepi grid kalau gagal gap-check -- kalau diturunkan dari data, grid
    # bisa "menyusut" diam-diam dan salah tafsir sbg ukuran domain berubah.
    PIXEL_GRID_SHAPE = (5, 7)

    # Threshold TBB (Kelvin) buat klasifikasi 3 kelas risk_banjir/kelas_awan
    # (SAMA PERSIS dipakai kedua panel, cuma label beda -- lihat
    # visualize.classify_tbb_grid()). TBB rendah = puncak awan
    # tinggi/dingin = konveksi kuat = risiko hujan lebat/banjir.
    TBB_RISK_THRESHOLDS = (200.0, 270.0)

    # Direktori output visualisasi (Tahap 7). Sudah ada di .gitignore.
    VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualizations")

    # Durasi tiap frame GIF (ms). 600ms/frame -> total ~10.8s utk 18
    # step, cukup lambat dibaca tapi tidak terlalu lama nunggu 1 putaran.
    VISUALIZATION_FRAME_DURATION_MS = 600

    # Path GeoJSON batas kecamatan Kota Bandung, dipakai sbg overlay
    # opsional di tiap panel peta (garis tipis, murni referensi visual --
    # BUKAN dipakai buat join/agregasi data apapun, sekadar konteks
    # geografis). Sama CRS dgn pixel grid (lat/lon derajat WGS84 polos),
    # jadi align langsung tanpa transformasi. Kalau file tidak
    # ada/gagal di-parse, overlay di-skip diam-diam (fitur pelengkap,
    # bukan wajib) -- lihat visualize.load_kecamatan_boundaries().
    KOTA_BANDUNG_GEOJSON = os.path.join(PROJECT_ROOT, "scripts", "geojson", "KotaBandung.geojson")

    # Direktori output ringkasan presentasi (scripts/tools/generate_summary_report.py)
    # -- agregat Stage 02-05 (statistik dataset, perbandingan model, MAE/metrik
    # spasial per step, contoh forecast) jadi tabel Markdown + chart PNG siap
    # dipakai bahan PPT. Folder-per-run (pola SAMA PERSIS dgn INFERENCE_DIR/
    # VISUALIZATION_DIR): {model}_t0..._run.../ -- identitas run forecast Stage
    # 05 yang dipilih via menu CLI, SENGAJA idempotent (re-generate dari run
    # forecast yang SAMA overwrite folder yang sama, run BEDA bikin folder baru).
    SUMMARY_REPORT_DIR = os.path.join(PROJECT_ROOT, "summary_report")

    # Jarak antar anchor (stride) default di dataset_builder.build_dataset().
    # 1 = semua posisi start valid dipakai (data maksimal, tapi overlap
    # tinggi antar anchor bersebelahan). Naikkan kalau dataset full run
    # ternyata kegedean -- lihat CLAUDE.md §12.
    ANCHOR_STRIDE_DEFAULT = 1

    # Jumlah proses paralel default untuk baca file NetCDF di
    # dataset_builder.load_pixel_grid() (Tahap 2). Bottleneck sebelumnya:
    # loop sekuensial 34.420 file makan ~12.5 jam (I/O/parse-bound, bukan
    # compute -- lihat CLAUDE.md §10 poin 4). Default: semua core kecuali 1
    # (biar OS/proses lain tetap responsif). Override lewat --workers di
    # 02_build_expanding_features.py atau param n_workers eksplisit kalau
    # mau tuning (mis. workers lebih dikit kalau I/O disk jadi bottleneck
    # duluan sebelum CPU, bukan sebaliknya).
    NETCDF_READ_WORKERS = max(1, (os.cpu_count() or 4) - 1)

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