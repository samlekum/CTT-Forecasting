# CLAUDE.md — CTT-Forecasting-Expanding

Dokumen ini adalah panduan konteks untuk sesi Claude berikutnya yang mengerjakan
repo **`CTT-Forecasting-Expanding`**. Baca ini SEBELUM mulai ngoding apapun.

---

## 1. Konteks & Kenapa Repo Ini Ada

Ini adalah **project baru, terpisah total** dari repo lama
`Bandung-Weather-Forecast-Himawari-09`
(https://github.com/nugrahsdhka/Bandung-Weather-Forecast-Himawari-09). Repo
lama TIDAK diubah/disentuh oleh project ini — anggap read-only reference.

Alasan dibuat terpisah: dosen pembimbing Dhika meminta metode recursive
forecasting dengan skema **expanding window**, yang secara fundamental beda
dari pendekatan **fixed sliding window** (`LAG_COUNT=6`) yang sudah dipakai
dan sudah matang di repo lama. Daripada rewrite pipeline lama yang sudah
terbukti jalan (chronological/stratified split, reliability calibration,
dll), keduanya dijalankan sebagai dua project paralel.

**Status saat penulisan dokumen ini: BELUM ADA KODE SAMA SEKALI.**
Semua yang ada di bawah ini adalah hasil sesi *desain/diskusi*, bukan
implementasi. Sesi berikutnya mulai dari nol berdasarkan spesifikasi di
dokumen ini.

---

## 2. Metode Inti: Expanding Window Recursive

Diminta dosen, formatnya:

```
IS1 = [1, 2, 3, 4, 5, 6]        OS1 = [7]
IS2 = [1, 2, 3, 4, 5, 6, 7]     OS2 = [8]
IS3 = [1, 2, 3, 4, 5, 6, 7, 8]  OS3 = [9]
...
IS18 = [1..23]                  OS18 = [24]
```

- Window **input** (IS) mulai dari 6 titik observasi awal, lalu **tumbuh 1
  titik setiap step** (BUKAN sliding/fixed-size seperti pipeline lama).
- Horizon: 18 step × 10 menit = **3 jam**, karena data Himawari resolusinya
  per 10 menit.
- Ini **recursive single-model**: satu model dilatih untuk memprediksi 1
  titik ke depan dari window (berapapun panjangnya), lalu saat inference,
  hasil prediksi step sebelumnya ditambahkan ke window untuk prediksi step
  berikutnya. BUKAN 18 model terpisah per step.
- Dihitung **per pixel** (grid area Bandung), bukan agregat spasial.

### Kenapa nggak bisa raw values langsung jadi kolom

Model tabular (XGBoost/LightGBM/CatBoost) butuh jumlah kolom fitur konstan.
IS1 punya 6 titik, IS18 punya 23 titik — nggak bisa langsung jadi kolom
mentah satu-satu. Solusinya: window (berapapun panjangnya) diringkas jadi
**fitur statistik fixed-size**.

---

## 3. Fitur: Closed-Form Vectorized (INI KUNCI UTAMA)

### Masalah yang harus dihindari

Pendekatan naive (loop Python per anchor × per step, panggil `np.polyfit`
ulang tiap iterasi buat hitung slope) itu SALAH untuk skala data ini.
Konteks: pipeline lama `03a_build_features.py` sudah makan **10 jam+**
untuk fixed-window biasa, dengan **34.420 file .nc**. Pendekatan naive untuk
expanding window akan JAUH lebih lambat dari itu karena tiap window makin
panjang dan di-recompute dari nol tiap step.

### Solusi wajib: cumulative closed-form statistics

Precompute sekali per pixel (per time series):
- `cumsum_y` = `cumsum(y)`
- `cumsum_y2` = `cumsum(y**2)`
- `cumsum_ty` = `cumsum(t * y)` di mana `t` = index waktu global (bukan
  relatif ke window)

Lalu untuk window mana pun `[s, e]` (inklusif), semua fitur dihitung **O(1)**
pakai aritmatika dari cumsum di atas — TIDAK ada re-loop atau re-fit ulang
per window:

- `sum_window = cumsum_y[e] - cumsum_y[s-1]`
- `mean_window = sum_window / n` (n = e - s + 1)
- `std_window` dari `cumsum_y2` (formula varians: `E[y²] - E[y]²`)
- `min_window` / `max_window`: pakai `np.minimum.accumulate` /
  `np.maximum.accumulate` sekali per pixel, lalu lookup index
- `slope` (regresi linear window): closed-form pakai `cumsum_y`,
  `cumsum_ty`, dan index — TIDAK pakai `np.polyfit` di dalam loop
- `first_value`, `last_value`, `delta_first_last`, `n_points`: trivial O(1)

Target hasil: seluruh feature engineering per pixel jadi operasi
**numpy vectorized** (array-to-array / broadcasting), bukan loop Python per
baris. Ini yang bikin waktu proses turun drastis dibanding pendekatan naive.

**PENTING**: sebelum dipakai untuk build dataset penuh, closed-form ini WAJIB
divalidasi correctness-nya dengan cara dibandingkan terhadap implementasi
naive (loop biasa + `np.polyfit`) di subset kecil data, pastikan hasilnya
identik (toleransi floating point kecil) sebelum dipercaya untuk full run.

### Daftar fitur final (9 kolom, fixed-size di semua step)

`mean_window`, `std_window`, `min_window`, `max_window`, `first_value`,
`last_value`, `delta_first_last`, `slope`, `n_points`

---

## 4. Gap Handling

Himawari selalu mengirim data per 10 menit. **Gap = ada timestamp yang
hilang** di antara titik awal window sampai target step terakhir (misal file
gagal download, atau pixel ke-mask NaN karena cloud/kualitas data).

**Aturan**: kalau ada satu saja gap di rentang waktu yang dibutuhkan untuk
satu anchor (dari titik pertama window sampai target OS18), **seluruh anchor
itu di-skip untuk pixel tersebut** — bukan cuma step yang bolong. Alasannya:
step-step berikutnya butuh window yang utuh dari titik awal, jadi kalau ada
bolong di tengah, semua step turunan dari anchor itu jadi tidak valid.

---

## 5. Model

**XGBoost, LightGBM, CatBoost.** SVR SUDAH DI-DROP (keputusan final) —
terlalu lambat untuk ukuran training set expanding window per-pixel
(potensi jutaan baris), independen dari seberapa cepat feature engineering-nya.

---

## 6. Split Strategy

**Langsung pakai `stratified_monthly_split()`** dari awal — TIDAK mulai dari
`chronological_split()` biasa. Ini keputusan sadar berdasarkan pelajaran dari
repo lama: `chronological_split()` biasa menyebabkan bias (bulan-bulan
tertentu, terutama musim konvektif, 0% representasi di test set), yang
akhirnya memutarbalikkan ranking model dan bikin metrik reliability jadi
optimistically bias. Jangan ulangi masalah yang sama di project baru ini.

---

## 7. Evaluasi

**MAE aktual vs prediksi per step.** Reliability calibration (`mae_expected`,
percentile-based conservative estimate, dll seperti di repo lama) **SENGAJA
DI-SKIP** untuk versi ini — itu bukan bagian dari requirement dosen, dan
kompleksitasnya (persentil antar-t0, isu kalibrasi 2-2.3x) berisiko
membengkakkan scope sebelum metode intinya solid.

Meski di-skip, **interface evaluasi harus didesain supaya gampang ditambah
reliability calibration belakangan** tanpa perlu re-arsitektur — misalnya
simpan hasil evaluasi recursive dalam format yang bisa di-extend dengan
kolom tambahan nanti (pola serupa `recursive_evaluation.csv` di repo lama),
bukan format yang kaku.

---

## 8. Nama Kolom Target

`target_tbb_13` — disamakan persis dengan repo lama untuk konsistensi
(walau dataset dan skema fiturnya beda total).

---

## 9. Struktur Repo

```
ctt-forecasting-expanding/
├── .env.example
├── .gitignore
├── README.md                        # BELUM DIBUAT -- ada di rencana awal, belum eksis
├── CLAUDE.md                        # dokumen ini
├── scripts/
│   ├── 01_download_data.py          # REUSE apa adanya dari repo lama, TIDAK diubah
│   ├── 02_build_expanding_features.py  # ✅ SELESAI -- CLI + smoke-test mode
│   │                                 # (--max-files), progress bar per-file & per-pixel (§14)
│   ├── 03_train_models.py           # ✅ SELESAI -- xgboost, lightgbm, catboost,
│   │                                 # progress bar per-model (§14)
│   ├── 04_recursive_evaluate.py     # ✅ SELESAI -- eval recursive per t0,
│   │                                 # MAE per step + metrik spasial (§15),
│   │                                 # flag --damping-factor (§17 poin 5)
│   ├── 05_run_inference.py          # BELUM dibahas -- sesi berikutnya
│   ├── 06_visualize.py              # BELUM dibahas -- sesi berikutnya
│   ├── pipeline/
│   │   ├── config.py                # terpusat: FTP, path, LAG_COUNT (lama),
│   │   │                             # MIN_WINDOW_SIZE/HORIZON_STEPS/TARGET_CHANNEL/
│   │   │                             # EXPANDING_MODELS_DIR/TEST_FRAC (expanding window)
│   │   ├── ftp_client.py            # reuse dari repo lama
│   │   ├── netcdf_tools.py          # reuse dari repo lama
│   │   ├── file_tracker.py          # reuse dari repo lama
│   │   ├── telegram_notifier.py     # reuse dari repo lama
│   │   ├── expanding_features.py    # BARU -- closed-form vectorized engine (lihat §3)
│   │   ├── dataset_builder.py       # BARU -- generator sample IS/OS per pixel, gap-aware
│   │   ├── validate_expanding_features.py  # BARU -- validasi closed-form vs naive.
│   │   │                             # CATATAN: file ini ada di scripts/pipeline/, BUKAN
│   │   │                             # scripts/tools/ seperti draft awal §9/§10 -- kalau
│   │   │                             # mau samain struktur sama repo lama, tinggal
│   │   │                             # dipindah, belum dilakukan.
│   │   ├── model_training.py        # ✅ SELESAI -- stratified_monthly_split (reuse
│   │   │                             # verbatim dari repo lama, time_col="anchor_t0") +
│   │   │                             # training 3 model + save .joblib + training_summary.csv
│   │   ├── recursive_eval.py        # ✅ SELESAI -- rollout recursive vectorized
│   │   │                             # lintas anchor, spatial_collapse_ratio/correlation
│   │   │                             # (§15), _apply_damping() buat fix exposure bias (§17)
│   │   ├── inference.py             # BELUM dibahas
│   │   └── utils.py
│   ├── tools/                       # diagnostic scripts, pola serupa repo lama.
│   │   │                             # (validate_expanding_features.py justru ada di
│   │   │                             # pipeline/, lihat catatan di atas -- BUKAN di sini)
│   │   ├── diagnose_recursive_drift.py              # §17 poin 3.1
│   │   ├── compare_interior_vs_edge_spatial_metrics.py  # §17 poin 3.2
│   │   └── check_true_spatial_variance.py           # §17 poin 3.3
│   └── ui/
│       └── terminal_display.py      # reuse dari repo lama
```

---

## 10. Scope Sesi Berikutnya

**KOREKSI (sesi ini)**: draft sebelumnya nulis `02_build_expanding_features.py`
"BELUM" di §9 -- itu KELIRU, file itu udah ada & udah di-commit dari sesi
sebelumnya (lihat poin 3 di daftar bawah). Fokus sesi berikutnya sekarang
tinggal **04 (`recursive_eval.py`) saja**, lanjut ke `05`/`06` yang emang
belum pernah dibahas desainnya.

Fokus sesi ini + berikutnya: **02 → 03 (sudah selesai) → 04** (build
features → train → recursive evaluate). `01` tinggal disalin dari repo
lama. `05` (inference) dan `06` (visualisasi) **belum dibahas sama sekali**
— jangan diasumsikan desainnya, harus didiskusikan dulu di sesi terpisah
sebelum dikoding.

### Urutan kerja yang disarankan untuk sesi berikutnya:
1. ✅ **SELESAI** — `pipeline/expanding_features.py` (closed-form vectorized).
2. ✅ **SELESAI** — Validasi closed-form vs naive loop
   (`scripts/pipeline/validate_expanding_features.py` — CATATAN: path aslinya
   direncanakan `scripts/tools/`, tapi implementasi aktual naruhnya di
   `scripts/pipeline/`, lihat §9), identik dalam toleransi floating point,
   ~38x lebih cepat di test 12.798 window.
3. ✅ **SELESAI** — `pipeline/dataset_builder.py` — generate training samples per
   pixel, terapkan gap-skip rule (§4). Divalidasi pakai file `.nc` sintetis
   (grid 5x7, gap file hilang + gap cloud-mask NaN disengaja) — cross-check
   manual numpy match persis, gap-skip rule dibuktikan nggak ada anchor yang
   overlap posisi NaN. Lihat §12 untuk keputusan desain yang diambil di sini.
4. ✅ **SELESAI** — `pipeline/model_training.py` + `scripts/03_train_model.py`
   (PENTING: nama file aktual **singular** `03_train_model.py`, bukan
   `03_train_models.py` -- typo lama di dokumen ini & di header komentar
   file itu sendiri sudah dibenerin sesi ini) —
   `stratified_monthly_split()` di-reuse VERBATIM dari repo lama (cuma
   `time_col` diganti jadi `"anchor_t0"`), training 3 model (xgboost/
   lightgbm/catboost, SVR di-drop sesuai §5), save `.joblib` +
   `training_summary.csv` ke `Config.EXPANDING_MODELS_DIR` (baru,
   tersentralisasi di config.py). Divalidasi pakai dataset sintetis 3
   bulan: cutoff per-bulan bekerja (semua bulan punya representasi di
   test set, bukan 0% kayak masalah repo lama), training end-to-end
   jalan, model tersimpan valid. Lihat §13 untuk detail keputusan desain.

   **Full run pertama** (34.420 file .nc, 13.621.230 baris, 8 bulan data):
   xgboost MAE=2.505K RMSE=4.514K R²=0.9711 (204.7s), lightgbm MAE=2.517K
   R²=0.9710 (79.7s, tercepat), catboost MAE=2.522K R²=0.9711 (227.9s).
   **PENTING**: ini metrik FLAT/teacher-forced (semua step 1-18 dicampur
   di satu test set, window SELALU observasi real) -- BUKAN performa
   recursive sebenarnya. Lihat hasil recursive di §15.

   **Temuan performa (full run ini)**: closed-form vectorized feature
   engineering di `expanding_features.py` cuma makan **~10 detik** untuk
   35 pixel (sesuai klaim §10 poin 2). TAPI loop baca 34.420 file `.nc`
   satu-satu (`xr.open_dataset()` di `load_pixel_grid()`) makan **12 jam
   29 menit** (~1.3 detik/file) -- total waktu proses 02 jadi 45.222 detik
   (~12.5 jam), justru LEBIH LAMA dari `03a_build_features.py` di repo
   lama (10 jam+) yang jadi alasan awal project ini dibuat (§1, §11).
   Bottleneck-nya SEKARANG di I/O baca file, BUKAN di compute.
   **SUDAH DIBENAHI** di sesi paralelisasi I/O -- lihat §16.
5. ✅ **SELESAI** — `pipeline/recursive_eval.py` (`scripts/04_recursive_evaluate.py`)
   — evaluasi recursive per t0, hitung MAE per step 1-18. Lihat §15 untuk
   detail desain & keputusan.

---

## 14. Keputusan Desain Tambahan (sesi penambahan progress bar)

Diminta Dhika: `02` dan `03` dikasih progress bar biar keliatan jalan,
konsisten sama pengalaman `01` (yang punya per-file download bar dari
`ftp_client.download_with_progress`). Perubahan:

- **`ui/terminal_display.py`**: tambah `make_progress_bar(iterable, desc,
  unit)` -- generalisasi dari `make_total_progress_bar` yang sudah ada
  (yang itu spesifik untuk kata "file" dan sebenarnya belum pernah dipanggil
  di `01_download_data.py`, cuma di-import). `make_progress_bar` dipakai di
  tempat baru (bukan gantiin yang lama), style bar tetep konsisten (tqdm,
  `ncols=80`, `tqdm.write` untuk log biar nggak tabrakan sama bar).
- **`pipeline/dataset_builder.py`** (dipanggil dari `02`): 2 progress bar
  ditambahkan --
  1. `load_pixel_grid()`: loop baca file NetCDF satu-satu (`xr.open_dataset`)
     -- ini I/O paling berat di Tahap 2 kalau file ribuan, jadi paling
     penting buat kasih progress bar (desc "Baca NetCDF", unit "file").
  2. `build_dataset()`: loop per-pixel (biasanya cuma puluhan pixel, jadi
     bar-nya cepet selesai, tapi tetep dikasih + `set_postfix_str` nunjukin
     running total anchor ketemu, biar user tau proses gap-skip rule lagi
     ngapain).
- **`pipeline/model_training.py`** (dipanggil dari `03`): loop
  `Config.MODEL_NAMES` di `train_all_models()` dikasih progress bar (desc
  "Training model", unit "model") + `say_info`/`say_ok`/`say_error` per
  model (mulai training, selesai dengan MAE/RMSE/R2, atau error kalau
  library belum terinstall) -- pola yang sama kayak `say_download`/`say_ok`
  per-file di `01`.
- **Divalidasi** pakai file `.nc` sintetis (6 pixel, 30 timestamp, tanpa
  gap) end-to-end: `build_dataset()` → dataset 756 baris → `train_all_models()`
  → model `.joblib` + `training_summary.csv` (dicoba pakai LightGBM asli,
  MAE ~0.4K di data sintetis -- angka ini nggak representatif, cuma buat
  mastiin pipeline nggak crash). Progress bar tampil normal di kedua tahap,
  tidak ada tabrakan output sama `say_*()`.
- Tidak ada perubahan logika inti (fitur, gap-skip, split, training) --
  murni penambahan visibilitas progres.

---

## 11. Kebiasaan Kerja (Berlaku Sama Seperti Repo Lama)

- Komunikasi santai, Bahasa Indonesia informal.
- Progress step-by-step, bukan sekaligus semua — konfirmasi dulu sebelum
  ngoding bagian besar.
- Kalau ada operasi berat (build features skala penuh, training penuh),
  **jangan langsung full-run** tanpa validasi di subset kecil dulu — belajar
  dari `03a_build_features.py` yang makan 10 jam+ di repo lama.
- Update dokumen ini kalau ada keputusan desain baru supaya sesi berikutnya
  nggak perlu re-diskusi dari nol.

## 12. Keputusan Desain Tambahan (dari sesi implementasi `dataset_builder.py`)

Bagian ini belum ada di draft desain awal, diputuskan pas implementasi.

**KOREKSI (sesi berikutnya)**: kalimat asli di sini bilang "nggak ada
preseden/reference code karena `03a_build_features.py` TIDAK ada di
GitHub". Itu KELIRU — repo lama yang dicek sebelumnya salah URL (lihat
koreksi §1). Repo lama yang benar
(https://github.com/nugrahsdhka/Bandung-Weather-Forecast-Himawari-09)
justru punya pipeline LENGKAP yang udah di-push: `03a_build_features.py`,
`pipeline/feature_engineering.py`, `pipeline/spatial_features.py`,
`04_train_models.py`, `pipeline/model_training.py`,
`05_recursive_evaluation.py`, `pipeline/recursive_eval.py`, plus
`tools/check_seasonal_distribution.py` dkk. Keputusan desain di bawah ini
TETAP VALID (sudah divalidasi manual terpisah, lihat CLAUDE.md §10 &
riwayat sesi), tapi kalau ada sesi berikutnya yang perlu preseden buat
modul lain (mis. `inference.py`, `visualize.py`), CEK DULU implementasi
yang sepadan di repo lama yang benar sebelum desain dari nol.

- **Sumber data**: `dataset_builder.py` baca langsung banyak file
  `subset_*.nc` di `data_bandung/` (bukan ada tahap ekstraksi CSV
  perantara). Timestamp diambil dari nama file pakai
  `extract_time_from_filename()` yang udah ada di `netcdf_tools.py`.
- **Definisi pixel**: grid Bandung ~5×7 = 35 pixel (lat×lon), diberi
  `pixel_id` format `"{lat_idx}_{lon_idx}"`.
- **Channel yang dipakai**: HANYA `tbb_13` (sesuai target column §8) —
  TIDAK pakai channel lain (`tbb_07`..`tbb_16`) sebagai fitur tambahan,
  karena CLAUDE.md desain awal cuma bahas window stats dari satu series `y`.
- **Anchor stride = 1** (default): setiap posisi start yang lolos gap-skip
  rule dipakai jadi anchor (window growing dari situ). Bisa diubah lewat
  parameter `anchor_stride` di `build_dataset()` kalau dataset hasil full
  run ternyata kegedean / terlalu redundant antar anchor yang overlap.
- **Grid pixel diasumsikan konsisten** bentuknya (shape sama) antar semua
  file. Grid canonical diambil dari file valid pertama yang dibaca; file
  lain yang shape-nya menyimpang dari itu dianggap **gap penuh di semua
  pixel** untuk timestamp tsb (bukan di-reshape paksa atau bikin crash).
- **Bug yang ketemu & fix penting**: `np.cumsum` mem-propagate `NaN` — satu
  NaN di time series bikin SEMUA cumsum setelah posisi itu ikut NaN, walau
  window yang di-query nggak menyentuh posisi NaN tsb. Fix: `y` di-
  `np.nan_to_num(nan=0.0)` dulu SEBELUM masuk `compute_cumsum_stats()`. Ini
  aman karena `find_valid_anchors()` sudah menjamin window manapun yang
  lolos gap-skip rule nggak pernah menyentuh posisi NaN — jadi nilai
  pengganti 0 nggak pernah ikut kehitung di window valid manapun. `y` asli
  (dengan NaN utuh) tetap dipakai untuk `first_value`/`last_value`/min-max
  (aman karena posisi yang di-index dijamin bukan NaN).

---

## 13. Keputusan Desain Tambahan (dari sesi implementasi `model_training.py`)

- **`stratified_monthly_split()` reuse verbatim** dari repo lama (lihat
  koreksi §1/§12 -- fungsinya ada di `pipeline/model_training.py` repo
  lama). Logikanya sudah generic (`time_col` parameterized), jadi TIDAK
  ada modifikasi logika, cuma dipanggil dengan `time_col="anchor_t0"`
  (bukan `"base_time"` seperti skema fixed-window repo lama).
- **Fitur training**: HANYA 9 kolom `FEATURE_COLUMNS` dari
  `expanding_features.py` (§3) -- TIDAK ditambah `lat`/`lon`/`hour_sin`
  dkk seperti `FEATURE_COLUMNS` di repo lama, karena desain awal §3
  eksplisit cuma nyebut 9 kolom itu sebagai fitur final. `n_points` di
  antara 9 kolom itu berfungsi implisit sebagai penanda step (n_points =
  MIN_WINDOW_SIZE + step - 1), jadi model tetap tahu "sejauh apa" window
  tanpa perlu kolom step eksplisit.
- **Model disimpan sebagai `.joblib`** (bukan pickle biasa) ke
  `Config.EXPANDING_MODELS_DIR` (baru, ditambahkan ke config.py di sesi
  ini) -- path `{model_name}.joblib`, TIDAK per-interval/per-folder
  seperti repo lama (project ini cuma punya satu skema window, beda dari
  repo lama yang punya banyak `INTERVALS_MINUTES`).
- **Evaluasi di `model_training.py` sengaja basic** (MAE/RMSE/R2 di
  seluruh test set tercampur semua step) -- BUKAN evaluasi per-step.
  Evaluasi per-step 1..18 itu scope `recursive_eval.py` (Tahap 04, belum
  dikerjakan), sesuai keputusan diskusi (jangan digabung ke
  `model_training.py`).
- **`noise injection` (fix #3 di repo lama) TIDAK di-port** ke sini --
  belum ada kebutuhan/keputusan buat itu di expanding window, dan bukan
  bagian dari scope §5-§8. Kalau nanti dibutuhkan (mis. setelah lihat
  hasil `recursive_eval.py`), diskusikan dulu sebelum ditambah.
- **Test coverage sebelum dianggap aman**: dataset sintetis 3 bulan (Jan-
  Mar 2024, 5000 baris) dipakai buat verifikasi `stratified_monthly_split`
  menghasilkan representasi test di SEMUA bulan (bukan 0% di bulan
  tertentu), lalu `train_all_models()` dijalankan end-to-end sampai
  model `.joblib` + `training_summary.csv` ke-generate dengan benar.
  BELUM divalidasi ke dataset asli hasil `dataset_builder.py` (karena
  `02_build_expanding_features.py` belum ada) -- WAJIB dicoba di data
  asli sebelum dianggap final produksi.

---

## 15. Keputusan Desain Tambahan (dari sesi implementasi `recursive_eval.py`)

**PENTING -- cek preseden dulu**: repo lama (koreksi §12) ternyata PUNYA
`pipeline/recursive_eval.py` (`scripts/05_recursive_evaluation.py`).
Beberapa konsep di-adopsi/disamakan namanya di sini; beberapa TIDAK bisa
langsung di-port karena skema window beda total (expanding vs fixed
sliding). Rincian di bawah.

### Masalah inti: window recursive butuh raw values, bukan cuma fitur agregat

`expanding_features.csv` cuma nyimpen 9 fitur teragregasi (mean/std/slope/
dll), bukan nilai mentah per titik. Buat recursive eval, window mulai step
2 harus di-extend pakai HASIL PREDIKSI model sendiri (bukan observasi real
lagi) -- butuh nilai mentah buat recompute 9 fitur tiap step.

**Solusi yang dipilih Dhika**: `dataset_builder.build_dataset()` sekarang
JUGA nyimpen cache raw time series (`data_matrix` + `timeline` +
`pixel_meta`) ke `.npz` (`Config.EXPANDING_RAW_CACHE_FILE`, default
`dataset/expanding_raw_cache.npz`) via `save_raw_cache()`/`load_raw_cache()`
di `dataset_builder.py`. Alasan: baca ulang 34rb+ file `.nc` khusus buat
recursive eval bakal ulang bottleneck I/O 12.5 jam yang udah ditemukan di
poin 4 atas -- cache bikin `recursive_eval.py` load raw values dalam
hitungan detik.

**BUG YANG KETEMU & FIX**: numpy>=2.0, `array_of_str.astype(str)` bisa
menghasilkan `StringDType` (variable-length) yang tersimpan sebagai OBJECT
array di `.npz` -- gagal di-load dengan `allow_pickle=False` (yang memang
sengaja dipakai, jangan diubah ke `allow_pickle=True` demi keamanan). Fix:
pakai `np.array(values, dtype="<U32")` eksplisit (fixed-width unicode) buat
`pixel_id`, bukan `.astype(str)`. Sudah divalidasi round-trip save/load.

**Trade-off cache**: `data_matrix` disimpan sebagai `float32` (bukan
`float64`) buat hemat ukuran file -- introduce presisi hilang ~1e-5 K
(divalidasi: fitur step=1 hasil recursive_eval dibanding fitur step=1 di
CSV asli, max diff antar kolom ~1e-5, murni floating point round-off,
BUKAN bug logika). Diterima karena jauh di bawah presisi sensor TBB.

### Kenapa slope tetap valid pakai indeks lokal (bukan global)

`expanding_features.py` (training) hitung slope pakai `t` = indeks GLOBAL
(posisi di timeline penuh per pixel). `recursive_eval.py` hitung ulang
fitur tiap step dari window LOKAL yang tumbuh (`series` array per anchor,
mulai `MIN_WINDOW_SIZE` titik, +1 tiap step) pakai `t` = indeks LOKAL
(0..L-1). Ini VALID karena slope regresi linear invarian terhadap
pergeseran konstan pada `t` (slope = Cov(t,y)/Var(t), keduanya tidak
berubah kalau `t` digeser rata) -- fitur lain (mean/std/min/max/first/last)
sama sekali nggak bergantung pada `t`, jadi otomatis identik juga. Divalidasi
numerik: fitur step=1 lokal vs fitur step=1 tersimpan di CSV cocok dalam
toleransi float32 (lihat poin di atas).

### Vectorisasi: per-anchor lintas step TIDAK bisa, lintas anchor BISA

Beda dari `expanding_features.py` (butuh cumsum trick karena window bisa
berapa pun panjang & jutaan anchor sekaligus per pixel), di sini window
SELALU sama panjang untuk semua anchor pada step yang sama (expanding
window tumbuh sinkron), jadi fitur dihitung full-recompute tiap step
langsung pakai numpy `axis=1` di atas matrix `(n_anchor, L)` -- TANPA
cumsum, karena `L` maksimal cuma 23 (murah). Step k+1 butuh hasil step k
(prediksi jadi titik window berikutnya) -- ini SATU-SATUNYA bagian yang
sekuensial (loop Python 18 iterasi per model), lintas ANCHOR tetap
vectorized penuh di tiap iterasi.

### Pemilihan anchor test set

`select_test_anchors()` panggil `stratified_monthly_split()` PERSIS SAMA
seperti `03_train_model.py` (bukan split baru) -- anchor yang dievaluasi
recursive adalah anchor yang SAMA yang jadi test set training, biar hasil
recursive eval konsisten dibandingkan sama metrik flat di §4 (metrik flat
"kepake" evaluasi di data yang sama, recursive eval juga).

**BEDA dari repo lama** (`select_valid_t0`/`select_valid_t0_stratified`,
yang milih t0 BERSAMA lintas SEMUA pixel via `_is_valid_t0` -- karena
fixed-window model lama itu joint-model semua pixel sekaligus): expanding
window ini modelnya per-pixel independen (§8, HANYA `tbb_13`, single
series), jadi anchor dipilih independen per pixel dari test set yang udah
ada, TIDAK perlu constraint "semua pixel valid di t0 yang sama".

### Metrik spasial (`spatial_collapse_ratio`, `spatial_correlation`) -- diadopsi dari repo lama

Awalnya cuma direncanakan MAE per step (CLAUDE.md §7). Tapi setelah cek
preseden repo lama (`pipeline/recursive_eval.py` di sana), ketemu
`spatial_collapse_ratio()` (std prediksi / std aktual ANTAR PIXEL, di
anchor_t0+step yang sama) dan `spatial_correlation()` -- metrik ini
LANGSUNG menjawab masalah utama yang jadi alasan project ini di-rebuild
(§1: "prediction standard deviation collapsed dramatically across
recursive steps, spatial correlation dropping to near zero by step 2").
MAE saja TIDAK bisa mendeteksi model yang "rata"/flat secara spasial tapi
tetap MAE rendah. Naming & formula disamakan persis dengan repo lama.

Implementasi: `_spatial_metrics_per_step()` group `detail_df` per
(model, step, anchor_t0), filter grup dengan >=2 pixel (anchor_t0 yang
cuma valid di 1 pixel di-skip -- gap-skip rule per-pixel bisa bikin nggak
semua pixel valid bareng di anchor_t0 yang sama), hitung ratio & korelasi
per grup, lalu rata-ratakan lintas anchor_t0 per (model, step). Kolom
`n_t0_groups` di summary nunjukin berapa banyak anchor_t0 yang kepakai --
kalau kecil/0, hati-hati interpretasi (statistik nggak stabil).

### Output & format (extensible sesuai CLAUDE.md §7)

- `Config.RECURSIVE_EVAL_DETAIL_FILE` (`evaluation/recursive_evaluation.csv`):
  long-format, satu baris per (model, pixel_id, anchor_t0, step) --
  kolom: model, pixel_id, anchor_t0, step, target_time, y_true, y_pred,
  abs_error. Gampang di-extend kolom nanti (mis. reliability calibration)
  tanpa re-arsitektur.
- `Config.RECURSIVE_EVAL_SUMMARY_FILE` (`evaluation/recursive_mae_summary.csv`):
  satu baris per (model, step) -- kolom: mae, rmse, n_samples, pred_std,
  true_std (variance keseluruhan, BEDA dari spatial_collapse_ratio --
  ini nyampur variasi antar-anchor & antar-pixel jadi satu, cuma indikator
  kasar tambahan), spatial_collapse_ratio, spatial_correlation, n_t0_groups.

### Divalidasi

Dataset sintetis grid 4x4 (16 pixel), 40 timestamp tanpa gap, pola spasial
tetap per pixel: `build_dataset()` (dgn cache) → `train_all_models()` →
`run_all_models()` (recursive eval) end-to-end jalan tanpa error, 96 anchor
test, `spatial_collapse_ratio`/`spatial_correlation` konsisten stabil
0.9-0.99 (data sintetis sederhana, wajar korelasinya tetap tinggi). Fitur
step=1 lokal vs CSV asli cocok dalam toleransi float32 (~1e-5 K, lihat
poin cache di atas). **BELUM divalidasi di dataset asli 13.6 juta baris**
(karena itu butuh re-run `02_build_expanding_features.py` buat generate
cache-nya dulu -- dataset lama belum punya cache) -- WAJIB dicoba di data
asli sebelum dianggap final. Lihat instruksi re-run di respons chat sesi ini.

### Belum dikerjakan / di luar scope sesi ini

- ~~Perbaikan bottleneck I/O `load_pixel_grid()`~~ -- **SUDAH** (lihat §16).
  Checkpointing tiap N file BELUM (masih di luar scope, lihat §16).
- `pipeline/inference.py` / `05_run_inference.py` -- BELUM dibahas sama
  sekali, jangan diasumsikan desainnya (§10).

---

## 16. Keputusan Desain Tambahan (sesi paralelisasi I/O `load_pixel_grid()`)

Nyerang bottleneck yang ditemukan di §10 poin 4: loop sekuensial baca
34.420 file `.nc` makan ~12.5 jam. Root cause bukan compute (closed-form
feature engineering cuma ~10 detik), tapi overhead buka/parse tiap file
NetCDF satu-satu.

### Desain: ProcessPoolExecutor, BUKAN ThreadPoolExecutor

Library HDF5/netCDF4 C di balik `xr.open_dataset()` pada build default
**TIDAK thread-safe** (thread-safety HDF5 cuma aktif kalau library-nya
dikompilasi eksplisit dengan `--enable-threadsafe`, jarang jadi default
paket binary). Baca paralel lewat banyak thread Python di satu proses
berisiko crash/korupsi diam-diam. Proses terpisah aman total karena tiap
worker punya instance HDF5/netCDF4 sendiri, tidak share state C library
apapun.

### Alur 2 fase di `load_pixel_grid()`

1. **Sekuensial (cepat)**: scan `entries` satu-satu sampai ketemu file
   valid PERTAMA (punya `target_channel`, berhasil dibuka) -- dipakai
   nentuin `canonical_shape` & grid lat/lon, yang WAJIB ada sebelum
   `data_matrix` bisa dialokasikan dan sebelum file lain bisa divalidasi
   shape-nya. File yang di-skip di fase ini (missing channel / gagal
   baca, biasanya di awal urutan kronologis) dicatat sekali, TIDAK dibaca
   ulang di fase 2.
2. **Paralel**: sisa file (setelah file valid pertama) dibaca lintas
   proses via `ProcessPoolExecutor.map()` dengan worker top-level
   `_read_netcdf_worker()` (HARUS top-level module, bukan closure, biar
   bisa di-pickle ke proses lain). Hasil di-assign balik ke `data_matrix`
   di proses utama pakai `t_idx` -- aman tanpa lock karena tiap file
   nulis ke baris array berbeda (`t_idx` unik per timestamp, nggak ada
   overlap write).

`chunksize` untuk `executor.map()` dihitung `len(remaining) // (n_workers
* 4)` -- task-nya kecil (buka 1 file) tapi jumlahnya puluhan ribu, jadi
submit satu future per file mahal karena overhead IPC (pickle/unpickle)
per-task; kelompokkan beberapa file per chunk mengamortisasi overhead itu.

### Config & CLI baru

- `Config.NETCDF_READ_WORKERS` (config.py): default `os.cpu_count() - 1`
  (semua core kecuali 1, biar OS/proses lain tetap responsif).
- `--workers N` di `02_build_expanding_features.py`, override default.
  `--workers 1` = fallback sekuensial murni (skip overhead spawn
  `ProcessPoolExecutor` sama sekali) -- berguna untuk debug atau kalau
  disk I/O (bukan CPU) yang jadi bottleneck di mesin tertentu, di mana
  paralel proses nggak nambah kecepatan (malah nambah overhead).
- `build_dataset()` dapet parameter baru `n_workers=None`, diteruskan ke
  `load_pixel_grid()`.

### Divalidasi

Correctness dites di sandbox terpisah (xarray/netCDF4 diinstall manual di
situ, bukan bagian environment dev utama) pakai file `.nc` sintetis, BUKAN
di data asli 34rb file (belum sempat, lihat "belum dikerjakan" di bawah):

- **Kasus normal** (300 file bersih, grid 5x7): `data_matrix` &
  `pixel_meta` hasil `n_workers=1` vs `n_workers=3` identik byte-per-byte
  (`np.array_equal(..., equal_nan=True)` True, `pixel_meta.equals()` True).
- **Kasus edge** (150 file dengan 3 file tanpa `target_channel` di AWAL
  urutan -- sebelum file valid pertama --, 2 file korup/bukan NetCDF valid
  di tengah run, 1 file shape-mismatch grid): NaN row pattern
  (`[0,1,2,5,40,80]`), counter `n_missing_channel`/`n_mismatched`/
  `n_errors` semua identik antara jalur sekuensial dan paralel.

Sandbox validasi cuma 1 CPU -- speedup RIIL belum diukur di sandbox itu,
cuma correctness yang kebukti. Speedup riil di mesin produksi Dhika sudah
diukur setelahnya -- lihat "Hasil smoke-test di mesin produksi" di bawah.

### Hasil smoke-test di mesin produksi (Windows, 4 physical core / 8 logical thread)

Diukur Dhika pakai `--max-files 500` (bukan estimasi/dugaan), data asli
Himawari (bukan sintetis):

| Workers | Waktu total | Waktu baca (fase paralel) | Speedup vs 2 workers |
|---|---|---|---|
| 2 | 477.2s | ~470s | 1.0x (baseline) |
| 4 | 297.9s | ~291s | 1.60x |
| 8 | 237.7s | ~229s | 2.01x |

**Scaling TIDAK linear** -- diminishing returns jelas kelihatan mulai
4→8 workers (cuma +25% speedup padahal worker dilipatgandakan), padahal
2→4 workers efisiensinya jauh lebih baik (+60% speedup buat pelipatan
yang sama). Root cause dikonfirmasi via
`Get-CimInstance -ClassName Win32_Processor`: mesin Dhika cuma punya
**4 physical core**, `NumberOfLogicalProcessors=8` itu hasil
hyperthreading, BUKAN 8 core fisik independen. Kerjaan parsing
HDF5/netCDF itu CPU-bound berat (bukan I/O-bound murni di kasus ini) --
dua logical thread di physical core yang sama rebutan unit eksekusi
fisik yang sama, jadi worker ke-5 sampai ke-8 cuma dapet speedup parsial
dari hyperthreading, bukan paralelisme penuh kayak worker 1-4.

**Kesimpulan praktis**: `--workers 8` tetap yang tercepat dari semua
percobaan (bukan berarti workers lebih banyak selalu lebih baik di mesin
manapun, tapi di kasus ini masih net-positive dibanding 4), jadi tetap
dipakai buat full run. Ekstrapolasi kasar ke 34.420 file pakai rate
workers=8 (0.476 detik/file dari fase paralel): ~4.5 jam, dibanding
baseline sekuensial lama 12.5 jam (~2.7x lebih cepat) -- signifikan,
meski bukan 8x seperti yang diharapkan kalau scaling linear sempurna.
Angka ini ekstrapolasi dari 500 file pertama (kronologis) di awal
rentang data, belum tentu representatif ke seluruh 8 bulan data kalau
ada variasi kondisi file/disk antar periode.

**Implikasi buat mesin lain**: default `Config.NETCDF_READ_WORKERS`
(`cpu_count - 1`) pakai `os.cpu_count()` yang di Python ngembaliin
LOGICAL processor count, bukan physical core. Di mesin dengan
hyperthreading/SMT aktif, ini artinya default-nya bisa "overestimate"
jumlah paralelisme CPU-bound yang efektif -- bukan salah/bug, cuma perlu
diketahui kalau nanti mau tuning lebih jauh di mesin lain. Belum ada
logic buat auto-deteksi physical-core-only di config.py (di luar scope
sesi ini kalau mau ditambah, bisa pakai `psutil.cpu_count(logical=False)`
kalau dependency itu diterima).

### Belum dikerjakan / di luar scope sesi ini

- **Full run 34rb file BELUM dijalankan** -- baru smoke-test
  `--max-files 500`. Dhika sudah menjadwalkan full run
  (`--workers 8`, tanpa `--max-files`) setelah smoke-test ini, lanjut
  `03_train_model.py` dan `04_recursive_evaluate.py`. Hasil aktual
  (durasi total, apakah ada crash/error di file-file yang belum pernah
  ke-exercise di smoke-test 500 file pertama) BELUM diketahui saat
  dokumen ini ditulis.
- **Checkpointing tiap N file** (disebut sebagai kandidat di §10 poin 4)
  MASIH belum diimplementasi -- run full masih nggak save intermediate
  state. Risikonya sekarang lebih kecil (perkiraan ~4.5 jam, bukan 12.5
  jam), tapi tetap ada kalau crash di tengah jalan.
- **Auto-deteksi physical core** (lihat catatan `os.cpu_count()` di
  atas) belum diimplementasi -- default masih pakai logical processor
  count apa adanya.

---

## 17. Full Run Pertama Tahap 02→03→04 + Investigasi Bug "Recursive Drift" + Fix Damping

**STATUS SINGKAT**: full run 34.420 file `.nc` SELESAI (bukan lagi
smoke-test 500 file di §16). Tahap 02→03→04 semua jalan tanpa error.
Ditemukan bug nyata di hasil recursive eval (`spatial_collapse_ratio`
NAIK jauh di atas 1, bukan turun ke 0 seperti masalah repo lama).
Investigasi 4 tools diagnostic (`scripts/tools/`) sudah dilakukan.
**FIX SUDAH DIIMPLEMENTASI** (damping geometris, lihat poin 4) tapi
**BELUM DIVALIDASI dengan re-run** -- itu tugas paling urgent sesi
berikutnya (lihat "Belum dikerjakan" di akhir section ini).

### 1. Hasil full run 02→03→04 (data asli, BUKAN sintetis)

- **02** (`--workers 8`): 34.420 file `.nc` dibaca, waktu total **14.358
  detik (~4 jam)** -- sesuai ekstrapolasi §16 (dulu diperkirakan ~4.5 jam
  dari smoke-test 500 file, realisasinya malah sedikit lebih cepat).
  Output: 13.621.230 baris, 35 pixel, 756.735 anchor unik, rentang
  2025-12-01 s/d 2026-07-31 (8 bulan).
- **03** (`03_train_models.py` -- PENTING: nama file final **plural**
  `03_train_models.py`, dokumen §10 poin 4 sebelumnya masih nyebut
  singular `03_train_model.py`, sudah di-rename commit `e67a021`, JANGAN
  bingung lagi soal ini di sesi depan): split 11.576.250 train /
  2.044.980 test (8 bulan). Metrik FLAT (bukan recursive):
  xgboost MAE=2.505K R²=0.9711 (238.1s), lightgbm MAE=2.517K R²=0.9710
  (68.7s, tercepat), catboost MAE=2.522K R²=0.9711 (222.1s). Angka ini
  MIRIP dengan full run pertama (§10 poin 4) yang sudah dilaporkan
  sebelumnya -- konsisten, bukan regresi.
- **04** (`04_recursive_evaluate.py`, TANPA damping -- ini baseline):
  113.610 anchor test unik, 3.246 grup anchor_t0 valid (>=2 pixel) buat
  metrik spasial. **Hasil MAE per step naik wajar** (2.55K di step 1 ->
  14.3K di step 18, xgboost) -- tapi **`spatial_collapse_ratio` naik dari
  ~1.03 (step 1) ke ~2.32 (step 18, xgboost)**, BUKAN turun ke 0.
  `spatial_correlation` ambruk dari ~0.81 ke ~0.08. Detail lengkap ada di
  `evaluation/recursive_evaluation.csv` / `recursive_mae_summary.csv`
  (baseline, damping_factor=1.0 -- lihat poin 4, JANGAN ketimpa file ini
  kalau nanti nyoba damping_factor<1.0, otomatis ke file suffix beda).

### 2. Kenapa ini BUG (bukan cuma variasi normal)

`spatial_collapse_ratio` = std(prediksi antar pixel) / std(aktual antar
pixel) pada anchor_t0+step yang sama (lihat §15 utk definisi & alasan
metrik ini ada). Nilai jauh di ATAS 1 (bukan mendekati 0 kayak masalah
repo lama di §1) berarti prediksi model MELEBAR terlalu jauh dibanding
variasi TBB asli antar pixel -- kombinasi ratio tinggi + correlation
rendah = variasinya ADA tapi salah tempat/salah arah (bukan sekadar flat).

### 3. Investigasi (`scripts/tools/`, 3 script, urutan kronologis sesi)

Semua 3 tools TIDAK re-train, cuma baca ulang artifact yang sudah ada
(`recursive_evaluation.csv`, cache `.npz`, model `.joblib`). Ditaruh di
`scripts/tools/` sesuai konvensi §9.

1. **`diagnose_recursive_drift.py`** -- cek runaway pixel (deviasi dari
   rata-rata anchor), clamping (`frac_near_train_min/max`), dan feature
   distribution shift (overlap persentil 5-95 rollout vs training).
   **Temuan**: pixel `4_6` (pojok grid, `lat_idx=4` & `lon_idx=6`
   sama-sama ekstrem) KONSISTEN jadi kontributor deviasi terbesar di
   HAMPIR SEMUA step & model. `frac_near_train_min/max` = **0.0 di semua
   baris** -- BUKAN clamping/leaf-boundary, prediksi beneran divergen ke
   luar wajar, bukan cuma nempel batas. Fitur `std_window`, `slope`,
   `delta_first_last` overlap p5-p95 turun sampai **~0.5 di step 10-18**
   -- indikasi kuat covariate shift/ekstrapolasi.
2. **`compare_interior_vs_edge_spatial_metrics.py`** -- follow-up,
   hipotesis awal "pixel tepi = sumber distorsi", exclude 20 dari 35
   pixel tepi (sisa 15 interior). **Temuan MENGEJUTKAN (berlawanan
   hipotesis)**: exclude tepi malah bikin `spatial_collapse_ratio` makin
   NAIK (2.32 -> 2.82 di step 18, xgboost) dan `spatial_correlation`
   makin TURUN (0.082 -> 0.067). Pixel tepi BUKAN satu-satunya penyebab.
3. **`check_true_spatial_variance.py`** -- follow-up dari kejutan poin 2.
   Ukur langsung `std(y_true)` & `std(y_pred)` antar pixel, full grid vs
   interior. **Temuan**: `true_std` (variasi TBB ASLI) di interior lebih
   kecil ~21% dibanding full grid (konsisten di semua step) -- **INI
   YANG JADI PENYEBAB** kenapa ratio interior lebih jelek dari full di
   poin 2 (penyebut mengecil, bukan model tambah ngaco). Script ini
   sendiri nge-print "hipotesis kekonfirmasi" dan sesi sebelumnya
   (sebelum sesi ini) SEMPAT DISIMPULKAN itu jawaban lengkapnya.

### 4. KOREKSI PENTING atas kesimpulan sesi sebelumnya (poin 3 di atas TIDAK CUKUP)

Temuan poin 3 (`check_true_spatial_variance.py`) HANYA jawab pertanyaan
HORIZONTAL: "kenapa interior_15px lebih jelek dari full_35px di step yang
SAMA?" -- itu valid dan benar. TAPI itu BUKAN jawaban untuk pertanyaan
VERTIKAL yang jadi bug utama di poin 1-2: **kenapa `spatial_collapse_ratio`
di full_35px SENDIRI naik dari 1.03 ke 2.32 seiring STEP bertambah?**

Bandingkan `true_std_mean` vs `pred_std_mean` (full_35px, xgboost) dari
output `check_true_spatial_variance.py` step 1 vs step 18:
- `true_std_mean`: 5.40 -> 4.95 (turun ~8%, stabil relatif)
- `pred_std_mean`: 5.48 -> 8.23 (**naik ~50%**)

Jadi `pred_std` yang MELEDAK, bukan `true_std` yang menyusut. Root cause
SEBENARNYA (gabungan poin 1 + angka ini): **exposure bias / covariate
shift klasik pada recursive rollout** -- model dilatih HANYA lihat window
isi observasi real, tapi dievaluasi recursive dengan window yang makin
lama makin isi prediksi sendiri -> fitur (`std_window`/`slope`/
`delta_first_last`) terdorong ke kombinasi yang jarang/nggak pernah
dilihat model pas training -> model ekstrapolasi liar, PALING PARAH di
pixel tepi/pojok grid (kemungkinan karena TBB di situ emang lebih
volatile secara fisik, atau artefak subsetting NetCDF di batas bbox).

**Kalau sesi depan ketemu referensi ke kesimpulan
`check_true_spatial_variance.py` sebagai "penjelasan lengkap" bug ini --
itu KELIRU/PARSIAL. Rujuk section ini (§17 poin 4) sebagai koreksi.**

### 5. Fix yang SUDAH diimplementasi: damping geometris (BELUM divalidasi re-run)

Diimplementasi di `pipeline/recursive_eval.py` (fungsi baru
`_apply_damping()`) + `scripts/04_recursive_evaluate.py` (flag CLI baru
`--damping-factor`). **TIDAK butuh retrain, TIDAK butuh rebuild dataset
02** (yang makan ~4 jam) -- murni post-processing di rollout Tahap 04.

**Cara kerja**: pisahkan prediksi mentah model jadi
`last_value + delta`, lalu kecilkan `delta` itu geometris tiap step
(`delta * damping_factor ** (step-1)`) SEBELUM dipakai sebagai forecast
final DAN sebelum di-feed balik ke window (jadi efeknya ikut kebawa ke
fitur step berikutnya, bukan cuma di angka yang dilaporkan).
`damping_factor=1.0` (default) = **TIDAK ADA PERUBAHAN sama sekali**,
100% backward-compatible dengan hasil baseline yang sudah ada di poin 1.

Detail lain:
- Kolom baru `y_pred_raw` (prediksi mentah SEBELUM damping) ditambah ke
  `recursive_evaluation.csv` -- biar bisa lihat efek damping tanpa re-run
  ulang dari nol.
- Kalau `damping_factor < 1.0`, output CSV (detail & summary) ditulis ke
  file BERBEDA (suffix `_dampXX`, mis. `recursive_evaluation_damp090.csv`)
  -- baseline (`damping_factor=1.0`, sudah ada dari poin 1) TIDAK
  ketimpa, biar bisa dibandingin langsung.
- Sudah di-smoke-test dengan dummy model (constant runaway drift
  +10/step): `damping_factor=1.0` hasilnya identik byte-per-byte dengan
  tanpa damping (verifikasi backward-compat). `damping_factor=0.5`
  terbukti bikin delta konvergen (step1 delta=10 penuh, step2=5,
  step3=2.5, ...) alih-alih terus melebar linear -- sesuai intuisi
  desain. TAPI ini BARU smoke-test logic dengan dummy model, BELUM
  dicoba di model asli (xgboost/lightgbm/catboost) + data asli.
- Pesan info di akhir `04_recursive_evaluate.py` juga diperbaiki --
  sebelumnya cuma warning "ratio mendekati 0", padahal bug asli di
  project ini justru "ratio menjauh dari 1 ke ATAS". Sekarang jelasin
  kedua arah.

### Belum dikerjakan / di luar scope sesi ini -- URGENT buat sesi depan

- **Re-run `04_recursive_evaluate.py --damping-factor <nilai>` dengan
  model asli & data asli BELUM DILAKUKAN.** Ini prioritas #1 sesi depan
  -- run cuma makan ~5-12 detik (bukan jam), jadi bisa coba beberapa
  nilai (mulai dari 0.9, turun bertahap ke 0.8/0.7 kalau perlu) dan
  bandingkan `spatial_collapse_ratio`/`spatial_correlation`/MAE per step
  vs baseline (`evaluation/recursive_evaluation.csv` yang sudah ada,
  damping_factor=1.0). Cari titik yang collapse_ratio & correlation udah
  stabil deket 1 tapi MAE nggak jauh lebih jelek dari baseline.
- **Kalau damping saja nggak cukup** (mis. MAE korban terlalu besar demi
  ratio bagus): opsi lanjutan yang BELUM dikerjakan -- noise injection ke
  fitur turunan window (`std_window`, `slope`, `last_value`, dst) SEBELUM
  training, biar model belajar robust ke window "asing" dari awal. Ada
  preseden di repo lama (`inject_lag_noise()` di
  `Bandung-Weather-Forecast-Himawari-09/scripts/pipeline/model_training.py`)
  tapi BUKAN implementasi verbatim -- skema fitur beda (9 fitur closed-form
  di sini vs lag mentah di repo lama), perlu didesain ulang, bukan
  copy-paste. Ini butuh retrain (~10 menit, TIDAK perlu rebuild 02).
- **CATATAN soal memori/klaim "exponential damping" dari sesi sebelumnya
  di luar chat ini**: sempat disebut ada fix serupa ("exponential
  damping") di repo lama dari fixes #8-#9 -- SUDAH DICEK, TIDAK ketemu di
  kode yang ke-push ke `Bandung-Weather-Forecast-Himawari-09` (kemungkinan
  itu perubahan lokal Dhika yang belum di-push). Damping di §17 poin 5 ini
  didesain dari nol berdasarkan diagnosis sesi ini, BUKAN port dari kode
  lama yang terverifikasi.
- Kalau nanti damping_factor optimal sudah ketemu, PERTIMBANGKAN update
  `Config` (`pipeline/config.py`) buat nyimpen nilai defaultnya di sana
  (sekarang default CLI masih hardcode `1.0` di `04_recursive_evaluate.py`,
  belum tersentralisasi kayak pola config lain di project ini).
- `pipeline/inference.py` / `05_run_inference.py` -- masih BELUM dibahas
  sama sekali (tetap seperti §10 poin 5 lama), TAPI kalau damping
  divalidasi berhasil, desain inference production nanti WAJIB pakai
  logika damping yang sama (jangan re-derive dari nol, reuse
  `_apply_damping()` dari `recursive_eval.py` atau extract ke modul
  shared kalau perlu dipakai di kedua tempat).

---