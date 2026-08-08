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
Investigasi 4 tools diagnostic (`scripts/tools/`) sudah dilakukan. Fix
damping geometris diimplementasi (poin 4) DAN **SUDAH DIVALIDASI** lewat
sweep 6 nilai damping_factor di model & data asli (poin 6) --
**`damping_factor=0.9` terbukti optimal** (MAE turun, bukan naik, sambil
motong ~49% kelebihan `spatial_collapse_ratio`). TAPI damping doang
**TIDAK CUKUP** buat nutup gap sepenuhnya (ratio plateau di ~1.55,
correlation plateau di ~0.19 walau damping_factor diturunin sampai 0.5) --
next step urgent sesi depan: **noise injection** ke fitur turunan window.
**UPDATE**: desain + implementasi noise injection (poin 7) SUDAH SELESAI &
smoke-tested (fixture kecil, bukan data asli) -- next step urgent sesi
depan sekarang jadi RETRAIN pakai `--noise-std` di data asli, DIIKUTI
re-sweep `damping_factor` pakai model hasil retrain (lihat "Belum
dikerjakan" di akhir section ini).

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

### 6. Hasil sweep damping_factor + kesimpulan (SUDAH divalidasi, model & data asli)

Divalidasi pakai `scripts/tools/sweep_damping.py` (tool baru, load cache +
model SEKALI lalu loop `run_recursive_evaluation()` per faktor -- lebih
cepat daripada manual re-run `04_recursive_evaluate.py` per nilai. TIDAK
retrain, TIDAK rebuild dataset 02, sama seperti poin 5. Total sweep 6
faktor (`1.0, 0.9, 0.8, 0.7, 0.6, 0.5`) makan ~13.5 menit buat 113.610
anchor test x 3 model x 18 step). Output: `evaluation/damping_sweep_comparison.csv`
(semua faktor x model x step) + `evaluation/recursive_evaluation_dampXX.csv` /
`recursive_mae_summary_dampXX.csv` per faktor (suffix sama seperti poin 5).

**MAE step 18, per faktor (konsisten di ketiga model)**:

| damping_factor | xgboost | lightgbm | catboost |
|---|---|---|---|
| 1.0 (baseline) | 14.325 | 13.982 | 14.171 |
| **0.9** | **13.699** | **13.548** | **13.548** |
| 0.8 | 13.933 | 13.873 | 13.868 |
| 0.7 | 14.099 | 14.065 | 14.066 |
| 0.6 | 14.202 | 14.181 | 14.184 |
| 0.5 | 14.273 | 14.258 | 14.261 |

**Temuan kunci**:
- **`damping_factor=0.9` adalah titik MINIMUM MAE di ketiga model** --
  BUKAN cuma "MAE nggak terlalu jelek dibanding baseline", tapi beneran
  LEBIH RENDAH dari baseline (`1.0`) DAN dari semua faktor lain yang
  dicoba. Ini di luar ekspektasi awal (biasanya damping = trade-off
  ratio-vs-MAE) -- ternyata overshoot yang diredam itu sendiri sumber
  error besar (exposure bias, §17 poin 4), jadi meredam delta mengurangi
  DUA masalah sekaligus (bukan tukar satu masalah dengan masalah lain).
- **`spatial_collapse_ratio` & `spatial_correlation` membaik cepat sampai
  `damping_factor=0.8`, lalu PLATEAU** (diminishing returns tajam).
  Contoh xgboost, excess ratio di atas 1: `1.0→0.9` turun 49% (1.323→0.675),
  `0.9→0.8` turun 15% lagi (→0.572), tapi `0.8→0.7→0.6→0.5` cuma turun
  total ~4% lagi (→0.550, nyaris flat). Pola sama persis di
  `spatial_correlation` (naik cepat lalu landai).
  - Di titik paling ekstrem yang dicoba (`0.5`): ratio cuma nyampe **~1.55**
    (bukan 1) dan correlation cuma **~0.19** (bukan "tinggi" seperti step 1
    yang ~0.81) -- damping TIDAK BISA menutup gap sepenuhnya, ada floor
    struktural.
- **Turun ke bawah `0.8` itu KELIRU secara trade-off** -- MAE terus naik
  (menjauh dari titik optimal 0.9 DAN dari baseline), sementara
  ratio/correlation cuma naik recehan (sudah plateau). Nggak ada
  damping_factor yang menang di semua sisi di bawah 0.8.

**KESIMPULAN**: `damping_factor=0.9` adalah nilai optimal buat
production (menang MAE + ratio/correlation, tanpa trade-off). TAPI
damping post-hoc (murni post-processing rollout, poin 5) **PUNYA BATAS
STRUKTURAL** -- nggak bisa benar-benar nutup gap ratio/correlation ke
level ideal (step 1), sesuai dugaan awal di "Belum dikerjakan" versi
sebelumnya. Root cause exposure bias (§17 poin 4) belum tersentuh dari
akarnya, cuma diredam gejalanya. **Next step wajib: noise injection**
(lihat "Belum dikerjakan" di bawah).

### 7. Noise injection: DESAIN + IMPLEMENTASI SUDAH SELESAI (BELUM di-retrain pakai data asli)

**STATUS**: kode SUDAH ditulis & di-smoke-test (fixture kecil, dataset asli
via `build_pixel_samples()` beneran + cache asli, bukan data sintetis
tebak-tebakan) -- semua invarian tervalidasi (lihat "Divalidasi" di bawah).
**BELUM dijalankan di model & data asli** (34.420 file, 756.735 anchor) --
itu tugas #1 sesi depan.

**KENAPA didesain ulang, bukan port `inject_lag_noise()` repo lama**: repo
lama nyuntik noise Gaussian INDEPENDEN ke tiap kolom lag mentah
(`tbb_13_t/tm1/tm2`, semua sama-sama unit Kelvin tunggal, gampang). Di sini
fiturnya 9 kolom AGREGAT window (mean/std/min/max/first/last/delta/
slope/n_points) -- noise independen per kolom bisa menghasilkan kombinasi
yang secara fisik nggak mungkin (mis. `min_window > max_window`, atau
`std_window` nggak sinkron sama `mean_window`/`slope`-nya), yang justru
bisa bikin model belajar pola yang SALAH, bukan robust.

**Desain yang dipakai** (di `pipeline/model_training.py::inject_recursive_style_noise()`):
1. Untuk tiap baris training (train_df, PUNYA `pixel_id`/`anchor_t0`/
   `n_points`), rekonstruksi window MENTAH dari cache raw (`.npz`) --
   teknik SAMA persis dengan yang dipakai `recursive_eval.py` buat window
   awal (IS1).
2. Titik window di posisi **> `MIN_WINDOW_SIZE`** (yaitu titik yang di
   kondisi recursive SUNGGUHAN bakal diisi prediksi model, bukan observasi
   real -- window step 1 selalu 100% real) ditambah noise Gaussian i.i.d.
   `N(0, noise_std)`. Titik 0..`MIN_WINDOW_SIZE-1` (window IS1) TIDAK
   pernah dinoise -- baris step 1 (`n_points == MIN_WINDOW_SIZE`) jadi
   no-op total.
3. Ke-9 fitur DIHITUNG ULANG dari window yang sudah dinoise, pakai fungsi
   closed-form yang SAMA dengan `recursive_eval.py` (lihat "Refactor
   pendukung" di bawah) -- ini yang jamin konsistensi fisik antar fitur
   (bukan sekadar nempel noise ke tabel fitur).
4. Hanya `X_train` yang dinoise, `X_test` TETAP bersih/real -- evaluasi
   flat (§13) tetap apple-to-apple, efek noise cuma keliatan lewat
   recursive eval (Tahap 04).
5. `noise_std <= 0` = no-op murni, backward-compatible (divalidasi byte-
   identik lewat smoke-test, lihat "Divalidasi").

**Refactor pendukung** (biar nggak duplikasi logika closed-form dari
window matrix): fungsi `_compute_window_features_local()` yang tadinya
private di `pipeline/recursive_eval.py` (§15) DIPINDAH & dijadiin public
`compute_window_features_matrix()` di `pipeline/expanding_features.py`
(modul yang nggak punya dependency ke modul pipeline lain, jadi aman
diimport dari `recursive_eval.py` DAN `model_training.py` tanpa circular
import -- `model_training.py` juga sekarang import `load_raw_cache` dari
`dataset_builder.py`, juga aman karena `dataset_builder.py` nggak
dependency ke `model_training.py`). `scripts/tools/diagnose_recursive_drift.py`
(§17 poin 3) ikut di-update importnya biar konsisten (fungsi lama yang
dihapus dari `recursive_eval.py` bakal bikin ImportError kalau nggak
di-update).

**CLI baru** (`03_train_models.py`):
```
python .\03_train_models.py --noise-std 2.5
```
`--noise-std` default `0.0` (nonaktif, perilaku lama). `--cache` opsional
(default `Config.EXPANDING_RAW_CACHE_FILE`), WAJIB ada file-nya kalau
`--noise-std > 0`. `training_summary.csv` sekarang punya kolom tambahan
`noise_std` buat traceability model mana dilatih dengan noise berapa.

**Nilai awal yang disaranin buat dicoba**: mulai dari **~2.5** (seukuran
MAE step-1 aktual di `damping_factor=0.9`, lihat §17 poin 6 -- ~2.5-2.6K
di ketiga model), BUKAN diagreb dari 0. WAJIB di-sweep (coba beberapa
nilai: 1.5, 2.5, 4.0, dst), bukan asal pilih satu angka -- pola yang sama
kayak damping_factor (ada titik optimal, terlalu besar bisa bikin model
under-fit ke sinyal asli / MAE flat naik banyak).

**Divalidasi** (smoke-test, `build_pixel_samples()` asli + cache asli,
4 pixel x 122 anchor x 18 step, BUKAN model/data asli):
- `noise_std=0` -> `train_df` dikembalikan IDENTIK (no-op sempurna,
  `df.equals()` True).
- Baris step 1 (`n_points == MIN_WINDOW_SIZE`) IDENTIK sebelum/sesudah
  noise (`np.allclose` True) -- window 100% real nggak disentuh.
- Baris step > 1 BERUBAH signifikan setelah noise (`noise_std=1.5` ->
  rata-rata |Δmean_window| ~0.23K).
- Konsistensi fisik terjaga 100% di SEMUA baris noised:
  `min_window <= mean_window/first_value/last_value <= max_window`.
- `n_points` dan kolom target TIDAK berubah (cuma 9 fitur closed-form
  yang diganti).
- End-to-end `03_train_models.py --noise-std 1.5` jalan tanpa error,
  MAE flat test set naik (0.067K -> 0.37K di xgboost, WAJAR karena model
  belajar toleransi ke window "kotor" -- test set TETAP bersih, jadi
  kenaikan MAE flat ini BUKAN indikasi bug, cuma trade-off yang
  diharapkan; robustness-nya baru keliatan lewat MAE per-step recursive,
  BELUM dicek di data asli).

### Belum dikerjakan / di luar scope sesi ini -- URGENT buat sesi depan

- **Retrain dengan `--noise-std` di data & model ASLI (34.420 file,
  756.735 anchor) BELUM DILAKUKAN.** Ini prioritas #1 sesi depan. Kode &
  desain sudah selesai (poin 7) -- cuma tinggal jalanin
  `python .\03_train_models.py --noise-std 2.5` (mulai dari situ, coba
  beberapa nilai lain juga), makan waktu training ~10 menit per run
  (BUKAN jam, TIDAK perlu rebuild dataset 02).
- **Setelah retrain, WAJIB re-run `scripts/tools/sweep_damping.py` lagi**
  (bukan cuma `04_recursive_evaluate.py` default) buat cek apakah
  `damping_factor=0.9` (§17 poin 6) masih optimal dengan model baru yang
  udah robust ke noise, atau optimal-nya geser (kemungkinan BISA geser,
  karena model baru mungkin nggak butuh redaman seagresif sebelumnya).
- Kalau hasil noise_std tertentu bagus (ratio/correlation membaik LEBIH
  jauh dari plateau ~1.55/~0.19 di §17 poin 6, TANPA MAE flat naik
  kelewat parah), PERTIMBANGKAN sweep gabungan (noise_std x
  damping_factor) buat cari kombinasi optimal, bukan tuning satu-satu
  terpisah.
- **CATATAN soal memori/klaim "exponential damping" dari sesi sebelumnya
  di luar chat ini**: sempat disebut ada fix serupa ("exponential
  damping") di repo lama dari fixes #8-#9 -- SUDAH DICEK, TIDAK ketemu di
  kode yang ke-push ke `Bandung-Weather-Forecast-Himawari-09` (kemungkinan
  itu perubahan lokal Dhika yang belum di-push). Damping di §17 poin 5 ini
  didesain dari nol berdasarkan diagnosis sesi ini, BUKAN port dari kode
  lama yang terverifikasi.
- **`damping_factor=0.9` optimal SUDAH ketemu (poin 6) -- PERTIMBANGKAN
  update `Config`** (`pipeline/config.py`) buat nyimpen nilai defaultnya
  di sana (sekarang default CLI masih hardcode `1.0` di
  `04_recursive_evaluate.py`, belum tersentralisasi kayak pola config
  lain di project ini). Belum dilakukan sesi ini -- cuma divalidasi lewat
  CLI, belum dijadikan default resmi.
- `pipeline/inference.py` / `05_run_inference.py` -- masih BELUM dibahas
  sama sekali (tetap seperti §10 poin 5 lama). Damping SUDAH divalidasi
  berhasil (poin 6) -- desain inference production nanti WAJIB pakai
  `damping_factor=0.9` dan logika damping yang sama (jangan re-derive dari
  nol, reuse `_apply_damping()` dari `recursive_eval.py` atau extract ke
  modul shared kalau perlu dipakai di kedua tempat). TAPI tunggu hasil
  noise injection dulu (poin di atas) sebelum finalisasi, karena optimal
  damping_factor bisa geser setelah retrain.

---

## 18. Investigasi & Fix Temporal Leakage + Bug Fixes + Simplifikasi (sesi lanjutan)

**STATUS SINGKAT**: sesi ini menemukan & memperbaiki masalah yang JAUH
lebih signifikan dari yang diduga sebelumnya: **temporal leakage** di
`stratified_monthly_split()` yang tidak pernah diinvestigasi di sesi mana
pun sebelum ini. Urutan kerja: audit kode-level menyeluruh (diminta Dhika,
BUKAN dari laporan bug manapun) → investigasi leakage mendalam (dibuktikan
matematis + numerik terhadap kode asli, BUKAN dugaan) → fix purge/embargo
dua sisi → **audit ULANG independen terhadap fix itu sendiri** (diminta
Dhika secara eksplisit, "jangan asumsikan kerjaan sebelumnya sudah pasti
benar") → ketemu 2 temuan dokumentasi minor, diperbaiki → retrain + re-eval
end-to-end di data produksi asli.

### 18.1 Catatan penting: CLAUDE.md sempat basi SEBELUM sesi ini pun mulai

Ditemukan lewat cek mtime file (bukan cuma baca dokumen): retrain dengan
`--noise-std 2.5` DAN damping resweep (§17 poin 7, ditandai "URGENT sesi
depan") ternyata SUDAH dijalankan Dhika di luar sesi Claude manapun --
mtime `models/*.joblib` & `evaluation/*damp*.csv` menunjukkan run terjadi
SEBELUM sesi audit ini dimulai. Artinya dokumen ini sudah basi terhadap
state repo aktual bahkan sebelum investigasi leakage dimulai. **Pelajaran
buat sesi depan**: jangan percaya "Belum dikerjakan" di CLAUDE.md tanpa
verifikasi mtime/isi file aktual dulu -- CLAUDE.md bisa telat update
relatif terhadap kerjaan yang dilakukan di luar sesi Claude.

### 18.2 Temuan utama: Temporal Leakage di `stratified_monthly_split()`

Ditemukan lewat audit kode-level menyeluruh terhadap `stratified_monthly_split()`
(`pipeline/model_training.py`). **Dibuktikan, bukan diduga** -- dengan
menjalankan fungsi ASLI (bukan reimplementasi) terhadap timeline sintetis
dan menelusuri index mentah secara eksplisit.

**Mekanisme**: split membagi train/test HANYA berdasarkan `anchor_t0`
(titik AWAL window), padahal window/target satu row bisa menjangkau sampai
`ANCHOR_SPAN-1` = 23 titik (230 menit) KE DEPAN dari `anchor_t0`-nya.
Anchor train yang `anchor_t0`-nya dekat `cutoff_time` bisa punya
window/target yang menyentuh raw observasi yang SECARA WAKTU sudah masuk
wilayah test -- bahkan bisa jadi nilai PERSIS SAMA dengan target salah
satu test row. Model jadi "melihat" (lewat agregasi fitur closed-form)
realisasi atmosfer yang nanti dia diminta prediksi buta di evaluasi.

**Skala** (diukur di dataset sintetis 1 bulan, representatif per bulan
produksi): ~0,2-0,4% baris train terkontaminasi, ~1,1-1,9% baris test
targetnya "sudah dilihat" training. Ekstrapolasi ke 8 bulan × 35 pixel
produksi: ~38.000-43.000 baris terdampak dari 11,6 juta train / 2,04 juta
test.

**Mekanisme KEDUA** ditemukan belakangan (pas implementasi fix pertama
gagal validasi, bukan diantisipasi dari awal): anchor test PALING AKHIR
suatu bulan bisa punya target yang menjorok ke BULAN BERIKUTNYA -- window
training di awal bulan berikutnya (jauh dari cutoff bulan itu sendiri)
ternyata bisa berbagi raw value dengan target test bulan sebelumnya itu.
Purge satu-sisi ("sebelum cutoff" saja) TIDAK CUKUP -- wajib dua sisi,
diterapkan lintas bulan.

### 18.3 Fix: Purge/Embargo Dua Sisi

**`Config.ANCHOR_SPAN`** (baru, `pipeline/config.py`) = `MIN_WINDOW_SIZE +
HORIZON_STEPS` -- disentralkan (sebelumnya cuma dihitung lokal di
`dataset_builder.py`), sekarang satu sumber kebenaran dipakai
`dataset_builder.py` DAN `model_training.py`.

**`Config.PURGE_STEPS`** (baru) = `ANCHOR_SPAN - 1` = 23 tick = 230 menit
(dgn default `MIN_WINDOW_SIZE=6`/`HORIZON_STEPS=18`) -- batas konservatif,
TIDAK di-hardcode, otomatis ikut berubah kalau `MIN_WINDOW_SIZE`/
`HORIZON_STEPS` diubah.

**`stratified_monthly_split()`** dapat parameter baru `purge_steps`
(default `Config.PURGE_STEPS`). Logic jadi 2-pass:
1. Pass 1: hitung `cutoff_time` tiap bulan (TIDAK BERUBAH dari sebelumnya)
   + kumpulkan interval purge DUA SISI per bulan (sebelum cutoff bulan
   itu, DAN setelah anchor test terakhir bulan itu -- sisi kedua bisa
   "menembus" ke bulan berikutnya).
2. Pass 2: terapkan SEMUA interval purge itu sebagai satu mask GLOBAL ke
   seluruh dataframe (lintas bulan, bukan per-grup independen kayak versi
   lama) -- baris di zona purge dibuang total (bukan train, bukan test).

`cutoff_time` & definisi TEST **TIDAK BERUBAH SAMA SEKALI** (`anchor_t0 >=
cutoff_time`, dihitung persis sama seperti sebelumnya) -- purge HANYA
mengurangi train. `select_test_anchors()` (Stage 4) otomatis tetap
konsisten karena manggil fungsi yang SAMA dgn default yang sama.
`purge_steps=0` mereproduksi PERSIS perilaku lama (tanpa purge) --
disediakan untuk perbandingan/debug, BUKAN default produksi.

### 18.4 Validasi: `validate_no_leakage.py` (baru, `pipeline/validate_no_leakage.py`)

Pola sama seperti `validate_expanding_features.py` -- pakai fungsi ASLI
(`find_valid_anchors`, `build_pixel_samples`, `stratified_monthly_split`,
`select_test_anchors`), timeline sintetis, BUKAN data produksi (cepat,
self-contained, WAJIB PASS sebelum purge dipercaya).

**5 skenario, SEMUA PASS**:
1. 1 bulan / 1 pixel, tanpa gap
2. 3 bulan / 2 pixel, tanpa gap
3. 4 bulan / 3 pixel, tanpa gap
4. 3 bulan / 2 pixel, DENGAN gap 48 jam persis di batas bulan
   Januari/Februari (stress-test purge lintas-bulan dgn kepadatan anchor
   tidak beraturan)
5. Sparse-month ekstrem (1 bulan cuma punya 36 anchor valid, semua di
   ujung akhir bulan)

5 check per skenario: (1) window training tidak menyentuh timestamp >=
cutoff, (2) target training tidak >= cutoff, (3) tidak ada raw value
dipakai ganda (feature train & target test), (4) test set TIDAK berubah
oleh purge, (5) `select_test_anchors()` konsisten dgn
`stratified_monthly_split()` langsung.

### 18.5 Audit ulang independen (diminta Dhika, dijalankan SETELAH fix)

Instruksi eksplisit: audit ulang MANDIRI terhadap fix itu sendiri, "jangan
asumsikan kerjaan sebelumnya sudah pasti benar". Hasil:
- Diff review baris-per-baris semua file berubah -- edge case
  (`purge_steps=0`, bulan tunggal, sparse month, gap, `ANCHOR_SPAN`
  berubah) di-derivasi manual & terbukti aman.
- Reproducibility CONFIRMED: retrain lightgbm dari nol menghasilkan MAE
  identik dgn yang dilaporkan (2,751500K), `n_train`/`n_test` persis sama.
- Semua caller fungsi yang diubah (`sweep_damping.py`,
  `diagnose_recursive_drift.py`, dll) dicek -- TIDAK ADA breaking change.
- **2 temuan minor** (dokumentasi/precondition, BUKAN bug fungsional --
  semua hasil empiris tetap benar, sudah diperbaiki):
  1. Komentar `PURGE_STEPS` di `config.py` awalnya cuma jelasin sisi
     "sebelum cutoff", lupa update setelah fix dua-sisi ditambahkan.
  2. Precondition implisit (`time_col` harus grid-aligned per
     `FREQ_MINUTES` biar matematika interval closed-boundary valid) tidak
     didokumentasikan di `stratified_monthly_split()` -- sekarang ada
     catatan eksplisit di docstring.

### 18.6 Retrain + Re-eval End-to-End (data produksi asli, split terpurge)

Model lama (noise_std=2.5, TANPA purge) di-backup ke
`_backup_before_leakage_fix_<timestamp>/` (bukan dihapus), lalu retrain
ulang dgn split yang sudah dipurge:

| Metrik | Sebelum fix | Sesudah fix | Δ |
|---|---|---|---|
| `n_train` | 11.576.250 | 11.490.570 | -85.680 (-0,74%) |
| `n_test` | 2.044.980 | 2.044.980 | **0 (identik)** |
| anchor test unik | 113.610 | 113.610 | **0 (identik)** |
| MAE flat xgboost | 2,7437K | 2,7454K | +0,06% |
| MAE flat lightgbm | 2,7497K | 2,7515K | +0,07% |
| MAE flat catboost | 2,7262K | 2,7302K | +0,15% |
| MAE recursive step 18 (xgboost) | 13,196K | 13,194K | -0,02% |
| `spatial_collapse_ratio` step 18 (xgboost) | 1,371 | 1,366 | -0,3% |

**Kesimpulan**: leakage terbukti tertutup total (0 mismatch di semua
check, 5 skenario), TAPI dampak ke angka metrik agregat yang sudah
dilaporkan sebelumnya SANGAT KECIL (<0,2% di semua metrik) -- fix ini
murni memperbaiki kebenaran metodologi evaluasi, bukan mengubah kesimpulan
performa model. Model `.joblib` & `evaluation/*.csv` di disk SEKARANG
adalah hasil split yang sudah dipurge (per 2026-08-08).

### 18.7 Bug fixes lain (di luar leakage)

| # | File | Bug | Status |
|---|---|---|---|
| A | `config.py` | `Config.__doc__` tidak valid (docstring salah posisi) | Fixed |
| B | `03_train_models.py` | Header masih nyebut nama file lama | Fixed |
| C | `validate_expanding_features.py` | Header masih nyebut path lama (`scripts/tools/`) | Fixed |
| D | `01_download_data.py` | Dead import `make_total_progress_bar` | Fixed |
| E | `config.py` | `TBB_CHANNELS`/`INTERVALS_MINUTES`/`LAG_COUNT` dead code | SENGAJA TIDAK dihapus (didokumentasikan dipertahankan utk reuse) |
| F | `config.py` | `MIN_WINDOW_SIZE`/`HORIZON_STEPS` tanpa validasi | `assert MIN_WINDOW_SIZE >= 2` & `assert HORIZON_STEPS >= 1` ditambah |

### 18.8 Simplifikasi redundant code

- `diagnose_recursive_drift.py::_rollout_with_features()` (duplikat penuh
  rollout recursive) -- **DIHAPUS**, diganti panggilan
  `recursive_eval.run_recursive_evaluation(capture_features=True)`
  (parameter baru, default `False` = perilaku lama, backward-compatible).
  Sekarang cuma ada SATU implementasi rollout recursive di seluruh
  pipeline.
- `compare_interior_vs_edge_spatial_metrics.py::_spatial_metrics_for_subset()`
  -- logic groupby/aggregate duplikat dihapus, sekarang wrapper tipis di
  atas `recursive_eval._spatial_metrics_per_step()`.

### Belum dikerjakan / status terkini (menggantikan list "URGENT" di §17)

- ~~Retrain dgn `--noise-std` di data asli~~ -- **SUDAH**, 2x malah
  (sekali dgn split lama/leaky di luar sesi terdokumentasi -- lihat
  §18.1, sekali lagi dgn split terpurge di sesi ini -- lihat §18.6).
- ~~Re-sweep `damping_factor` pasca-retrain+purge~~ -- **SUDAH**, lihat
  §18.9 di bawah. **KESIMPULAN BERUBAH dari §17 poin 6**: `damping_factor=0.9`
  TIDAK LAGI optimal.
- `pipeline/inference.py` / `05_run_inference.py` -- masih BELUM dibahas
  sama sekali (tetap seperti §10/§17).
- `_backup_before_leakage_fix_<timestamp>/` ada di root repo -- bukan
  bagian permanen pipeline, hapus manual kalau sudah tidak diperlukan.

### 18.9 Hasil re-sweep `damping_factor` (model noise_std=2.5 + split terpurge)

Dijalankan `scripts/tools/sweep_damping.py` (6 faktor: `1.0, 0.9, 0.8, 0.7,
0.6, 0.5`, ~12 menit, 113.610 anchor × 3 model × 18 step). Output:
`evaluation/damping_sweep_comparison.csv` (menimpa hasil sweep lama yang
dijalankan terhadap model SEBELUM fix leakage -- lihat §18.1, sweep lama
itu juga sudah basi terhadap model saat ini, bukan cuma karena leakage
tapi karena noise injection juga sudah mengubah karakteristik model).

**MAE step 18** (konsisten di ketiga model, MENURUN monoton dari
`damping_factor` kecil ke besar -- KEBALIKAN dari pola §17 poin 6):

| damping_factor | xgboost | lightgbm | catboost |
|---|---|---|---|
| **1.0 (tanpa redaman)** | **13.194** | **13.157** | **13.182** |
| 0.9 | 13.422 | 13.404 | 13.462 |
| 0.8 | 13.805 | 13.792 | 13.816 |
| 0.7 | 14.023 | 14.011 | 14.014 |
| 0.6 | 14.149 | 14.135 | 14.132 |
| 0.5 | 14.230 | 14.215 | 14.208 |

**KOREKSI PENTING (ditemukan Dhika lewat pertanyaan lanjutan, sesi ini
juga)**: kesimpulan "1.0 menang" di atas HANYA benar untuk step 18 /
rata-rata seluruh horizon -- BUKAN benar di setiap step. Ada **pola
crossover** antara `damping_factor=1.0` dan `0.9` yang tidak kelihatan
kalau cuma lihat step 18 atau rata-rata. Tabel di bawah: `diff_pct` =
`(MAE@0.9 - MAE@1.0) / MAE@1.0 * 100` -- NEGATIF berarti `0.9` lebih baik
(MAE lebih rendah), POSITIF berarti `1.0` lebih baik.

| step | xgboost | lightgbm | catboost | pemenang |
|---|---|---|---|---|
| 1 | 0,00% | 0,00% | 0,00% | seri (damping tidak berlaku di step 1, by design) |
| 2 | -0,31% | -0,35% | -0,34% | `0.9` |
| 3 | -0,38% | -0,47% | -0,48% | `0.9` |
| 4 | -0,45% | -0,53% | -0,58% | `0.9` |
| 5 | -0,54% | -0,57% | -0,63% | `0.9` |
| 6 | -0,55% | -0,58% | -0,65% | `0.9` (margin terbesar `0.9`) |
| 7 | -0,53% | -0,52% | -0,56% | `0.9` |
| 8 | -0,42% | -0,38% | -0,44% | `0.9` |
| 9 | -0,34% | -0,26% | -0,36% | `0.9` |
| 10 | -0,25% | -0,15% | -0,22% | `0.9` |
| 11 | -0,13% | +0,01% | -0,04% | ~seri (titik crossover) |
| 12 | +0,03% | +0,20% | +0,23% | `1.0` (mulai unggul) |
| 13 | +0,22% | +0,39% | +0,48% | `1.0` |
| 14 | +0,47% | +0,64% | +0,81% | `1.0` |
| 15 | +0,74% | +0,92% | +1,13% | `1.0` |
| 16 | +1,07% | +1,21% | +1,42% | `1.0` |
| 17 | +1,40% | +1,57% | +1,74% | `1.0` |
| 18 | +1,73% | +1,88% | +2,12% | `1.0` (margin terbesar `1.0`) |

**Baca polanya**: `damping_factor=0.9` menang TIPIS di horizon
menengah-pendek (step 2-11, margin 0,01%-0,65%), tapi `damping_factor=1.0`
menang JELAS dan MEMBESAR di horizon panjang (step 12-18, margin
0,03%-2,12%, sampai ~3-10x lebih besar dari margin kemenangan `0.9` di
step awal). Ini BUKAN hubungan monoton "1.0 selalu terbaik" -- ada
crossover nyata di sekitar step 11-12.

**KENAPA `damping_factor=1.0` tetap dipilih jadi konfigurasi final**
(BUKAN karena menang di semua step -- karena TIDAK): **prioritas use-case
pipeline ini adalah horizon panjang (step 12-18)**. Metode rekursif
membuat error terakumulasi paling signifikan justru di step-step akhir --
itu ujian utama kualitas model (bisa nge-generalize sejauh apa rollout-nya
sebelum divergen), bukan step-step awal yang secara inheren lebih gampang
diprediksi (window masih didominasi observasi real). Karena prioritasnya
di situ, margin kemenangan `1.0` yang MEMBESAR justru di step 12-18 (bukan
mengecil) langsung relevan dengan tujuan pipeline -- sementara margin
kekalahan `1.0` di step 2-11 kecil (<0,65%) dan di area yang bukan fokus
utama. Kalau prioritas use-case berbeda (mis. cuma peduli forecast <2 jam
ke depan / step <=11), kesimpulannya bisa BALIK ke `0.9`.

**`spatial_collapse_ratio`**: hampir flat antara `1.0` (1,366) dan `0.9`
(1,362) untuk xgboost (beda nyaris tidak signifikan), lalu justru MEMBURUK
(menjauh dari 1) seiring damping_factor turun lebih jauh (0,8→1,412,
0,5→1,484) -- KEBALIKAN dari pola §17 poin 6 (dulu ratio membaik seiring
damping ditambah).

**`spatial_correlation`**: satu-satunya metrik yang masih membaik dgn
damping (0,175 di `1.0` → 0,199 di `0,5`), tapi kenaikannya landai
(diminishing returns tajam setelah `0.7`) dan TIDAK cukup untuk
mengkompensasi kenaikan MAE yang jauh lebih besar secara relatif di step
12-18.

**Apakah selisih ini noise atau sinyal nyata?** Pipeline ini
**deterministik sepenuhnya** (dibuktikan lewat reproducibility check §18.5
-- retrain dgn config sama menghasilkan MAE identik sampai 6 digit
desimal). Jadi SEMUA selisih di tabel atas BUKAN noise statistik --
murni efek eksak & 100% reproducible dari formula damping, bukan variasi
run-to-run. Signifikansi *praktis*-nya beda per rentang step: di step
2-11 selisihnya sebanding dgn angka <0,2-0,4% yang dianggap "tidak
signifikan" di perbandingan before/after fix leakage (§18.6) -- praktis
diabaikan. Di step 15-18 selisihnya 3-10x lebih besar dari baseline itu --
nyata, bukan derau.

**KESIMPULAN FINAL (menggantikan §17 poin 6)**: `damping_factor=1.0`
(tanpa redaman) dipilih sebagai **konfigurasi produksi**, dengan alasan
eksplisit **prioritas horizon panjang** di atas -- BUKAN karena unggul di
seluruh horizon (ia TIDAK, lihat tabel crossover). **Root cause pergeseran
dari §17 poin 6** (dulu `0.9` optimal di SEMUA step): noise injection
(§17 poin 7) ternyata SENDIRIAN sudah menutup sebagian besar gap exposure
bias yang dulu jadi alasan damping dibutuhkan (§17 poin 4) -- damping
post-hoc yang dulu perlu buat "mengerem" model yang belum robust ke
window kotor, sekarang di horizon panjang malah cuma menambah bias
(menarik forecast ke persistence) tanpa manfaat yang sepadan lagi;
di horizon pendek-menengah efeknya masih sedikit menguntungkan tapi kecil.
**Rekomendasi**: PAKAI `damping_factor=1.0` (default, tanpa perlu
diaktifkan) untuk model saat ini -- JANGAN otomatis pakai `0.9` lagi
seperti rekomendasi §17, itu sudah usang UNTUK USE-CASE INI. Kalau nanti
noise_std, skema training, ATAU prioritas horizon (mis. jadi lebih peduli
step awal) berubah, WAJIB re-sweep ulang & re-evaluasi trade-off per
rentang step -- jangan asumsikan `1.0` tetap optimal selamanya atau untuk
semua tujuan.

---