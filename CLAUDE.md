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
│   ├── 05_run_inference.py          # ✅ SELESAI -- forecast produksi, auto-detect
│   │                                 # model dari hasil 04, flag --t0/--model/--damping-factor (§19)
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
│   │   ├── inference.py             # ✅ SELESAI -- lihat §19
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
- ~~`pipeline/inference.py` / `05_run_inference.py`~~ -- **SUDAH**, lihat §19.
- `_backup_before_leakage_fix_<timestamp>/` ada di root repo -- bukan
  bagian permanen pipeline, hapus manual kalau sudah tidak diperlukan.
- `pipeline/visualize.py` / `06_visualize.py` -- masih BELUM dibahas sama
  sekali (tetap seperti §10).

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

## 19. Stage 05: `pipeline/inference.py` + `scripts/05_run_inference.py` (SELESAI)

**STATUS SINGKAT**: Stage 05 (inference) yang dari awal §10 ditandai "BELUM
dibahas sama sekali, jangan diasumsikan desainnya" sekarang sudah dirancang
& diimplementasi lewat sesi diskusi terpisah (pakai plan mode, plan
DITOLAK sekali & direvisi berdasar feedback konkret sebelum diterima) +
smoke-test end-to-end pakai data asli `data_bandung/` (bukan sintetis).
`06_visualize.py` -- lihat §20 (SELESAI, sesi terpisah setelah ini).

### 19.1 Keputusan desain (final, dikonfirmasi user)

1. **Trigger**: batch terjadwal, full grid 35 pixel. Scheduling (cron/Task
   Scheduler) di luar scope script ini -- sama seperti 01-04, tidak
   self-schedule, cuma perlu aman dijalankan berulang.
2. **Sumber data**: baca file `.nc` yang SUDAH ada di `data_bandung/`
   langsung -- BUKAN fetch FTP baru, BUKAN pakai `EXPANDING_RAW_CACHE_FILE`
   (cache itu snapshot dataset training, bukan data real-time terbaru).
3. **Model**: SATU model production, dipilih **OTOMATIS** dari hasil
   AKTUAL `04_recursive_evaluate.py` (`evaluation/recursive_mae_summary.csv`,
   file baseline damping_factor=1.0) -- **TIDAK di-hardcode** (revisi dari
   plan pertama yang sempat hardcode `"lightgbm"`). Kriteria: rata-rata MAE
   terendah di `Config.INFERENCE_PRIORITY_STEP_RANGE = (12, 18)` (prioritas
   horizon panjang, ikut kesimpulan final §18.9). User tetap bisa override
   manual via `--model`.
4. **t0 (titik referensi waktu)**: default = data TERBARU yang tersedia
   (BUKAN t0 tetap/hardcode) -- tapi bisa dioverride manual via `--t0`
   (revisi dari plan pertama yang cuma asumsikan "selalu data terbaru").
   **Semantik `--t0`**: titik waktu observasi TERAKHIR yang jadi basis
   window (BUKAN `anchor_t0` versi codebase yang berarti titik AWAL
   window) -- forecast dimulai dari `t0 + FREQ_MINUTES menit`. Dipilih
   makna ini karena paling natural buat user ("data terbaru 2 Juli, t0 =
   2 Juli").
5. **Output**: DUA file per run -- CSV tabular (12 kolom, satu baris per
   pixel×step) + GeoJSON (SATU `FeatureCollection` per run, satu `Feature`
   per pixel/Point, 18 step forecast di-embed di `properties.forecast`,
   BUKAN 18 file terpisah). `scripts/geojson/KotaBandung.geojson` yang
   sudah ada di repo TERNYATA cuma batas administratif 30 kecamatan (GADM,
   `NAME_3`/`TYPE_3`="Kecamatan"), TIDAK match skema grid pixel
   (`lat_idx`/`lon_idx`/`pixel_id`) -- jadi GeoJSON forecast ini file BARU
   & independen, bukan hasil join ke file itu.
6. **Retensi**: TIMESTAMPED, riwayat semua run disimpan (BUKAN overwrite
   "latest") -- **nama model DAN t0 ikut masuk ke nama file** (revisi dari
   plan pertama yang cuma nama model di kolom CSV, lalu direvisi LAGI
   setelah Dhika minta t0 juga masuk nama file -- lihat CATATAN REVISI).
   **KOREKSI (§21, sesi lanjutan)**: struktur asli di sini FLAT (2 file
   langsung di `forecast_output/`) SUDAH DIRESTRUKTUR jadi folder-per-run
   (`forecast_output/{model_name}_t0..._run.../forecast.csv`/`.geojson`,
   pola sama dgn `visualizations/` Stage 06) -- lihat §21 utk struktur
   final & alasannya, bagian di bawah ini historis/sudah usang:
   ~~`forecast_output/forecast_{model_name}_t0{YYYYMMDD_HHMM}_run{YYYYMMDD_HHMM}.csv`
   / `.geojson`~~. `t0` = `window_end_time` yang BENAR-BENAR kepakai (titik
   observasi terakhir dari window, otomatis terisi walau `--t0` tidak
   dioverride -- representasi "forecast ini buat kapan"), `run` = kapan
   script-nya dieksekusi (representasi "kapan file ini dibuat" -- beda
   dari `t0` kalau backtest pakai `--t0` masa lalu). Sampai MENIT, bukan
   detik.

### 19.2 Implementasi

**`pipeline/inference.py`** (fungsi baru, urutan pemanggilan dari `run_inference()`):
- `load_recent_window_data(data_dir, tail_files, target_t0, ...)` -- baca
  `Config.INFERENCE_TAIL_FILES` (100) file TERAKHIR (mundur dari `target_t0`
  kalau diisi, BUKAN dari file paling akhir -- supaya `--t0` di masa lalu
  tidak salah baca file yang tidak relevan/buang I/O). Reuse VERBATIM
  `dataset_builder.discover_nc_files/build_uniform_timeline/load_pixel_grid`.
  **WAJIB slice `entries` SEBELUM dilempar ke 2 fungsi itu** -- keduanya
  baca SEMUA entries tanpa filter internal, lupa slice = re-trigger
  bottleneck I/O berjam-jam (§16).
- `select_anchor_per_pixel(data_matrix, timeline, pixel_meta, window_size, target_t0)`
  -- reuse VERBATIM `dataset_builder.find_valid_anchors(span=MIN_WINDOW_SIZE)`
  (BUKAN `ANCHOR_SPAN` -- inference cuma butuh window bootstrap bebas-gap,
  bukan horizon depan bebas-gap karena itu justru yang mau diprediksi),
  ambil anchor PALING BARU yang window-nya berakhir di/sebelum `target_t0`
  (atau tanpa batas kalau `target_t0=None`). Pixel tanpa anchor valid =
  skip dgn counter (bukan crash), pola sama seperti `_map_anchors_to_indices()`.
- `select_inference_model(eval_summary_path, priority_step_range)` -- baca
  `recursive_mae_summary.csv`, filter step di `priority_step_range`,
  `groupby("model")["mae"].mean()`, ambil terkecil. Return juga
  `ranking_df` lengkap (transparansi, CLI print alasan pemilihan, bukan
  black-box). File tidak ada -> `FileNotFoundError` arahkan ke `04_recursive_evaluate.py`.
- `run_forecast_rollout(model, anchors_df, data_matrix, timeline, damping_factor)`
  -- loop rollout BARU (bukan panggil `run_recursive_evaluation()`
  langsung): alasan konkret, `run_recursive_evaluation()` meng-index
  `timeline[target_idx]`/`data_matrix[target_idx,...]` karena anchor
  evaluasi historis (window & target sama-sama ada di matrix). Target
  masa depan inference TIDAK ADA di matrix -- `target_time` WAJIB dihitung
  via aritmatika waktu (`anchor_t0 + L*FREQ_MINUTES`, `L`=panjang window
  SEBELUM extend step ini -- identik matematis dgn `target_idx=starts+L`
  di `run_recursive_evaluation`, diverifikasi manual formula-nya sama
  persis, cuma domain beda: waktu vs index). Yang di-reuse (bukan
  diduplikasi): `compute_window_features_matrix()` & `_apply_damping()`
  (import langsung dari `recursive_eval.py`, BUKAN re-derive formula).
- `build_geojson_feature_collection()`, `save_forecast_outputs()`,
  `run_inference()` (orchestrator) -- lihat source untuk detail.

**`scripts/05_run_inference.py`**: ikuti pola persis `04_recursive_evaluate.py`.
Flag: `--data-dir`, `--models-dir`, `--model` (default `None`=auto-detect),
`--t0` (default `None`=data terbaru), `--tail-files` (default `Config.INFERENCE_TAIL_FILES`),
`--damping-factor` (default `1.0`, help text eksplisit bilang "0.9 sudah
usang, lihat §18.9"), `--output-dir`, `--workers`.

**`config.py`** tambahan: `INFERENCE_TAIL_FILES=100` (~16,7 jam cakupan,
biaya I/O ~54 detik terukur -- lihat 19.3), `INFERENCE_PRIORITY_STEP_RANGE=(12,18)`,
`INFERENCE_DIR=forecast_output/` (sudah ada di `.gitignore`). **TIDAK ADA**
`INFERENCE_MODEL_NAME` (sengaja, model tidak hardcode -- lihat 19.1 poin 3).

### 19.3 Verifikasi (data ASLI `data_bandung/`, sampai 2026-07-31, BUKAN sintetis)

Semua PASS:
1. **Smoke run default** (`--tail-files 30 --workers 1`): 35/35 pixel
   ke-forecast, model auto-detect = `lightgbm` (avg MAE=12,0054K, cocok
   persis dgn ranking manual §18.9), 2 file output ke-generate dgn nama
   `forecast_lightgbm_<timestamp>.{csv,geojson}`.
2. **Struktur output**: CSV 12 kolom persis sesuai spesifikasi, 630 baris
   (35 pixel × 18 step, tanpa gap/dupe step per pixel), `target_time` naik
   tepat 10 menit per step. GeoJSON valid, 35 feature (1 per pixel),
   `geometry.coordinates` cocok `[longitude, latitude]` pixel, tiap
   `properties.forecast` persis 18 entri.
3. **Sanity damping**: `--damping-factor 1.0` (default) -> `y_pred ==
   y_pred_raw` di SEMUA baris (`_apply_damping` benar2 no-op). `--damping-factor
   0.8` -> identik di step 1 (formula no-op by design di step 1), BEDA di
   step>=2 (delta diredam ke arah `last_value`, terverifikasi arah &
   magnitude-nya masuk akal).
4. **Model manual override** (`--model catboost`): log eksplisit bilang
   "manual via --model" (bukan auto-detect), nama file pakai `forecast_catboost_*`.
5. **`--t0` historis** (`--t0 "2026-03-15 12:00"`): terverifikasi
   `window_end_time` yang dipakai PERSIS `2026-03-15 12:00:00` (bukan data
   terbaru/2026-07-31) -- pembacaan file terverifikasi mundur dari t0 (cuma
   baca file di sekitar Maret, BUKAN baca s.d. Juli lalu filter).
6. **Skip-path total**: `--tail-files 1` (dijamin < `MIN_WINDOW_SIZE=6`)
   -> `ValueError` jelas ("naikkan --tail-files..."), ditangkap CLI dgn
   rapi (bukan traceback mentah), exit bersih.
7. **Full-default run** (`--tail-files 100`, workers default, data asli):
   **54,3 detik wall-clock** (2 kali run, konsisten), 35/35 pixel
   ke-forecast (0 skip -- data terbaru bersih, tanpa gap dekat ujung).
   Jauh di bawah estimasi worst-case plan (45-130s) -- di ujung cepat
   karena data bersih (tidak perlu scan jauh ke belakang buat cari window
   valid).

**Partial-skip TIDAK dites eksplisit di data asli** (data terbaru saat
verifikasi ini kebetulan bersih, tidak ada pixel yang gagal) -- tapi
mekanismenya (per-pixel `find_valid_anchors`) adalah fungsi yang SAMA
persis yang sudah divalidasi ekstensif di `validate_no_leakage.py`
(skenario gap & sparse-month, §18.4), jadi risikonya rendah. Kalau
operasional nanti nemuin kasus partial-skip yang aneh, cek dulu ke situ
sebelum curiga ada bug baru di `inference.py`.

### 19.4 Belum dikerjakan / di luar scope

- `pipeline/visualize.py` / `06_visualize.py` -- SELESAI, lihat §20.
- Scheduling OS-level (Windows Task Scheduler / cron) buat batch terjadwal
  -- di luar scope kode ini, tanggung jawab operasional user.
- Partial-skip belum diverifikasi di data asli (lihat 19.3) -- kalau mau
  lebih yakin, bisa dites pakai `--tail-files` kecil (6-10) di periode
  data yang diketahui ada gap/cloud cover, atau tunggu kejadian natural.

### CATATAN REVISI (setelah smoke-test awal, dari feedback Dhika langsung)

- **Format timestamp nama file** direvisi dari `%Y%m%d_%H%M%S` (sampai
  detik) jadi `%Y%m%d_%H%M` (sampai menit saja) -- nama file lebih ringkas,
  cukup buat retensi run batch terjadwal. Trade-off: 2 run model+t0 yang
  sama dalam menit yang sama akan saling timpa (jarang terjadi buat batch
  terjadwal, bukan on-demand rapid-fire).
- **`t0` ditambahkan ke nama file** (revisi kedua, Dhika masih bingung
  retensi cuma berdasar `run_timestamp` -- nggak kelihatan dari nama file
  itu forecast buat t0 KAPAN, cuma keliatan kapan filenya DIBUAT). Fungsi
  `save_forecast_outputs()` dapat parameter baru `t0_label` (diisi
  `run_inference()` dari `anchors_df["window_end_time"].max()` -- SELALU
  ada nilai, baik `--t0` dioverride maupun pakai data terbaru). Nama file
  final: `forecast_{model}_t0{YYYYMMDD_HHMM}_run{YYYYMMDD_HHMM}.{csv,geojson}`.
  Divalidasi 2 skenario: `--t0 "2026-05-10 06:00:00"` ->
  `forecast_lightgbm_t020260510_0600_run20260809_0105.csv`; default (data
  terbaru) -> `forecast_lightgbm_t020260731_2320_run20260809_0106.csv`
  (t0 otomatis = window_end_time data terbaru, BUKAN "unknown"/kosong).
- **`y_pred` == `y_pred_raw` di CSV** sempat dikira aneh/bug oleh Dhika --
  **BUKAN bug**, ini perilaku BY DESIGN: default `--damping-factor=1.0`
  bikin `_apply_damping()` no-op (return `raw_pred` apa adanya), jadi kedua
  kolom identik. Pola ini SAMA PERSIS dgn `recursive_evaluation.csv` di
  Stage 4 (§17 poin 5) -- `y_pred_raw` disimpan buat perbandingan kalau
  `--damping-factor < 1.0` dipakai (kolom baru beda, sudah diverifikasi di
  19.3 poin 3). Tidak ada perubahan kode buat ini, cuma klarifikasi.
- **`anchor_t0` (22:30) vs `target_time` step 1 (23:30) beda 60 menit**,
  bukan 10 menit -- sempat bikin bingung Dhika. **BUKAN bug**: `anchor_t0`
  = titik AWAL window (6 titik observasi, `MIN_WINDOW_SIZE`), bukan titik
  observasi TERAKHIR. Step 1 = 10 menit setelah titik TERAKHIR window
  (`window_end_time`), bukan 10 menit setelah `anchor_t0`. Jaraknya =
  `MIN_WINDOW_SIZE*FREQ_MINUTES` = 60 menit, persis definisi IS1=[1..6]/
  OS1=[7] di CLAUDE.md §2. Konsekuensi praktis: **`--t0` = `window_end_time`**
  (titik observasi TERAKHIR, sesuai desain 19.1 poin 4), BUKAN `anchor_t0`
  -- jadi kalau mau forecast 3 jam ke depan DARI jam 06:00, isi `--t0
  06:00:00` LANGSUNG, JANGAN dikurangi 1 jam jadi `05:00:00` (sudah
  diverifikasi: `--t0 06:00:00` -> `window_end_time=06:00:00`,
  `anchor_t0` otomatis mundur sendiri jadi `05:10:00`, step 1 =
  `06:10:00`, step 18 = `09:00:00`).

### 19.5 Fitur tambahan: `y_true`/`abs_error` (perbandingan ke observasi asli, opsional)

Diminta Dhika setelah lihat output pertama: tambahkan nilai `tbb_13` ASLI
(bukan cuma prediksi) buat tahu seberapa besar error tiap step -- berguna
sebagai **backtest manual**: kalau `--t0` diisi ke masa lalu yang "masa
depan"-nya (target_time) kebetulan SUDAH ada di `data_bandung/` (karena
arsip lokal sudah py unya data s.d. 2026-07-31), bisa langsung dibandingkan
prediksi vs kenyataan tanpa perlu tunggu waktu beneran lewat.

**`load_actual_values(detail_df, data_dir, ...)`** (baru, `pipeline/inference.py`):
scan `data_bandung/` buat file yang timestamp-nya masuk rentang
`[min(target_time), max(target_time)]` dari `detail_df`, baca via
`load_pixel_grid` (reuse verbatim), bangun lookup `{(pixel_id, timestamp):
nilai}`. Kalau TIDAK ADA file yang cocok (forecast produksi murni, masa
depan genuine belum terjadi/didownload) -> return dict KOSONG, BUKAN
error -- ini kondisi NORMAL, bukan kegagalan.

`run_inference()` panggil ini SETELAH `run_forecast_rollout()` (sengaja
dipisah -- rollout tetap fokus prediksi murni, pencarian observasi asli
jadi langkah opsional terpisah yang boleh kosong tanpa mengganggu hasil
utama), lalu isi kolom `y_true` (lookup per `(pixel_id, target_time)`,
`NaN` kalau tidak ketemu) & `abs_error = abs(y_pred - y_true)` (otomatis
`NaN` juga kalau `y_true` NaN).

**Output**: CSV `CSV_COLUMNS` bertambah jadi 14 kolom (`y_true`,
`abs_error` disisipkan setelah `y_pred_raw`, sebelum `model_name`/
`damping_factor`). GeoJSON: tiap entri `forecast` di `properties` juga
dapat `y_true`/`abs_error` -- **`None` (JSON `null`) buat NaN**, BUKAN
`NaN` literal (itu bukan JSON valid, `json.dump` Python defaultnya nulis
`NaN` literal kalau tidak di-guard manual -- WAJIB `None if pd.isna(...)
else float(...)`). CLI (`05_run_inference.py`) print tabel MAE aktual per
step di ringkasan akhir KALAU ada minimal 1 baris dgn `y_true` non-NaN,
kalau tidak ada sama sekali cuma info singkat (bukan tabel kosong).

**Divalidasi** (data asli, 2 skenario):
- `--t0 "2026-05-10 06:00:00"` (masa lalu, ~2,5 bulan sebelum data
  terakhir 2026-07-31): **630/630 baris** (35 pixel × 18 step) ketemu
  observasi asli, MAE aktual per step ke-print (step 1 ~1,27K naik ke
  step 13 ~8,58K, turun lagi ke step 18 ~6,86K -- wajar untuk SATU
  instance/anchor tunggal, bukan rata-rata ribuan anchor kayak
  `recursive_mae_summary.csv`, jadi tidak harus monoton naik).
- Default (data terbaru, target_time genuinely di masa depan 2026-07-31
  23:30 dst): **0/630 baris** ketemu observasi (diverifikasi CSV `y_true`
  semua `NaN`, GeoJSON semua `null`, valid JSON -- di-parse ulang berhasil
  tanpa error), CLI print info "tidak ada observasi asli" (bukan tabel
  kosong/crash).

### 19.6 Audit ulang kode Stage 05 (diminta Dhika, setelah semua fitur di atas)

Review baris-per-baris seluruh `pipeline/inference.py` + `05_run_inference.py`
+ tambahan `config.py`. **1 bug nyata & berisiko tinggi ketemu, sudah
diperbaiki & diverifikasi**:

**BUG: `--tail-files 0` (atau negatif) tidak divalidasi, diam-diam baca
SELURUH riwayat `data_bandung/`** -- `entries[-tail_files:]` di Python
TIDAK error untuk `tail_files<=0`: `entries[-0:]` (tail_files=0) balik
jadi `entries[0:]` (SEMUA entries, karena `-0 == 0` di Python -- tidak ada
negative zero buat integer), dan `entries[-(-5):]` (tail_files=-5) jadi
`entries[5:]` (bukan "5 file terakhir", tapi "semua KECUALI 5 file
pertama"). Tanpa guard, ini re-trigger PERSIS bottleneck I/O berjam-jam
(§16) yang jadi alasan `--tail-files` ini didesain sejak awal. **Diperparah
bug kedua**: log ringkasan CLI pakai `args.tail_files or
cfg.INFERENCE_TAIL_FILES` -- `0` itu falsy di Python, jadi `0 or 100`
jatuh ke `100`, log salah nampilin "Tail files: 100" padahal yang
BENERAN dipakai `0` (mustinya paling gampang ketauan operator justru dari
log ini, tapi malah nyembunyiin masalahnya).

**Fix**: (1) `load_recent_window_data()` sekarang validasi eksplisit
`if tail_files < 1: raise ValueError(...)` SEBELUM slicing manapun
(melindungi kedua jalur -- `--t0` diisi maupun tidak, karena keduanya
pakai pola slicing yang sama rentan). (2) Log CLI diganti dari `args.tail_files
or cfg.INFERENCE_TAIL_FILES` jadi `args.tail_files if args.tail_files is
not None else cfg.INFERENCE_TAIL_FILES` (`is not None`, bukan `or` --
supaya `0` tetap tampil sebagai `0`, bukan disulap jadi default).

**Divalidasi**: `--tail-files 0` -> error cepat & jelas ("--tail-files
harus >= 1, dapat 0"), TIDAK mencoba baca file sama sekali. `--tail-files
-5` -> error serupa. Regression test `--tail-files 30` (jalur normal) --
tetap jalan seperti biasa, 35/35 pixel, output tidak berubah.

**Area lain yang diperiksa TIDAK ketemu bug** (diverifikasi logikanya,
bukan cuma dibaca sekilas): `select_anchor_per_pixel()`'s pembatasan
`target_t0` (redundan-tapi-aman terhadap `load_recent_window_data`,
bukan bug); asumsi `window_size` seragam lintas pixel di
`run_forecast_rollout()` (valid, `window_size` memang parameter tunggal
bukan per-pixel); alignment `pixel_id` antara `load_actual_values()`'s
`load_pixel_grid()` terpisah vs yang dipakai forecast (aman, sama-sama
dari grid canonical yang konsisten); NaN handling di `abs_error` &
GeoJSON `None`-cast (benar); kemungkinan `HORIZON_STEPS=0`/`MIN_WINDOW_SIZE<2`
bikin crash di rollout (sudah dicegah `assert` di `config.py`, tidak
reachable dari CLI manapun).

---

## 20. Stage 06: `pipeline/visualize.py` + `scripts/06_visualize.py` (BARU, SELESAI)

**STATUS SINGKAT**: render hasil forecast Stage 05 jadi animasi GIF 6-panel
peta. Plan mode dipakai, plan DITOLAK DUA KALI sebelum diterima (revisi
pertama: user minta kembali ditanya pertanyaan klarifikasi yang sempat
ke-skip krn salah pencet tool; revisi kedua & paling penting: draft awal
sempat punya 1 panel TEKS info -- user tegas koreksi "3. semuanya dalam
bentuk map", SEMUA 6 panel WAJIB peta/spatial plot, metadata jadi
title/caption saja). Diimplementasi + diverifikasi end-to-end pakai CSV
forecast ASLI yang sudah ada di `forecast_output/` (bukan data sintetis).

### 20.1 Layout final (2x3, SEMUA 6 panel = peta, TIDAK ADA panel teks)

| Baris | Kolom 1 | Kolom 2 | Kolom 3 |
|---|---|---|---|
| **1** | **Input @ t0**: peta observasi TBB asli di titik observasi TERAKHIR sebelum forecast mulai (`window_end_time`) | **Prediksi**: peta `y_pred` step berjalan | **Aktual**: peta `y_true` (placeholder-map kalau NaN semua) |
| **2** | **Kelas Awan**: peta kategorikal dari `y_pred` (`tidak hujan`/`mendung`/`hujan`) | **Risiko Banjir**: peta kategorikal dari `y_pred` SAMA (breakpoint sama, label beda: `aman`/`waspada`/`bahaya`) | **Error Map**: peta `\|y_pred - y_true\|` (placeholder-map kalau NaN semua) |

`fig.suptitle`: model, damping_factor, t0, target_time, step X/18, MAE step
ini (atau "n/a" kalau tidak ada observasi), ringkasan status (jumlah pixel
aman/waspada/bahaya). Metadata TIDAK PERNAH jadi panel tersendiri --
placeholder-map (Aktual/Error saat NaN semua) tetap render frame geografis
penuh (extent + boundary overlay + teks di tengah), BUKAN axis kosong
polos, supaya "tetap map" konsisten di semua 6 slot bahkan saat data
kosong.

### 20.2 Masalah desain: panel "Input @ t0" butuh data yang TIDAK ADA di CSV Stage 05

CSV Stage 05 (`inference.CSV_COLUMNS`) cuma simpan `y_pred`/`y_true`/
`abs_error` per (pixel, step) -- TIDAK simpan observasi mentah di titik t0
itu sendiri. Solusi: `t0` per pixel diturunkan dari data yang SUDAH ADA
(`compute_t0_per_pixel()`: `window_end_time = target_time(step=1) -
FREQ_MINUTES`, aritmatika sederhana, row-level bukan diasumsikan sama
semua pixel), lalu file `.nc` yang relevan dibaca ULANG dari
`data_bandung/` -- **BUKAN implementasi baru**, reuse fungsi Stage 05.

**Refactor pendukung** (`pipeline/inference.py`): logic inti
`load_actual_values()` (scan `data_dir` utk rentang waktu, baca via
`load_pixel_grid`, bangun dict `{(pixel_id, timestamp): value}`)
DIEKSTRAK jadi fungsi public baru `load_raw_values_lookup(data_dir,
min_time, max_time, freq_minutes=None, n_workers=None) -> dict`.
`load_actual_values()` sekarang wrapper TIPIS di atasnya (hitung
`min_time`/`max_time` dari `target_time`, panggil fungsi baru, tambah
logging `n_found`/`n_possible` sendiri) -- **perilaku TIDAK BERUBAH**,
diverifikasi regression test manual (`--t0 "2026-05-10 06:00:00"
--tail-files 30 --workers 1 --model lightgbm"` -> tetap "630/630 baris
forecast" identik sebelum & sesudah refactor). `pipeline/visualize.py`
import & reuse fungsi yang SAMA (`load_input_snapshot()` di
`visualize.py` panggil `load_raw_values_lookup()` dgn rentang waktu beda:
`window_end_time` per pixel, bukan `target_time`).

### 20.3 Geometri peta: extent & aspect ratio DIHITUNG DARI CONFIG, BUKAN dari CSV

`compute_grid_geometry()` hitung `extent=[lon_min,lon_max,lat_min,lat_max]`
(padding setengah-sel) dari **`Config.BANDUNG_LAT_MIN/MAX`,
`LON_MIN/MAX`** (bounding box download tetap) + `Config.PIXEL_GRID_SHAPE`
-- **BUKAN** dari lat/lon pixel yang HADIR di CSV forecast tertentu.
Alasan: kalau dihitung dari data yang hadir, extent bisa "menyusut" diam-
diam kalau kebetulan pixel TEPI di-skip Stage 05 (window pixel itu invalid/
gap), bikin peta antar-run tidak konsisten padahal domain geografisnya
sama. Diverifikasi numerik lat/lon pixel asli: `lat_idx=0` -> `-6.800003`
(utara), `lat_idx=4` -> `-7.000000` (selatan), `lon_idx=0` -> `107.5`
(barat), `lon_idx=6` -> `107.8` (timur) -- match hampir persis dgn
`Config.BANDUNG_LAT_MIN/MAX`/`LON_MIN/MAX` (deviasi ~3e-6, floating point).

`pixel_rows_to_grid()` pakai `lat_idx`/`lon_idx` (integer grid index,
EXACT) buat PENEMPATAN nilai di array -- ORTHOGONAL dari geometri fisik
(extent, dari poin di atas). `origin="upper"` (default `imshow`) otomatis
match orientasi "utara di atas" karena `lat_idx=0` (baris 0 array) =
latitude PALING UTARA (`-6.8`, angka lebih besar/kurang negatif) -- TIDAK
perlu flip manual.

Aspect ratio: `ax.set_aspect(1/cos(radians(lat_tengah)))` -- koreksi
equirectangular standar utk area sekecil Bandung, cukup akurat, tidak
perlu `cartopy`/proyeksi peta penuh.

### 20.4 Overlay batas kecamatan

`load_kecamatan_boundaries()` parse `scripts/geojson/KotaBandung.geojson`
pakai stdlib `json` (BUKAN geopandas -- hindari dependency berat baru).
File itu (dipakai HANYA sbg overlay visual, sesuai §19.1 poin 5 -- BUKAN
sumber data/grid) berisi 30 kecamatan GADM (`MultiPolygon`, `NAME_3`/
`TYPE_3="Kecamatan"`), rentang `lon 107.546-107.739`/`lat -6.970 s.d.
-6.837` -- overlap baik dgn extent pixel grid. Cuma ring LUAR tiap polygon
diambil (abaikan lubang -- cukup utk garis referensi visual). Gagal parse
/ file tidak ada -> `say_info` + return `[]` (overlay di-skip, BUKAN
crash -- fitur pelengkap, bukan wajib). **Divalidasi**: path sengaja
diarahkan ke file tidak ada -> graceful skip terkonfirmasi, tidak ada
exception.

### 20.5 Klasifikasi Kelas Awan & Risiko Banjir

`classify_tbb_grid(value_grid, thresholds=Config.TBB_RISK_THRESHOLDS)` --
SATU fungsi generik, breakpoint `(200.0, 270.0)` sesuai keputusan Dhika
(§9 daftar poin `s` di riwayat sesi): `<200`=code 0, `200-270`=code 1,
`>270`=code 2. **Diklasifikasi dari `y_pred`** (bukan `y_true`/Input) --
alasan: ini visualisasi FORECAST, `y_pred` SELALU tersedia sedangkan
`y_true` bisa NaN utk forecast masa depan genuine. Dipanggil SEKALI per
step, `code_grid` yang SAMA dipakai render 2 panel (Kelas Awan & Risiko
Banjir) dgn label/warna beda saja (`render_categorical_map_panel()`
generik, terima `class_labels`/`colors` sbg parameter).

### 20.6 Warna (skill `dataviz`, lihat `references/palette.md`)

- **Input/Prediksi/Aktual**: sequential blue 13-step dari `palette.md`,
  **DIBALIK** (reversed) dari orientasi default skill ("terang=dekat
  nol") -- di domain TBB, nilai RENDAH = puncak awan tinggi/dingin =
  konveksi kuat = BAHAYA, jadi harus menonjol GELAP (bukan memudar ke
  surface spt asumsi skill "0=tidak penting" yang tidak berlaku di sini).
  vmin/vmax **PER STEP** (`compute_value_range()`, dipanggil ULANG di
  dalam loop render tiap step -- lihat §20.13, REVISI dari desain awal
  yang GLOBAL lintas 18 step) -- vmax selalu = nilai TBB tertinggi yang
  benar2 muncul PADA STEP ITU (gabungan Input+Prediksi+Aktual step
  tsb), diminta Dhika biar tiap frame pakai kontras/dynamic range
  penuh. Trade-off: warna TIDAK LAGI apple-to-apple lintas step (mis.
  "biru gelap" step 1 vs step 18 bisa berarti suhu beda) -- 3 panel
  DALAM 1 frame yang sama tetap comparable satu sama lain.
- **Error Map**: ramp sintetis 2-stop (`palette.md` cuma kasih 1 hex
  orange, bukan ramp 13-step spt biru) dari surface terang `#fcfcfb` ke
  oranye kategorikal-slot-2 `#eb6834`, orientasi NORMAL (0 error=terang,
  krn 0 memang berarti "tidak ada apa-apa" -- beda semantik dari TBB).
  vmax juga PER STEP (`compute_mae_range()`, sama alasan/trade-off di atas).
- **Kelas Awan/Risiko Banjir**: status palette `palette.md`
  (good=`#0ca30c`/warning=`#fab219`/critical=`#d03b3b`, dokumentasi
  "fixed -- never themed", sudah pre-validated terpisah dari kategorikal)
  -- breakpoint SAMA di kedua panel (§20.5) jadi warnanya otomatis
  konsisten. Legend WAJIB tampil (icon+label) di tiap panel kategorikal --
  mitigasi kontras sub-3:1 `warning` di light surface (aturan
  `palette.md`, status color TIDAK BOLEH mengandalkan hue doang).
- **NaN/pixel hilang**: abu netral `#c8c7c2`, beda jelas dari semua warna
  data (`cmap.set_bad(...)`, `np.ma.masked_invalid(...)`).
- Validator `scripts/validate_palette.js` (skill dataviz) **TIDAK
  dijalankan ke status palette** -- footer validator sendiri bilang
  "scope: categorical palettes only", dan `palette.md` sudah
  mendokumentasikan status color sbg set independen yang pre-validated
  (kontras diberikan langsung di tabel dokumen).

### 20.7 Config tambahan

```python
PIXEL_GRID_SHAPE = (5, 7)
TBB_RISK_THRESHOLDS = (200.0, 270.0)
VISUALIZATION_DIR = os.path.join(PROJECT_ROOT, "visualizations")
VISUALIZATION_FRAME_DURATION_MS = 600
KOTA_BANDUNG_GEOJSON = os.path.join(PROJECT_ROOT, "scripts", "geojson", "KotaBandung.geojson")
```

**Bug kecil ketemu & di-fix SENDIRI (sebelum sempat dijalankan)**: draft
pertama `KOTA_BANDUNG_GEOJSON` pakai `SCRIPT_DIR` (direktori `config.py`
sendiri, `scripts/pipeline/`) -- salah, resolve ke
`scripts/pipeline/geojson/KotaBandung.geojson` yang tidak ada. Fix:
`PROJECT_ROOT` + komponen path eksplisit `"scripts"`. Diverifikasi
`os.path.exists(...)` True sebelum dipakai di kode manapun.

### 20.8 Struktur output & idempotensi

`visualizations/{model}_t0{YYYYMMDD_HHMM}_run{YYYYMMDD_HHMM}/frames/step{01-18}.png`
+ `.../animation.gif`. Nama folder = identitas persis file CSV sumber
(`t0_str`/`run_str` diambil LANGSUNG dari regex nama file, bukan dihitung
ulang). **SENGAJA idempotent** (run ulang `06_visualize.py` pada CSV yang
sama -> overwrite folder yang sama, BUKAN duplikat) -- beda dari Stage 05
yang tiap run SELALU unique (data baru tiap kali). Ini karena render adalah
fungsi DETERMINISTIK dari CSV sumber yang sudah ada, bukan proses yang
menghasilkan data baru. Diverifikasi: run 2x pada CSV sama -> `mtime`
`step01.png` berubah (02:44 -> 02:47), tapi jumlah folder tetap 1 (bukan
2 folder terpisah).

### 20.9 Trigger & CLI

Standalone, manual/on-demand -- `python 06_visualize.py [--forecast-dir]
[--csv] [--output-dir] [--frame-duration-ms]`. Pemilihan file forecast:
`discover_forecast_files()` (regex nama file BARU Stage 05 SAJA, file pola
lama tanpa token `t0`/`run` diabaikan diam-diam + 1 baris ringkasan
jumlah) + `prompt_select_forecast_file()` -- **0 hasil**: `ValueError`
jelas, CLI tangkap & `say_error`, berhenti bersih (bukan traceback). **1
hasil**: langsung dipakai + `say_info` transparansi (skip prompt). **>1
hasil**: menu bernomor (terurut `run_ts` DESCENDING, run terbaru di
atas), `input()` sampai angka valid. `--csv` melewati menu sepenuhnya
(parse `model`/`t0`/`run` langsung dari nama file via regex yang sama).

### 20.10 Verifikasi (data ASLI `forecast_output/`, BUKAN sintetis)

Semua PASS:
1. **Discovery & filter pola lama**: 5 CSV pola baru terdeteksi + N file
   pola lama (tanpa `t0`/`run`) diabaikan dgn 1 baris ringkasan (bukan
   spam per-file).
2. **Skenario 0 hasil**: folder kosong -> `ValueError` pesan jelas,
   terkonfirmasi via `prompt_select_forecast_file()` langsung (unit-level).
3. **Full end-to-end, y_true TERISI** (`forecast_lightgbm_t020260510_0600_run20260809_0233.csv`,
   630/630 baris ada observasi asli): 35/35 pixel lengkap (tidak ada yg
   di-skip), 18 frame + GIF ke-generate (~20 detik), `PIL.Image.open(...).n_frames
   == 18` terkonfirmasi, ukuran frame 1760x1100px. **Cek visual manual**
   (dikirim ke user): layout 2x3 benar, boundary kecamatan align rapi di
   dalam extent pixel grid, colorbar/legend terbaca, orientasi utara-atas
   benar. Bonus tak terduga: frame step 18 SECARA VISUAL menampilkan
   PERSIS fenomena "spatial collapse" yang jadi topik investigasi §17 --
   `y_true` step 18 punya cold-spot tajam di pixel barat (~240-250K) yang
   TIDAK tertangkap `y_pred` (tetap smooth ~260-270K), match tepat dgn
   `Error Map` yang menyorot lokasi sama sbg area error terbesar --
   mengkonfirmasi visualisasi ini bukan cuma "terlihat benar" tapi juga
   SECARA SUBSTANTIF berguna mendeteksi masalah model yang sudah
   didokumentasikan.
4. **Full end-to-end, y_true NaN SEMUA** (`forecast_lightgbm_t020260731_2320_run20260809_0124.csv`,
   forecast murni ke masa depan genuine): berjalan tanpa error, panel
   Aktual & Error Map render sbg placeholder-map (frame geografis +
   boundary + teks "Tidak ada observasi asli..." di tengah, BUKAN axis
   kosong polos) -- cek visual manual dikonfirmasi sesuai desain §20.1.
   CLI print info ringkas ("Panel Aktual/Error Map: NaN semua..."), bukan
   tabel MAE kosong.
5. **Missing-pixel robustness** (sintetis: 3 pixel di-drop paksa dari
   dataframe ASLI, termasuk 1 pixel pojok `4_6`): `warn_missing_pixels()`
   deteksi persis 3 pixel yg hilang, `pixel_rows_to_grid()` taruh `NaN`
   TEPAT di posisi `(lat_idx,lon_idx)` yg benar (bukan shift/salah
   posisi) -- render layer sudah teruji tidak crash krn NaN (dibuktikan
   di 2 full-run di atas yg juga lewat jalur `masked_invalid`).
6. **Idempotensi**: lihat §20.8 -- terkonfirmasi overwrite, bukan duplikat.
7. **GeoJSON overlay**: tampil benar di kedua full-run (31 ring, cek
   visual manual); path sengaja salah -> graceful skip + `say_info`,
   TIDAK crash (lihat §20.4).
8. **Regression Stage 05**: `load_actual_values()` (refactor §20.2) masih
   PASS test manual yang sama dgn sebelum refactor (630/630 baris,
   `--t0 "2026-05-10 06:00:00" --tail-files 30 --workers 1 --model
   lightgbm`) -- extract-method TIDAK mengubah perilaku.

**Catatan lingkungan (BUKAN bug kode)**: menjalankan script CLI (banner
`box-drawing` characters) langsung di beberapa terminal Windows (cp1252
default) memicu `UnicodeEncodeError` di `ui.terminal_display.hr()`/`banner()`
-- **workaround**: set `PYTHONIOENCODING=utf-8` sebelum run (mis. `set
PYTHONIOENCODING=utf-8 && python 06_visualize.py ...` di cmd, atau
`$env:PYTHONIOENCODING="utf-8"` di PowerShell). Ini pre-existing di semua
script CLI project ini (01-06 sama-sama pakai `ui.terminal_display`),
bukan spesifik Stage 06 -- dicatat di sini krn baru ketauan pas verifikasi
sesi ini.

### 20.11 Mask pixel di luar batas administratif Kota Bandung (revisi, sesi lanjutan)

**Masalah**: Dhika lihat hasil render & sadar ~20 dari 35 pixel (seluruh
cincin tepi grid: `lat_idx∈{0,4}` ATAU `lon_idx∈{0,6}`) SECARA VISUAL
berada di luar wilayah Kota Bandung -- masuk akal, karena bounding box
download (`Config.BANDUNG_LAT/LON_MIN/MAX`, dari Stage 01) sengaja lebih
lebar dari batas administratif kota (lon `107.475-107.825` grid vs
`107.546-107.739` kota asli; lat `-7.025 s.d -6.775` grid vs
`-6.970 s.d -6.837` kota asli -- kota cuma ~55% dari lebar/tinggi grid).
Diminta: pixel di luar area kota JANGAN dipakai di visualisasi -- TAPI
TIDAK BOLEH re-run `01_download_data.py` (tidak mau ubah bounding box
download/dataset asli).

**Keputusan**: mask murni di LAYER VISUALISASI Stage 06 saja (BUKAN
mengubah `data_bandung/`, dataset training, atau CSV forecast Stage 05
manapun) -- pixel yang di-mask tetap ADA di CSV/data, cuma tidak
digambar (jadi abu-abu) di GIF.

**Cara tes "pixel termasuk wilayah kota atau tidak"**: dicoba 2
pendekatan, dibandingkan hasilnya:
1. **Titik tengah pixel SAJA** (`Path.contains_point` pada 1 titik per
   pixel) -- TERLALU AGRESIF, cuma **6/35 pixel** lolos. Sebabnya:
   spacing grid ~0.05 derajat (~5,5 km) sementara Kota Bandung sempit &
   bentuknya tidak beraturan, jadi banyak pixel yang SEBAGIAN BESAR
   cell-nya beririsan dgn kota tapi titik tengahnya kebetulan jatuh di
   luar polygon (gap antar kecamatan/tepi) -- keliru exclude.
2. **Cell-overlap** (sampling grid `7x7=49` titik di dalam tiap cell
   pixel, `contains_points`, pixel lolos kalau MINIMAL 1 titik sample
   masuk polygon manapun) -- **13/35 pixel** lolos, HASIL DIPAKAI. Jauh
   lebih match dgn observasi visual Dhika: seluruh cincin tepi grid
   (`lat_idx=0`, `lat_idx=4`, `lon_idx=0`, `lon_idx=6`) KONSISTEN
   ter-exclude, sisa 13 pixel bentuknya menyerupai siluet kota (pola
   "plus/salib" kasar) yang align rapi dgn overlay batas kecamatan.

**Implementasi** (`pipeline/visualize.py`):
- `compute_boundary_mask(boundaries, grid_shape=None, n_samples=7)` --
  return grid boolean `(5,7)`, True = cell pixel beririsan dgn UNION
  seluruh polygon kecamatan (pakai `matplotlib.path.Path`, BUKAN
  dependency baru -- matplotlib sudah ada). `boundaries` kosong (gagal
  load geojson) -> return all-True (fail-OPEN, bukan fail-closed --
  fitur pelengkap, kalau geojson tidak ada JANGAN sampai semua pixel
  ke-mask tanpa sengaja).
- `apply_boundary_mask(grid, boundary_mask)` -- set NaN di posisi
  `False`, REUSE jalur render NaN yang SUDAH ADA (`masked_invalid` +
  `cmap.set_bad(NODATA_COLOR)`) -- TIDAK ada kode render baru, pixel
  ter-mask otomatis jadi abu-abu sama seperti pixel yang di-skip Stage
  05 krn gap.
- `boundary_mask_to_pixel_ids(boundary_mask)` -- konversi ke set
  `pixel_id` string, dipakai filter `detail_df`/`input_lookup` SEBELUM
  `compute_value_range()`/`compute_mae_range()` -- supaya skala warna
  GLOBAL (vmin/vmax) tidak lagi kebawa nilai pixel yang toh di-mask abu-
  abu (rentang jadi lebih fokus/relevan drpd sebelumnya).
- Diterapkan ke SEMUA 6 grid (`grid_input`, `grid_pred`, `grid_true`,
  `grid_mae`, & `code_grid` otomatis ikut ter-mask krn diturunkan dari
  `grid_pred` yang sudah NaN -- `classify_tbb_grid` propagate NaN).
- **Default NONAKTIF** (`mask_outside_bandung=False` di
  `render_all_frames()`/`visualize_forecast()`) -- awalnya sempat dibuat
  default AKTIF & dicoba (lihat "Divalidasi" di bawah, hasil 13/35 pixel
  terbukti benar & sesuai observasi Dhika), TAPI setelah lihat hasilnya
  Dhika memutuskan **"kek sebelumnya aja deh"** (kembali ke tampilan 35
  pixel penuh sbg default). Fitur TETAP ADA sbg opt-in via flag CLI
  `--mask-outside-bandung` (store_true) di `06_visualize.py` kalau nanti
  dibutuhkan lagi -- kode tidak dihapus, cuma default-nya dibalik.

**Divalidasi** (data ASLI, CSV forecast yang sudah ada):
- Full end-to-end render dgn `--mask-outside-bandung`: log konfirmasi
  "13/35 pixel dipertahankan", cek visual manual -- cincin tepi grid
  abu-abu penuh, 13 pixel tengah tetap berwarna & align dgn overlay
  batas kecamatan, ringkasan status suptitle ikut menyesuaikan (mis.
  "13 bahaya" bukan "35 bahaya" -- otomatis benar krn `code_grid` sudah
  ter-mask sebelum dihitung).
- Default (tanpa flag): regression-test, hasil identik dgn perilaku
  SEBELUM fitur mask ditambahkan (35 pixel penuh, tanpa log info mask) --
  ini yang jadi konfigurasi final.

### 20.12 Skala warna PER STEP (revisi, sesi lanjutan)

Diminta Dhika: warna paling pekat/tinggi di tiap frame harus
merepresentasikan nilai TERTINGGI PADA STEP ITU SENDIRI (TBB maupun
MAE), BUKAN nilai tertinggi lintas seluruh 18 step (desain awal §20.6).

**Perubahan**: `compute_value_range()`/`compute_mae_range()` (SAMA
persis logikanya, cuma parameter pertama diganti nama `detail_df` ->
`step_df` biar jelas maksudnya) sekarang dipanggil DI DALAM loop
`render_all_frames()` per step (pakai `step_df`, bukan `detail_df`
penuh), BUKAN sekali di luar loop sebelum render frame manapun spt
sebelumnya. Tidak ada perubahan SIGNATURE fungsi ataupun formula --
cuma titik pemanggilannya dipindah dari "sekali, global" jadi "berulang,
per step".

**Konsekuensi visual**: `grid_input` (Input @ t0, TIDAK berubah antar
step) tetap ikut masuk perhitungan `value_range` tiap step bareng
`grid_pred`/`grid_true` step itu -- jadi 3 panel TBB DALAM 1 frame yang
sama tetap saling sebanding, tapi rentang warnanya BEDA-BEDA antar
frame (step 1 dites: TBB 278-300K; step 18: TBB ~245-297K; MAE step 1:
0-4,5K; MAE step 18: 0-52K). Ini SENGAJA (diminta) -- kontras tiap
frame individual jadi maksimal, tapi warna TIDAK BISA lagi dibandingkan
lintas step cuma dari mata (harus liat suptitle/colorbar tiap frame).

**Divalidasi**: full re-render 2 CSV asli (skenario y_true terisi &
y_true NaN semua) -- keduanya jalan tanpa error, colorbar step 1 vs
step 18 dikonfirmasi visual BEDA rentang (bukti per-step benar-benar
aktif, bukan cuma ganti kode tanpa efek). `boundary_mask` (§20.11, kalau
`--mask-outside-bandung` dipakai) tetap terapply konsisten ke
perhitungan range per-step (filter pixel_id sebelum hitung min/max,
logic tidak berubah dari desain sebelumnya, cuma dipanggil lebih sering).

### 20.13 Fix arah threshold Kelas Awan/Risiko Banjir + polish tampilan (revisi, sesi lanjutan)

**Bug ketemu**: Dhika lihat panel Kelas Awan & Risiko Banjir SELALU merah
("bahaya"/"hujan") di semua step, semua CSV forecast yang ada. Dicek:
`y_pred` di SEMUA CSV forecast asli konsisten **276-300K** (kondisi
cerah/hangat, wajar buat Bandung) -- dgn breakpoint `(200,270)` versi
awal (`<200=aman`, `>270=bahaya`), nilai 276-300K SELALU masuk `>270`
jadi SELALU "bahaya". **Root cause: arah threshold TERBALIK** dari
konvensi fisik cloud-top brightness temperature -- TBB RENDAH = puncak
awan tinggi/dingin = konveksi kuat = BAHAYA, TBB TINGGI = langit
cerah/dekat suhu permukaan = AMAN. Versi awal malah nge-assign `code 0
(aman)` ke `value < lo` dan `code 2 (bahaya)` ke `value > hi` -- KEBALIK,
walau docstring `classify_tbb_grid()` sendiri SUDAH benar nyebut "TBB
rendah = ... = BAHAYA" (inkonsistensi comment vs kode, luput sebelumnya
krn belum ada CSV forecast asli yg diperiksa distribusi nilainya).

**Dikonfirmasi ke Dhika sebelum diubah** (bukan langsung diasumsikan) --
disetujui, arah dibalik. **Fix** (`classify_tbb_grid()`,
`pipeline/visualize.py`): breakpoint `Config.TBB_RISK_THRESHOLDS=(200,270)`
TIDAK BERUBAH, tapi assignment code dibalik --
`value < lo -> code 2 (bahaya/hujan)`, `lo<=value<=hi -> code 1
(waspada/mendung)`, `value > hi -> code 0 (aman/tidak hujan)`. Label &
warna per code (`render_categorical_map_panel()` call di
`build_step_figure()`) TIDAK berubah -- cuma pemetaan NILAI -> CODE yang
dibalik, jadi warna hijau/kuning/merah tetap konsisten artinya
aman/waspada/bahaya.

**Divalidasi**: re-render CSV forecast asli (y_pred 276-300K) -> Kelas
Awan & Risiko Banjir sekarang HIJAU penuh (35 aman, 0 waspada, 0
bahaya) -- sesuai ekspektasi kondisi cerah.

**Polish tampilan lain, diminta bareng** (semua di `build_step_figure()`/
`render_map_panel()`):
- **Judul panel disederhanakan**, hapus teks dalam kurung: "Input
  (Observasi @ t0)" -> "Input", "Prediksi (step N)" -> "Prediksi",
  "Aktual (Observasi)" -> "Aktual", "Kelas Awan (dari prediksi)" ->
  "Kelas Awan", "Risiko Banjir (dari prediksi)" -> "Risiko Banjir",
  "Error Map (\|prediksi - aktual\|)" -> "Error Map".
- **Step di suptitle zero-padded 2 digit**: `step 1/18` -> `step
  01/18` (`f"{step:02d}/{horizon_steps:02d}"`) -- konsisten dgn nama
  file frame (`step01.png`..`step18.png`) yg sudah begitu dari awal.
- **Semua angka desimal di visualisasi jadi 2 angka di belakang koma**:
  `damping=1.0` -> `damping=1.00` (suptitle), colorbar tick TBB & MAE
  (`FormatStrFormatter("%.2f")` di `cbar.ax.yaxis`, sebelumnya format
  default matplotlib yg jumlah desimalnya tidak konsisten antar tick).
  `MAE step ini` sudah `.2f` dari awal (§20.1), tidak berubah.

**Divalidasi**: full re-render, cek visual manual -- judul panel bersih,
`step 01/18`, `damping=1.00`, colorbar tampil `300.00`/`295.00`/dst
(2 desimal konsisten).

### 20.14 Audit ulang kode Stage 06 (diminta Dhika, setelah semua revisi di atas)

Review baris-per-baris seluruh `pipeline/visualize.py` + `06_visualize.py`
pasca semua revisi (mask boundary, skala per-step, fix threshold, polish
tampilan). **1 bug nyata ketemu, sudah diperbaiki & diverifikasi**:

**BUG: CSV forecast 0 baris (skema kolom valid, tapi kosong -- bisa
kejadian kalau Stage 05 gagal forecast SEMUA pixel, 0/35) bikin
`IndexError` mentah**, bukan pesan error yang jelas. Dibuktikan (BUKAN
diduga) pakai CSV dummy 0-baris asli: crash di
`damping_factor = float(detail_df["damping_factor"].iloc[0])`
(`visualize_forecast()`) -- `IndexError: single positional indexer is
out-of-bounds`. `IndexError` TIDAK ketangkep `except (FileNotFoundError,
ValueError)` di `06_visualize.py::main()`, jadi user lihat traceback
Python penuh, bukan `say_error` yang rapi. Kalau guard ini tidak ada,
skenario yang sama juga akan crash lebih jauh di `render_all_frames()`
(`steps[0]`, list kosong) atau `assemble_gif()` (`frames[0]`).

**Fix**: `load_forecast_csv()` sekarang validasi `if df.empty: raise
ValueError(...)` SETELAH cek kolom, SEBELUM return -- pesan jelas
("kemungkinan Stage 05 gagal forecast semua pixel"), ketangkep normal
oleh `except ValueError` yang sudah ada di CLI, TIDAK perlu ubah
`06_visualize.py` sama sekali.

**Divalidasi**: CSV dummy 0-baris (skema sama persis dgn CSV asli, dibuat
dari CSV forecast asli yg di-slice `iloc[0:0]`) -> `say_error` bersih
("CSV forecast ... kosong (0 baris) -- kemungkinan Stage 05 gagal
forecast semua pixel (0/35)..."), TIDAK ada traceback. Regression test
CSV normal (630 baris) -> tetap render 18 frame spt biasa, tidak ada
perubahan perilaku.

**Area lain yang diperiksa TIDAK ketemu bug** (diverifikasi logikanya,
bukan cuma dibaca sekilas): konsistensi arah `boundary_mask` (True=
dipertahankan) di `compute_boundary_mask()`/`apply_boundary_mask()`/
`boundary_mask_to_pixel_ids()` -- terpakai seragam di semua caller;
urutan masking SEBELUM `classify_tbb_grid()` dipanggil di
`build_step_figure()` (kalau kebalik, `code_grid`/status count bisa
masukin pixel di luar boundary); konsistensi antara `compute_mae_range()`
return `None` dan `grid_mae` all-NaN (dua-duanya pakai filter
`inside_ids` yang sama, jadi selalu align, placeholder tidak pernah
"nyasar" nampilin data kosong tanpa pesan); index alignment
`discover_forecast_files()` (`reset_index(drop=True)`) vs
`prompt_select_forecast_file()` (`iterrows()`/`iloc[]`) -- konsisten,
tidak ada off-by-one; potensi `vmin==vmax` di `imshow` (nilai seragam
1 step) -- dicek aman, matplotlib tidak crash, cuma render warna rata.

**Ronde audit KEDUA** (diminta Dhika lagi, "coba cek lagi baik-baik" --
lebih ketat, termasuk cross-check `config.py`/`inference.py`, bukan cuma
`visualize.py`): sempat curiga `Config.VISUALIZATION_FRAME_DURATION_MS`
ketemu bernilai `800` padahal comment & CLAUDE.md §20.7 bilang `600` --
ditelusuri (termasuk sempat curigai `__pycache__` basi di
`scripts/pipeline/__pycache__/` dkk sbg biang inkonsistensi runtime yang
sempat kebaca, sudah dibersihkan & dikonfirmasi BUKAN penyebab akar,
`.gitignore` pattern `**pycache**/` juga dites via `git check-ignore` --
TERBUKTI tetap match & ignore `__pycache__/` dgn benar, jadi bukan
masalah). **Klarifikasi dari Dhika**: `800` itu perubahan manual Dhika
sendiri (di luar sesi ini), `600` adalah nilai default yang memang
dimaksud -- **dikembalikan ke `600`** (comment, `config.py`, help text
`--frame-duration-ms` di `06_visualize.py`, CLAUDE.md §20.7, semuanya
disamakan balik ke `600`/`~10.8s`). Tidak ada bug kode di balik ini,
murni klarifikasi nilai default yang dimaksud.

Semua fix di §20.14 (bug empty-CSV) di-regression-test ULANG dari
`__pycache__` bersih setelah ronde audit kedua ini -- hasil konsisten
(empty-CSV -> error bersih, normal path + `--mask-outside-bandung` ->
13/35 pixel, threshold/per-step/2-desimal semua masih benar).

### 20.15 Menu pilih file forecast: navigasi panah/PageUp-PageDown (revisi, sesi lanjutan)

Diminta Dhika: ganti menu "ketik angka + Enter" jadi navigasi panah
atas/bawah atau PageUp/PageDown, Enter buat pilih (spt style menu pilihan
umumnya).

**Implementasi** (`pipeline/visualize.py`):
- `_interactive_menu_select(labels)` -- pakai `msvcrt` (stdlib BAWAAN
  Windows, TANPA dependency baru spt `questionary`/`inquirer`/`pick`,
  yang beberapa malah TIDAK support Windows native atau butuh
  `windows-curses` tambahan). Baca keypress mentah (`msvcrt.getch()`),
  redraw menu di tempat yang sama pakai ANSI cursor-up (`\033[{n}A`) +
  clear-line (`\033[K`) -- ANSI sudah kepakai di `ui/terminal_display.py`
  (`_c()`) jadi terminal target sudah pasti support. Baris terpilih
  di-highlight (prefix `> ` + warna cyan tebal `_c("1;36", ...)`).
  Mapping tombol: Up=`\xe0H`, Down=`\xe0P`, PageUp=`\xe0I` (loncat 5),
  PageDown=`\xe0Q` (loncat 5), Enter=pilih, Esc/Ctrl+C=batal
  (`KeyboardInterrupt`). Up/Down wrap-around (dari item terakhir tekan
  Down balik ke item pertama), PageUp/PageDown clamp di batas (TIDAK
  wrap -- beda perilaku disengaja, PageUp/PageDown itu "loncat", bukan
  "puter").
- **Fallback otomatis** ke `_numbered_input_select(labels)` (perilaku
  LAMA, ketik angka + Enter) kalau: bukan Windows (`os.name != "nt"`,
  `msvcrt` cuma ada di Windows) ATAU stdin/stdout bukan TTY interaktif
  (mis. dijalankan via pipe/non-interaktif/CI) -- guard INI yang bikin
  fitur baru aman dites lewat tool otomatis (Bash tool session Claude
  ini BUKAN TTY asli, jadi otomatis lewat fallback, bukan crash).
- `prompt_select_forecast_file()` disederhanakan: label tanpa prefix
  `[N]` lagi (nomor tidak relevan lagi di menu panah), tinggal
  `model=... t0=... run=... (filename)`.
- `06_visualize.py::main()` tangkap `KeyboardInterrupt` tambahan (Esc
  batal menu) -> `say_info("Dibatalkan.")` bersih, BUKAN traceback.

**Divalidasi**: fallback (`_numbered_input_select`, jalur yang bisa
dites otomatis krn Bash tool session ini non-TTY) -- pilih via angka
tetap jalan end-to-end (menu tampil, render 18 frame, GIF ke-generate,
tidak ada error). **Navigasi panah/PageUp-PageDown sendiri BELUM bisa
dites otomatis** (butuh keypress asli di TTY interaktif, di luar
kemampuan tool sesi ini) -- WAJIB dicoba langsung oleh Dhika di
terminal asli sebelum dianggap final.

### 20.16 Belum dikerjakan / di luar scope sesi ini

- Scheduling OS-level buat auto-render tiap Stage 05 selesai -- di luar
  scope, tanggung jawab operasional user (sama seperti §19.4).
- Kuantisasi warna GIF (256 warna palette-based, bawaan format GIF) bisa
  menyebabkan banding halus di heatmap kontinu -- diterima krn ini tool
  diagnostic internal, bukan output publikasi/presentasi resmi. Kalau
  nanti butuh kualitas lebih tinggi, pertimbangkan output MP4/APNG
  (dependency baru, belum dibahas).
- Workaround `PYTHONIOENCODING=utf-8` (lihat catatan di 20.10) belum
  di-otomatisasi di level kode (mis. `sys.stdout.reconfigure(encoding=...)`
  di entrypoint) -- kalau operasional Dhika sering ketemu error ini,
  pertimbangkan fix terpusat di `ui/terminal_display.py`, belum dilakukan
  sesi ini krn di luar scope Stage 06 murni.
- `compute_boundary_mask()` (§20.11) pakai APROKSIMASI sampling
  (`n_samples=7` titik per cell), BUKAN true polygon-rectangle
  intersection -- cukup akurat utk tujuan visual (grid ~0,05° vs sampling
  spacing ~0,008°), tapi kalau `PIXEL_GRID_SHAPE` diperbesar drastis
  (cell jadi jauh lebih kecil) atau butuh presisi geometri eksak,
  pertimbangkan `n_samples` lebih besar atau intersection library
  (shapely, dependency baru) -- belum dibutuhkan sejauh grid 5x7 ini.

---

## 21. Restrukturisasi output Stage 05 jadi folder-per-run (samain pola Stage 06)

**STATUS SINGKAT**: Stage 05 tadinya nyimpen 2 file FLAT langsung di
`forecast_output/` (`forecast_{model}_t0..._run....csv`/`.geojson`).
Diminta Dhika: samain ke pola Stage 06 (folder dulu, baru file di
dalamnya) -- retensi (timestamped per run, riwayat semua run disimpan,
BUKAN overwrite "latest") TETAP SAMA, cuma strukturnya jadi folder.

### Struktur baru

```
forecast_output/{model_name}_t0{YYYYMMDD_HHMM}_run{YYYYMMDD_HHMM}/
    forecast.csv
    forecast.geojson
```

Nama folder PERSIS pola Stage 06 (`{model}_t0..._run...`, TANPA prefix
`forecast_` -- prefix itu sekarang cuma nempel di nama file DI DALAM
folder). Nama file di dalam folder disederhanakan jadi generic
`forecast.csv`/`forecast.geojson` (bukan diulang lagi nama model/t0/run)
-- identitas run sudah disandang nama folder, ngulang di nama file jadi
redundan.

### Implementasi

- **`pipeline/inference.py::save_forecast_outputs()`**: bikin
  `run_dir = {output_dir}/{model_name}_t0{t0_str}_run{run_timestamp}`
  (`os.makedirs(..., exist_ok=True)`), simpan `forecast.csv`/
  `forecast.geojson` DI DALAM `run_dir`. Return jadi `(run_dir, csv_path,
  geojson_path)` -- nambah `run_dir`, konsisten sama pola
  `visualize.build_output_paths()` yang return `(run_dir, frames_dir,
  gif_path)`.
- **`pipeline/inference.py::run_inference()`**: tangkap `run_dir`, masukin
  ke return dict (`result["run_dir"]`).
- **`scripts/05_run_inference.py`**: tambah baris ringkasan `Folder
  output` (isi `result['run_dir']`), sebelum `Output CSV`/`Output
  GeoJSON` -- mirror gaya `06_visualize.py` (`Folder frame` + `GIF`
  terpisah).
- **`pipeline/visualize.py`**: `FORECAST_FILENAME_RE` (match nama FILE)
  diganti `FORECAST_FOLDER_RE` (match nama FOLDER, TANPA prefix
  `forecast_`/suffix `.csv`). `discover_forecast_files()` sekarang scan
  SUBFOLDER `forecast_dir` (`os.path.isdir(...)`), match nama folder ke
  regex, verifikasi `forecast.csv` ADA di dalamnya (folder cocok pola
  tapi `forecast.csv` tidak ada -> skip diam2, bukan crash). Kolom
  `path` di hasil tetap nunjuk ke file `forecast.csv` (bukan folder),
  jadi `load_forecast_csv()` TIDAK perlu berubah sama sekali. Kolom
  `filename` sekarang isi NAMA FOLDER (bukan nama file lagi -- nama file
  di dalam folder generic/sama semua, tidak informatif kalau dipakai di
  label menu). `visualize_forecast()`: waktu `--csv` dikasih langsung
  (skip menu), model/t0/run diparse dari NAMA FOLDER INDUK
  (`os.path.basename(os.path.dirname(csv_path))`), BUKAN dari nama file
  CSV-nya (yang sekarang generic).
- **`pipeline/config.py`**: update komentar `INFERENCE_DIR` (folder-per-
  run, bukan "nama file timestamped").

### File FLAT lama -- TIDAK ada migrasi, diperlakukan sama persis "pola lama"

File hasil Stage 05 versi SEBELUM restrukturisasi ini (`forecast_{model}_
t0..._run....csv` langsung di `forecast_output/`, bukan di dalam folder)
otomatis TIDAK ke-discover lagi (bukan subfolder) -- **TIDAK dihapus dari
disk**, cuma tidak muncul lagi di menu `06_visualize.py`. Ini KONSISTEN
dgn precedent yang sudah ada sejak fitur t0-di-nama-file ditambahkan
(§19: pola nama super-lama, sebelum t0 ada di nama file, sudah lebih dulu
diabaikan diam2 + 1 baris ringkasan jumlah) -- sekarang precedent yang
sama diperluas ke "file flat dari sebelum restrukturisasi folder-per-run
ini" juga.

### Divalidasi (data ASLI, `--tail-files 30 --workers 1`)

1. **Struktur folder**: `forecast_output/lightgbm_t020260731_2320_
   run20260809_0447/` kebentuk berisi PERSIS `forecast.csv` +
   `forecast.geojson`, TIDAK ada file flat baru di root
   `forecast_output/`.
2. **Discovery Stage 06**: `discover_forecast_files()` nemuin folder BARU
   ini dgn benar (model/t0/run kebaca dari nama folder), **16 file flat
   LAMA** (sisa dari sebelum restrukturisasi) diabaikan diam2 dgn 1 baris
   ringkasan ("16 file forecast_*.csv pola LAMA (flat, dari sebelum Stage
   05 direstruktur jadi folder-per-run) diabaikan").
3. **`--csv` override langsung**: `06_visualize.py --csv
   ".../forecast.csv"` -> model/t0/run terparse BENAR dari nama folder
   induk (`lightgbm`/`20260731_2320`/`20260809_0447`), render 18 frame
   sukses.
4. **Menu auto-pick** (tanpa `--csv`, 1 kandidat): berhasil auto-pick
   folder BARU itu, hasil render identik dgn poin 3 (idempotent, folder
   Stage 06 sama di-overwrite, bukan duplikat -- perilaku existing tidak
   berubah).
5. **Sanity data**: cek visual frame step 1 -- Input & Prediksi konsisten
   (wajar, step 1 dekat sekali dgn observasi asli), status "35 aman, 0
   waspada, 0 bahaya" (TBB 278-289K, semua > threshold 270 -- benar),
   placeholder Aktual/Error Map muncul benar (forecast ke masa depan
   genuine, belum ada observasi). Tidak ada perubahan LOGIKA rollout/
   fitur/model -- murni lokasi file output yang berubah.

---