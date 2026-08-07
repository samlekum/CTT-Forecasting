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
│   ├── 04_recursive_evaluate.py     # BELUM -- lihat §10
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
│   │   ├── recursive_eval.py        # BELUM
│   │   ├── inference.py             # BELUM dibahas
│   │   └── utils.py
│   ├── tools/                       # diagnostic scripts, pola serupa repo lama -- KOSONG
│   │                                 # untuk expanding window (validate_expanding_features.py
│   │                                 # justru ada di pipeline/, lihat catatan di atas)
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
4. ✅ **SELESAI** — `pipeline/model_training.py` + `scripts/03_train_models.py` —
   `stratified_monthly_split()` di-reuse VERBATIM dari repo lama (cuma
   `time_col` diganti jadi `"anchor_t0"`), training 3 model (xgboost/
   lightgbm/catboost, SVR di-drop sesuai §5), save `.joblib` +
   `training_summary.csv` ke `Config.EXPANDING_MODELS_DIR` (baru,
   tersentralisasi di config.py). Divalidasi pakai dataset sintetis 3
   bulan: cutoff per-bulan bekerja (semua bulan punya representasi di
   test set, bukan 0% kayak masalah repo lama), training end-to-end
   jalan, model tersimpan valid. Lihat §13 untuk detail keputusan desain.
5. **BELUM** — `pipeline/recursive_eval.py` — evaluasi recursive per t0, hitung MAE per
   step.

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