# CLAUDE.md — CTT-Forecasting-Expanding

Dokumen ini adalah panduan konteks untuk sesi Claude berikutnya yang mengerjakan
repo **`CTT-Forecasting-Expanding`**. Baca ini SEBELUM mulai ngoding apapun.

---

## 1. Konteks & Kenapa Repo Ini Ada

Ini adalah **project baru, terpisah total** dari repo lama `CTT-Forecasting`
(https://github.com/samlekum/CTT-Forecasting). Repo lama TIDAK diubah/disentuh
oleh project ini — anggap read-only reference.

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
├── README.md
├── CLAUDE.md                        # dokumen ini
├── scripts/
│   ├── 01_download_data.py          # REUSE apa adanya dari repo lama, TIDAK diubah
│   ├── 02_build_expanding_features.py
│   ├── 03_train_models.py           # xgboost, lightgbm, catboost
│   ├── 04_recursive_evaluate.py
│   ├── 05_run_inference.py          # BELUM dibahas -- sesi berikutnya
│   ├── 06_visualize.py              # BELUM dibahas -- sesi berikutnya
│   ├── pipeline/
│   │   ├── config.py
│   │   ├── ftp_client.py            # reuse dari repo lama
│   │   ├── netcdf_tools.py          # reuse dari repo lama
│   │   ├── file_tracker.py          # reuse dari repo lama
│   │   ├── telegram_notifier.py     # reuse dari repo lama
│   │   ├── expanding_features.py    # BARU -- closed-form vectorized engine (lihat §3)
│   │   ├── dataset_builder.py       # BARU -- generator sample IS/OS per pixel, gap-aware
│   │   ├── model_training.py        # BARU -- stratified_monthly_split + 3 model
│   │   ├── recursive_eval.py        # BARU
│   │   ├── inference.py             # BELUM dibahas
│   │   └── utils.py
│   ├── tools/                       # diagnostic scripts, pola serupa repo lama
│   └── ui/
│       └── terminal_display.py      # reuse dari repo lama
```

---

## 10. Scope Sesi Berikutnya

Fokus sesi berikutnya: **02 → 03 → 04 saja** (build features → train →
recursive evaluate). `01` tinggal disalin dari repo lama. `05` (inference)
dan `06` (visualisasi) **belum dibahas sama sekali** — jangan diasumsikan
desainnya, harus didiskusikan dulu di sesi terpisah sebelum dikoding.

### Urutan kerja yang disarankan untuk sesi berikutnya:
1. Implementasi `pipeline/expanding_features.py` (closed-form vectorized).
2. **Validasi wajib**: bandingkan hasil closed-form vs naive loop di subset
   kecil data, pastikan identik sebelum lanjut.
3. `pipeline/dataset_builder.py` — generate training samples per pixel,
   terapkan gap-skip rule (§4).
4. `pipeline/model_training.py` — `stratified_monthly_split` + training 3
   model.
5. `pipeline/recursive_eval.py` — evaluasi recursive per t0, hitung MAE per
   step.

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