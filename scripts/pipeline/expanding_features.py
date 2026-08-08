# ./scripts/pipeline/expanding_features.py
# Engine closed-form vectorized untuk menghitung fitur statistik dari window
# expanding (IS1..IS18) per pixel. TIDAK ada loop Python per window dan TIDAK
# ada np.polyfit di dalam loop -- semua fitur dihitung dari cumulative sum
# yang di-precompute sekali per time series (per pixel). Lihat CLAUDE.md §3
# untuk latar belakang & alasan desain.
#
# Konvensi: window [s, e] inklusif di kedua ujung, indeks 0-based relatif ke
# array time series `y` satu pixel (bukan relatif ke window). n_points = e - s + 1.

import numpy as np
import pandas as pd

# 9 kolom fitur final (urutan ini dipakai konsisten di seluruh pipeline)
FEATURE_COLUMNS = [
    "mean_window",
    "std_window",
    "min_window",
    "max_window",
    "first_value",
    "last_value",
    "delta_first_last",
    "slope",
    "n_points",
]


def compute_cumsum_stats(y):
    """Precompute cumulative sum sekali per time series (per pixel).

    Di-pad dengan 0 di depan supaya window [s, e] inklusif bisa dihitung
    langsung: sum(y[s..e]) = cumsum[e+1] - cumsum[s], tanpa perlu cek s==0.

    Return
    ------
    dict berisi cumsum_y, cumsum_y2, cumsum_ty (masing-masing panjang N+1).
    t (waktu) dipakai sebagai indeks global 0..N-1, BUKAN relatif ke window.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    t = np.arange(n, dtype=np.float64)

    cumsum_y = np.concatenate(([0.0], np.cumsum(y)))
    cumsum_y2 = np.concatenate(([0.0], np.cumsum(y * y)))
    cumsum_ty = np.concatenate(([0.0], np.cumsum(t * y)))

    return {"cumsum_y": cumsum_y, "cumsum_y2": cumsum_y2, "cumsum_ty": cumsum_ty}


def _sum_sq_upto(k):
    """Closed-form sum_{i=0}^{k} i^2 = k(k+1)(2k+1)/6, vectorized.

    Untuk k = -1 (kasus s=0, butuh sum_t2 sampai s-1) hasilnya harus 0 --
    di-handle eksplisit karena formula tetap konsisten (k(k+1)(2k+1)/6 di
    k=-1 kebetulan juga menghasilkan 0, tapi kita jaga eksplisit biar jelas
    kalau ada perubahan dtype/precision).
    """
    k = np.asarray(k, dtype=np.float64)
    result = k * (k + 1) * (2 * k + 1) / 6.0
    return np.where(k < 0, 0.0, result)


def _rolling_min_max_per_anchor(y, starts, ends):
    """Hitung min_window & max_window untuk banyak window sekaligus.

    Karena ini expanding window, semua window dalam satu anchor berbagi
    `start` yang sama (cuma `end` yang tumbuh). Jadi window-window dengan
    `start` sama di-group, lalu SATU kali running accumulate
    (np.minimum.accumulate / np.maximum.accumulate) dijalankan menutupi
    seluruh rentang yang dibutuhkan anchor itu -- hasilnya dipakai ulang
    (lookup index) untuk semua step di anchor tsb, bukan re-scan min/max
    dari awal tiap step seperti pendekatan naive.

    Catatan: ini BUKAN O(1) murni per window (beda dari mean/std/slope yang
    murni cumsum-based), tapi O(panjang window terpanjang) per anchor,
    dieksekusi vectorized. Untuk O(1) sejati per window arbitrary dibutuhkan
    struktur RMQ/sparse-table terpisah -- di luar scope closed-form yang
    diminta CLAUDE.md §3, dan tidak diperlukan karena struktur window di
    project ini (expanding, start tetap per anchor) sudah bikin pendekatan
    accumulate-per-anchor efisien.
    """
    y = np.asarray(y)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)
    n = len(starts)

    min_out = np.empty(n, dtype=np.float64)
    max_out = np.empty(n, dtype=np.float64)

    # Urutkan berdasarkan start supaya window dari anchor yang sama
    # bersebelahan -> bisa di-group tanpa hashmap.
    order = np.argsort(starts, kind="stable")
    starts_sorted = starts[order]
    ends_sorted = ends[order]

    i = 0
    while i < n:
        s = starts_sorted[i]
        j = i
        while j < n and starts_sorted[j] == s:
            j += 1

        e_max = ends_sorted[i:j].max()
        segment = y[s : e_max + 1]
        run_min = np.minimum.accumulate(segment)
        run_max = np.maximum.accumulate(segment)

        rel_e = ends_sorted[i:j] - s
        group_idx = order[i:j]
        min_out[group_idx] = run_min[rel_e]
        max_out[group_idx] = run_max[rel_e]

        i = j

    return min_out, max_out


def compute_expanding_window_features(y, starts, ends, cumsum_stats=None):
    """Hitung 9 fitur closed-form untuk banyak window sekaligus, vectorized.

    Parameters
    ----------
    y : array-like, shape (N,)
        Time series satu pixel (nilai TBB), berurutan kronologis, TANPA gap
        (gap-skip rule diterapkan di level anchor selection -- lihat
        dataset_builder.py / CLAUDE.md §4, bukan di sini).
    starts, ends : array-like of int, shape (M,)
        Indeks awal & akhir (inklusif) tiap window, relatif ke `y`. Untuk
        satu anchor, beberapa baris starts/ends berbagi `start` yang sama
        dengan `end` yang tumbuh (IS1, IS2, ..., IS18).
    cumsum_stats : dict, optional
        Hasil compute_cumsum_stats(y). Kalau None, dihitung ulang di sini
        (berguna untuk testing/validasi cepat). Untuk pemakaian produksi di
        dataset_builder.py, precompute sekali per pixel lalu pass ke sini
        supaya tidak recompute cumsum untuk tiap batch anchor.

    Return
    ------
    dict of np.ndarray, key sesuai FEATURE_COLUMNS, masing-masing shape (M,).
    """
    y = np.asarray(y, dtype=np.float64)
    starts = np.asarray(starts, dtype=np.int64)
    ends = np.asarray(ends, dtype=np.int64)

    if cumsum_stats is None:
        cumsum_stats = compute_cumsum_stats(y)

    cumsum_y = cumsum_stats["cumsum_y"]
    cumsum_y2 = cumsum_stats["cumsum_y2"]
    cumsum_ty = cumsum_stats["cumsum_ty"]

    n_points = (ends - starts + 1).astype(np.float64)

    # --- sum & mean: O(1) per window pakai cumsum_y ---
    sum_y = cumsum_y[ends + 1] - cumsum_y[starts]
    mean_window = sum_y / n_points

    # --- std: O(1) pakai cumsum_y2, formula var = E[y^2] - E[y]^2 ---
    sum_y2 = cumsum_y2[ends + 1] - cumsum_y2[starts]
    var_window = sum_y2 / n_points - mean_window**2
    # Guard floating point: var seharusnya >= 0 secara matematis, tapi
    # kadang jadi sedikit negatif (mis. -1e-12) karena floating point error
    # pada window dengan variansi hampir 0. Clip ke 0 sebelum sqrt.
    var_window = np.clip(var_window, 0.0, None)
    std_window = np.sqrt(var_window)

    # --- slope: O(1) pakai cumsum_ty + closed-form sum_t & sum_t2 ---
    # regresi linear y ~ t, t = indeks global (s..e), formula OLS 1D:
    # slope = (n*sum_ty - sum_t*sum_y) / (n*sum_t2 - sum_t^2)
    sum_ty = cumsum_ty[ends + 1] - cumsum_ty[starts]
    starts_f = starts.astype(np.float64)
    ends_f = ends.astype(np.float64)
    sum_t = n_points * (starts_f + ends_f) / 2.0
    sum_t2 = _sum_sq_upto(ends_f) - _sum_sq_upto(starts_f - 1)

    denom = n_points * sum_t2 - sum_t**2
    numer = n_points * sum_ty - sum_t * sum_y
    # denom == 0 kalau n_points == 1 (satu titik, slope tidak terdefinisi) ->
    # set slope 0 untuk kasus ini, jangan divide by zero.
    slope = np.where(denom != 0, numer / np.where(denom == 0, 1.0, denom), 0.0)

    # --- first/last/delta: trivial O(1), langsung index ke y ---
    first_value = y[starts]
    last_value = y[ends]
    delta_first_last = last_value - first_value

    # --- min/max: accumulate per anchor (lihat docstring _rolling_min_max_per_anchor) ---
    min_window, max_window = _rolling_min_max_per_anchor(y, starts, ends)

    return {
        "mean_window": mean_window,
        "std_window": std_window,
        "min_window": min_window,
        "max_window": max_window,
        "first_value": first_value,
        "last_value": last_value,
        "delta_first_last": delta_first_last,
        "slope": slope,
        "n_points": n_points,
    }


def compute_window_features_matrix(series_matrix):
    """Hitung 9 fitur closed-form dari MATRIX window yang SAMA panjangnya
    (shape (n_window, L)) -- beda dari compute_expanding_window_features()
    di atas yang butuh SATU time series `y` panjang + banyak (start, end)
    pasangan arbitrary via cumsum. Versi ini dipakai kalau window-nya sudah
    di-gather jadi matrix eksplisit (baris = window individual, sudah tidak
    terikat ke satu `y` per pixel), yaitu di dua tempat:

    1. `recursive_eval.run_recursive_evaluation()` -- window tumbuh dari
       hasil prediksi model sendiri (rollout), bukan dari cache/`y` biasa.
    2. `model_training.inject_recursive_style_noise()` -- window di-gather
       dari cache TAPI mau dinoise per-baris SEBELUM hitung fitur, jadi
       butuh matrix eksplisit juga (bukan pakai cumsum dari `y` asli).

    Dipisah dari compute_expanding_window_features() (bukan reuse cumsum
    trick) karena di sini window per baris sudah independen satu sama lain
    (beda pixel/anchor), full-recompute per window matrix tetap murah untuk
    window pendek (<=23 titik, lihat CLAUDE.md §15) dan lebih simpel
    daripada makna window per-anchor lewat cumsum global per pixel.

    Parameters
    ----------
    series_matrix : np.ndarray, shape (n_window, L)
        L = panjang window (SAMA untuk semua baris di panggilan ini).

    Return
    ------
    pd.DataFrame, kolom = FEATURE_COLUMNS, baris sejajar dengan series_matrix.
    """
    n_window, L = series_matrix.shape

    mean_window = series_matrix.mean(axis=1)
    var_window = np.clip(series_matrix.var(axis=1), 0.0, None)  # var(ddof=0), guard floating point sama seperti compute_expanding_window_features()
    std_window = np.sqrt(var_window)
    min_window = series_matrix.min(axis=1)
    max_window = series_matrix.max(axis=1)
    first_value = series_matrix[:, 0]
    last_value = series_matrix[:, -1]
    delta_first_last = last_value - first_value

    # Slope pakai t lokal 0..L-1 -- identik dengan t global karena slope
    # invarian terhadap pergeseran konstan.
    t = np.arange(L, dtype=np.float64)
    t_mean = t.mean()
    t_centered = t - t_mean
    denom = np.sum(t_centered**2)
    if denom == 0:  # L == 1, harusnya nggak pernah kejadian (MIN_WINDOW_SIZE >= 3 dijamin config), guard saja
        slope = np.zeros(n_window)
    else:
        numer = ((series_matrix - mean_window[:, None]) * t_centered).sum(axis=1)
        slope = numer / denom

    n_points = np.full(n_window, float(L))

    return pd.DataFrame({
        "mean_window": mean_window,
        "std_window": std_window,
        "min_window": min_window,
        "max_window": max_window,
        "first_value": first_value,
        "last_value": last_value,
        "delta_first_last": delta_first_last,
        "slope": slope,
        "n_points": n_points,
    })[FEATURE_COLUMNS]