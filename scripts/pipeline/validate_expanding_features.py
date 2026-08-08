# ./scripts/pipeline/validate_expanding_features.py
# Validasi correctness engine closed-form (expanding_features.py) vs
# implementasi naive (loop Python + np.polyfit). WAJIB dijalankan dan
# hasilnya identik (toleransi floating point kecil) sebelum closed-form
# dipakai untuk build dataset penuh. Lihat CLAUDE.md §3 & §10.

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "pipeline"))
from expanding_features import compute_expanding_window_features, compute_cumsum_stats
from config import Config


def naive_window_stats(y, starts, ends):
    """Implementasi naive: loop Python biasa + np.polyfit per window.
    Ini yang closed-form HARUS disamai hasilnya (dipakai cuma untuk validasi,
    TIDAK dipakai di produksi -- terlalu lambat untuk data skala penuh).
    """
    y = np.asarray(y, dtype=np.float64)
    results = {k: [] for k in [
        "mean_window", "std_window", "min_window", "max_window",
        "first_value", "last_value", "delta_first_last", "slope", "n_points",
    ]}

    for s, e in zip(starts, ends):
        window = y[s : e + 1]
        t = np.arange(s, e + 1, dtype=np.float64)
        n = len(window)

        mean = window.mean()
        std = window.std()  # ddof=0, population std -> samain dgn E[y^2]-E[y]^2
        mn = window.min()
        mx = window.max()
        first = window[0]
        last = window[-1]
        delta = last - first

        if n >= 2:
            slope = np.polyfit(t, window, 1)[0]
        else:
            slope = 0.0

        results["mean_window"].append(mean)
        results["std_window"].append(std)
        results["min_window"].append(mn)
        results["max_window"].append(mx)
        results["first_value"].append(first)
        results["last_value"].append(last)
        results["delta_first_last"].append(delta)
        results["slope"].append(slope)
        results["n_points"].append(n)

    return {k: np.array(v, dtype=np.float64) for k, v in results.items()}


def generate_anchors(n_series, horizon_start=None, horizon_steps=None):
    """Generate starts/ends ala CLAUDE.md §2: tiap anchor mulai dari
    beberapa titik awal berbeda, window tumbuh 1 titik per step, horizon
    18 step. Anchor2 dibuat dari berbagai posisi start supaya test
    mencakup banyak kombinasi s berbeda (bukan cuma s=0).

    horizon_start / horizon_steps : optional, default None -> ambil dari
    `Config.MIN_WINDOW_SIZE` / `Config.HORIZON_STEPS`. SEBELUMNYA hardcoded
    6/18 di sini, terpisah dari config.py -- artinya kalau MIN_WINDOW_SIZE
    atau HORIZON_STEPS diubah di config, validasi ini diam-diam tetap
    nge-test skema window yang lama (false confidence). Sekarang selalu
    ngetest skema yang lagi aktif di Config.
    """
    if horizon_start is None:
        horizon_start = Config.MIN_WINDOW_SIZE
    if horizon_steps is None:
        horizon_steps = Config.HORIZON_STEPS
    starts_all, ends_all = [], []
    # anchor bisa mulai dari t0 mana saja, selama masih cukup ruang untuk
    # horizon_start titik awal + horizon_steps step ke depan
    max_anchor_start = n_series - (horizon_start + horizon_steps)
    for anchor_start in range(0, max(max_anchor_start, 0), 7):  # step 7 biar variatif tapi nggak semua titik
        s = anchor_start
        first_e = s + horizon_start - 1  # IS1 = horizon_start titik pertama
        for step in range(horizon_steps):
            e = first_e + step
            if e >= n_series:
                break
            starts_all.append(s)
            ends_all.append(e)
    return np.array(starts_all), np.array(ends_all)


def run_validation(seed=42, n_series=200, verbose=True):
    rng = np.random.default_rng(seed)
    # Data dummy mirip TBB (suhu kecerahan awan, kisaran ~200-300K) + noise,
    # sengaja dikasih variasi termasuk nilai konstan lokal (buat test std=0
    # dan min==max case) dan trend naik-turun (buat test slope).
    y = 250 + 10 * np.sin(np.linspace(0, 6 * np.pi, n_series)) + rng.normal(0, 2, n_series)
    # sisipkan beberapa run konstan (simulasi cloud saturasi/flat region)
    y[50:55] = y[50]
    y[120:125] = y[120]

    starts, ends = generate_anchors(n_series)
    if verbose:
        print(f"Jumlah window yang divalidasi: {len(starts)}")
        print(f"Contoh 5 window pertama (s, e, n_points): "
              f"{list(zip(starts[:5].tolist(), ends[:5].tolist(), (ends[:5]-starts[:5]+1).tolist()))}")

    cumsum_stats = compute_cumsum_stats(y)
    closed_form = compute_expanding_window_features(y, starts, ends, cumsum_stats=cumsum_stats)
    naive = naive_window_stats(y, starts, ends)

    all_pass = True
    print("\n=== Hasil Validasi: closed-form vs naive ===")
    for key in closed_form:
        cf_vals = closed_form[key]
        nv_vals = naive[key]
        # toleransi floating point kecil (rtol relatif, atol absolut untuk nilai dekat 0)
        match = np.allclose(cf_vals, nv_vals, rtol=1e-8, atol=1e-6)
        max_diff = np.max(np.abs(cf_vals - nv_vals))
        status = "OK" if match else "MISMATCH"
        if not match:
            all_pass = False
        print(f"  {key:<20s} {status:<10s} max_abs_diff={max_diff:.3e}")

    print("\n=== KESIMPULAN ===")
    if all_pass:
        print("SEMUA FITUR IDENTIK (dalam toleransi floating point). "
              "Closed-form aman dipakai lanjut ke dataset_builder.py.")
    else:
        print("ADA MISMATCH -- jangan lanjut dulu, perlu debug closed-form.")

    return all_pass, closed_form, naive


if __name__ == "__main__":
    ok, _, _ = run_validation()
    sys.exit(0 if ok else 1)