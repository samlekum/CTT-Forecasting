# ./scripts/pipeline/validate_no_leakage.py
# Validasi purge/embargo di stratified_monthly_split() (fix temporal leakage
# -- lihat investigasi sesi ini). WAJIB PASS sebelum purge dipercaya benar2
# menutup leak yang sudah dibuktikan ada sebelum fix ini diterapkan.
#
# BEDA dari validate_expanding_features.py: script itu validasi MATEMATIKA
# fitur closed-form (mean/std/slope), script ini validasi SPLIT (batas
# train/test), pakai fungsi ASLI (find_valid_anchors, build_pixel_samples,
# stratified_monthly_split, select_test_anchors), bukan reimplementasi.
#
# Cara pakai:
#   python .\pipeline\validate_no_leakage.py

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import Config
from pipeline.dataset_builder import find_valid_anchors, build_pixel_samples, ANCHOR_SPAN
from pipeline.model_training import stratified_monthly_split, PURGE_STEPS
from pipeline.recursive_eval import select_test_anchors


def build_synthetic_multi_pixel_dataset(n_months=3, n_pixels=2, freq_minutes=10):
    """Timeline sintetis N bulan kalender penuh, tanpa gap, n_pixels pixel
    independen (y berbeda per pixel dgn offset besar biar index asal gampang
    ditrace & 2 pixel tidak pernah "collide" nilai). Dipakai buat exercise
    SEMUA cutoff bulanan sekaligus (bukan cuma 1 bulan/1 pixel)."""
    start = pd.Timestamp("2026-01-01 00:00")
    end = start + pd.DateOffset(months=n_months)
    timeline = pd.date_range(start, end, freq=f"{freq_minutes}min", inclusive="left")

    idx_of = {ts: i for i, ts in enumerate(timeline)}
    valid_mask = np.ones(len(timeline), dtype=bool)
    anchors = find_valid_anchors(valid_mask, span=ANCHOR_SPAN, stride=1)

    frames = []
    for p in range(n_pixels):
        y = np.arange(len(timeline), dtype=np.float64) + p * 10_000_000
        df = build_pixel_samples(y, anchors, pixel_id=f"px{p}", pixel_lat=0.0, pixel_lon=0.0, timeline=timeline)
        frames.append(df)

    return pd.concat(frames, ignore_index=True), timeline, idx_of


def build_synthetic_dataset_with_gap(n_months=3, n_pixels=2, freq_minutes=10,
                                      gap_start="2026-01-31 12:00", gap_ticks=288):
    """Sama seperti build_synthetic_multi_pixel_dataset(), TAPI dengan gap
    (data hilang, disimulasikan sebagai NaN di `y` -- pola sama seperti
    cloud-mask/file .nc hilang asli) sepanjang `gap_ticks` tick, ditaruh
    PERSIS DI SEKITAR BATAS BULAN (default: 48 jam terakhir Januari + awal
    Februari, gap_ticks=288 -> 2880 menit -> 48 jam).

    Ini stress-test KHUSUS untuk purge dua-sisi lintas bulan (audit sesi
    ini): dgn gap ini, kepadatan anchor di sekitar cutoff/boundary bulan
    jadi TIDAK beraturan (banyak posisi start yang gagal gap-skip rule di
    dekat gap), beda dari skenario tanpa-gap yang anchornya rapat seragam
    tiap 10 menit. Timeline TETAP uniform (tidak dipotong) -- yang hilang
    cuma nilai `y`-nya, sama seperti gap di data produksi asli.

    Return
    ------
    (dataset, timeline, idx_of, (gap_start_idx, gap_end_idx)) -- gap_end_idx
    eksklusif (indeks pertama SETELAH gap yang kembali valid).
    """
    start = pd.Timestamp("2026-01-01 00:00")
    end = start + pd.DateOffset(months=n_months)
    timeline = pd.date_range(start, end, freq=f"{freq_minutes}min", inclusive="left")
    idx_of = {ts: i for i, ts in enumerate(timeline)}

    gap_start_idx = idx_of[pd.Timestamp(gap_start)]
    gap_end_idx = gap_start_idx + gap_ticks  # eksklusif

    frames = []
    for p in range(n_pixels):
        y = np.arange(len(timeline), dtype=np.float64) + p * 10_000_000
        y_gapped = y.copy()
        y_gapped[gap_start_idx:gap_end_idx] = np.nan
        valid_mask = ~np.isnan(y_gapped)
        anchors = find_valid_anchors(valid_mask, span=ANCHOR_SPAN, stride=1)
        df = build_pixel_samples(
            y_gapped, anchors, pixel_id=f"px{p}", pixel_lat=0.0, pixel_lon=0.0, timeline=timeline
        )
        frames.append(df)

    dataset = pd.concat(frames, ignore_index=True)
    return dataset, timeline, idx_of, (gap_start_idx, gap_end_idx)


def check_no_window_touches_test_region(train_df, idx_of, cutoffs):
    """Buktikan TIDAK ADA feature window training yang menyentuh timestamp
    >= cutoff, dicek per bulan (cutoff beda tiap bulan)."""
    bulan_of = train_df["anchor_t0"].dt.to_period("M").astype(str)
    ok = True
    for bulan, cutoff_time in cutoffs.items():
        C = idx_of[cutoff_time]
        sub = train_df[bulan_of == bulan]
        if sub.empty:
            continue
        start_idx = sub["anchor_t0"].map(idx_of).astype(int).values
        end_idx = start_idx + sub["n_points"].astype(int).values - 1
        n_bad = int((end_idx >= C).sum())
        if n_bad > 0:
            print(f"  [MISMATCH] bulan {bulan}: {n_bad} baris train window-nya nyentuh index >= cutoff (C={C})")
            ok = False
    return ok


def check_no_target_touches_test_region(train_df, idx_of, cutoffs):
    """Buktikan TIDAK ADA target training yang >= cutoff, dicek per bulan."""
    bulan_of = train_df["anchor_t0"].dt.to_period("M").astype(str)
    ok = True
    for bulan, cutoff_time in cutoffs.items():
        C = idx_of[cutoff_time]
        sub = train_df[bulan_of == bulan]
        if sub.empty:
            continue
        target_idx = sub["target_time"].map(idx_of).astype(int).values
        n_bad = int((target_idx >= C).sum())
        if n_bad > 0:
            print(f"  [MISMATCH] bulan {bulan}: {n_bad} baris train target >= cutoff (C={C})")
            ok = False
    return ok


def check_no_shared_raw_value(train_df, test_df, timeline, idx_of, pixel_col="pixel_id"):
    """Buktikan TIDAK ADA raw observation yang dipakai SEKALIGUS sebagai
    feature window training DAN target test, PER PIXEL (leakage cuma valid
    dibandingkan antar row dalam pixel yang sama -- raw value pixel lain
    tidak relevan). Vectorized pakai prefix-sum (pola sama seperti
    find_valid_anchors), bukan loop Python per pasangan (s,e)."""
    ok = True
    T = len(timeline)
    for pixel_id, test_sub in test_df.groupby(pixel_col):
        train_sub = train_df[train_df[pixel_col] == pixel_id]
        if train_sub.empty or test_sub.empty:
            continue

        target_flag = np.zeros(T, dtype=np.int64)
        target_idx_all = test_sub["target_time"].map(idx_of).astype(int).values
        target_flag[target_idx_all] = 1
        prefix = np.concatenate(([0], np.cumsum(target_flag)))

        start_idx = train_sub["anchor_t0"].map(idx_of).astype(int).values
        end_idx = start_idx + train_sub["n_points"].astype(int).values - 1
        count_in_window = prefix[end_idx + 1] - prefix[start_idx]
        n_bad = int((count_in_window > 0).sum())
        if n_bad > 0:
            print(f"  [MISMATCH] pixel={pixel_id}: {n_bad} baris train window mengandung "
                  f"raw value yang JUGA jadi target test (leakage)")
            ok = False
    return ok


def check_test_set_unaffected_by_purge(dataset):
    """Buktikan test set TIDAK berubah oleh purge -- bandingkan test_df hasil
    purge_steps=0 (perilaku lama) vs purge_steps=default (Config.PURGE_STEPS)."""
    _, test_old, cutoffs_old = stratified_monthly_split(dataset, purge_steps=0)
    _, test_new, cutoffs_new = stratified_monthly_split(dataset, purge_steps=None)

    same_cutoffs = cutoffs_old == cutoffs_new
    test_old_sorted = test_old.sort_values(["pixel_id", "anchor_t0", "step"]).reset_index(drop=True)
    test_new_sorted = test_new.sort_values(["pixel_id", "anchor_t0", "step"]).reset_index(drop=True)
    same_test = test_old_sorted.equals(test_new_sorted)

    if not same_cutoffs:
        print("  [MISMATCH] cutoff_time berubah gara-gara purge (SEHARUSNYA tidak berubah)")
    if not same_test:
        print("  [MISMATCH] test_df berubah gara-gara purge (SEHARUSNYA identik)")
    return same_cutoffs and same_test


def check_select_test_anchors_consistency(dataset):
    """Buktikan select_test_anchors() (recursive_eval.py, Stage 4) menghasilkan
    set (pixel_id, anchor_t0) test yang IDENTIK dengan stratified_monthly_split()
    (Stage 3) langsung -- dua fungsi tsb HARUS sinkron karena select_test_anchors
    cuma reload dataset dari CSV lalu manggil stratified_monthly_split yang sama."""
    _, test_direct, _ = stratified_monthly_split(dataset)
    direct_anchors = test_direct[["pixel_id", "anchor_t0"]].drop_duplicates()
    direct_anchors = direct_anchors.sort_values(["pixel_id", "anchor_t0"]).reset_index(drop=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = os.path.join(tmpdir, "synthetic_dataset.csv")
        dataset.to_csv(csv_path, index=False)
        via_select = select_test_anchors(dataset_csv_path=csv_path)
        via_select = via_select.sort_values(["pixel_id", "anchor_t0"]).reset_index(drop=True)
        via_select["anchor_t0"] = pd.to_datetime(via_select["anchor_t0"])

    match = direct_anchors.equals(via_select)
    if not match:
        print(f"  [MISMATCH] select_test_anchors() ({len(via_select)} anchor) != "
              f"stratified_monthly_split() langsung ({len(direct_anchors)} anchor)")
    return match


def run_checks(dataset, timeline, idx_of, label=""):
    """Jalankan kelima check terhadap satu (dataset, timeline, idx_of) yang
    sudah dibangun -- dipakai bareng oleh skenario tanpa-gap maupun
    skenario dengan-gap, supaya logic check-nya SATU tempat (tidak
    diduplikasi per skenario)."""
    train_df, test_df, cutoffs = stratified_monthly_split(dataset)  # purge default (Config.PURGE_STEPS)
    print(f"Split (dgn purge default) [{label}]: train={len(train_df)}, test={len(test_df)}, "
          f"{len(cutoffs)} cutoff bulanan: {list(cutoffs.items())}")

    results = {}

    print("\n[1] Window training TIDAK menyentuh timestamp >= cutoff (semua bulan)...")
    results["no_window_touches_test"] = check_no_window_touches_test_region(train_df, idx_of, cutoffs)

    print("[2] Target training TIDAK >= cutoff (semua bulan)...")
    results["no_target_touches_test"] = check_no_target_touches_test_region(train_df, idx_of, cutoffs)

    print("[3] Tidak ada raw observation dipakai ganda: feature training & target test...")
    results["no_shared_raw_value"] = check_no_shared_raw_value(train_df, test_df, timeline, idx_of)

    print("[4] Test set TIDAK berubah oleh purge (purge_steps=0 vs default)...")
    results["test_unaffected_by_purge"] = check_test_set_unaffected_by_purge(dataset)

    print("[5] select_test_anchors() (Stage 4) konsisten dgn stratified_monthly_split() (Stage 3)...")
    results["select_test_anchors_consistent"] = check_select_test_anchors_consistency(dataset)

    all_pass = all(results.values())
    print(f"\n=== KESIMPULAN [{label}] ===")
    for name, ok in results.items():
        print(f"  {name:<32s} {'OK' if ok else 'MISMATCH'}")
    if all_pass:
        print(f"[{label}] SEMUA CHECK PASS -- purge/embargo menutup temporal leakage yang ditemukan "
              "sebelumnya, test set tidak terpengaruh, Stage 3/Stage 4 tetap konsisten.")
    else:
        print(f"[{label}] ADA MISMATCH -- purge belum menutup leakage dengan benar.")

    return all_pass, results


def run_validation(n_months=3, n_pixels=2, verbose=True):
    """Skenario TANPA gap -- semua anchor rapat seragam tiap FREQ_MINUTES."""
    print(f"Config: MIN_WINDOW_SIZE={Config.MIN_WINDOW_SIZE}, HORIZON_STEPS={Config.HORIZON_STEPS}, "
          f"ANCHOR_SPAN={Config.ANCHOR_SPAN}, PURGE_STEPS={PURGE_STEPS} "
          f"({PURGE_STEPS * Config.FREQ_MINUTES} menit)")

    dataset, timeline, idx_of = build_synthetic_multi_pixel_dataset(n_months=n_months, n_pixels=n_pixels)
    label = f"{n_months}bulan_{n_pixels}pixel_tanpa_gap"
    print(f"Dataset sintetis: {len(dataset)} baris, {n_pixels} pixel, {n_months} bulan kalender, tanpa gap.")

    return run_checks(dataset, timeline, idx_of, label=label)


def run_validation_with_gap(n_months=3, n_pixels=2, gap_start="2026-01-31 12:00", gap_ticks=288):
    """Skenario DENGAN gap 48 jam persis di batas bulan Januari/Februari --
    stress-test purge dua-sisi lintas bulan saat kepadatan anchor di
    sekitar boundary TIDAK beraturan (bukan rapat seragam)."""
    print(f"\nConfig: MIN_WINDOW_SIZE={Config.MIN_WINDOW_SIZE}, HORIZON_STEPS={Config.HORIZON_STEPS}, "
          f"ANCHOR_SPAN={Config.ANCHOR_SPAN}, PURGE_STEPS={PURGE_STEPS} "
          f"({PURGE_STEPS * Config.FREQ_MINUTES} menit)")

    dataset, timeline, idx_of, (gs, ge) = build_synthetic_dataset_with_gap(
        n_months=n_months, n_pixels=n_pixels, gap_start=gap_start, gap_ticks=gap_ticks
    )
    label = f"{n_months}bulan_{n_pixels}pixel_GAP48jam_di_batas_bulan"
    print(f"Dataset sintetis DENGAN GAP: {len(dataset)} baris, gap = index [{gs},{ge}) -> "
          f"waktu [{timeline[gs]} .. {timeline[ge - 1]}] ({gap_ticks} tick = {gap_ticks * Config.FREQ_MINUTES} menit hilang).")

    return run_checks(dataset, timeline, idx_of, label=label)


if __name__ == "__main__":
    scenarios = [
        lambda: run_validation(n_months=1, n_pixels=1),
        lambda: run_validation(n_months=3, n_pixels=2),
        lambda: run_validation(n_months=4, n_pixels=3),
        lambda: run_validation_with_gap(n_months=3, n_pixels=2),
    ]
    all_ok = True
    for scenario in scenarios:
        ok, _ = scenario()
        all_ok = all_ok and ok
        print()

    print("=" * 70)
    print("SEMUA SKENARIO PASS" if all_ok else "ADA SKENARIO YANG MISMATCH -- lihat detail di atas")
    sys.exit(0 if all_ok else 1)
