# ./scripts/tools/check_t0_month_distribution.py
#
# CARA PAKAI:
#   cd scripts
#   python tools/check_t0_month_distribution.py
#   python tools/check_t0_month_distribution.py --interval 30 --split-strategy stratified_monthly

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset, SPLIT_STRATEGIES
from pipeline.ground_truth import build_ground_truth_lookup
from pipeline.recursive_eval import select_valid_t0, select_valid_t0_stratified

HORIZON_MINUTES = 180
TEST_FRAC = 0.15
MAX_START_POINTS = 40


def parse_args():
    p = argparse.ArgumentParser(description="Cek distribusi bulan dari t0 yang terpilih untuk evaluasi recursive.")
    p.add_argument("--interval", type=int, default=10, choices=[10, 30, 60])
    p.add_argument("--split-strategy", choices=list(SPLIT_STRATEGIES.keys()), default="stratified_monthly")
    return p.parse_args()


def month_distribution(t0_list):
    if not t0_list:
        return pd.Series(dtype=int)
    return pd.Series([pd.Timestamp(t).to_period("M") for t in t0_list]).value_counts().sort_index()


def main():
    args = parse_args()
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")

    n_steps = HORIZON_MINUTES // args.interval
    src_path = os.path.join(dataset_dir, f"features_{args.interval}min_ar.csv")
    if not os.path.exists(src_path):
        print(f"[ERROR] File tidak ditemukan: {src_path}")
        return

    print("=" * 90)
    print(f"CEK DISTRIBUSI BULAN t0 -- interval {args.interval} menit, split_strategy={args.split_strategy}")
    print("=" * 90)

    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    lookup = build_ground_truth_lookup(df10)

    df = load_ar_dataset(src_path)
    split_fn = SPLIT_STRATEGIES[args.split_strategy]
    train_df, test_df, cutoff_info = split_fn(df, test_frac=TEST_FRAC)

    test_months = sorted(test_df["base_time"].dt.to_period("M").unique())
    print(f"\nBulan yang ada di test_df ({args.split_strategy}): {[str(m) for m in test_months]}")

    t0_seq = select_valid_t0(test_df, args.interval, lookup, n_steps, MAX_START_POINTS)
    t0_strat = select_valid_t0_stratified(test_df, args.interval, lookup, n_steps, MAX_START_POINTS)

    print(f"\n--- select_valid_t0() [SEQUENTIAL, dipakai default 05_recursive_evaluation.py] ---")
    print(f"Total t0 terpilih: {len(t0_seq)}")
    dist_seq = month_distribution(t0_seq)
    print("Distribusi per bulan:")
    print(dist_seq.to_string() if not dist_seq.empty else "  (kosong)")
    if len(dist_seq) == 1:
        print(f"[MASALAH] Semua {len(t0_seq)} titik t0 cuma dari SATU bulan ({dist_seq.index[0]}) "
              f"-- evaluasi recursive_evaluation.csv TIDAK merepresentasikan bulan lain di test set.")

    print(f"\n--- select_valid_t0_stratified() [BARU, --t0-selection stratified_monthly] ---")
    print(f"Total t0 terpilih: {len(t0_strat)}")
    dist_strat = month_distribution(t0_strat)
    print("Distribusi per bulan:")
    print(dist_strat.to_string() if not dist_strat.empty else "  (kosong)")

    print("\n" + "-" * 90)
    if len(dist_seq) <= 1 and len(dist_strat) > 1:
        print("[KESIMPULAN] Dugaan TERKONFIRMASI: sequential bias ke satu bulan, "
              "stratified_monthly berhasil menyebar t0 ke banyak bulan. "
              "Rerun 05_recursive_evaluation.py dengan --t0-selection stratified_monthly "
              "supaya hasil MAE benar-benar merepresentasikan seluruh test set.")
    elif len(dist_seq) > 1:
        print("[KESIMPULAN] sequential ternyata sudah menyentuh >1 bulan -- cek tabel di atas "
              "untuk lihat apakah distribusinya tetap timpang (mis. 35 di satu bulan, 5 di bulan lain).")
    else:
        print("[INFO] Tidak ada t0 valid ditemukan sama sekali -- cek n_steps/horizon atau gap data.")


if __name__ == "__main__":
    main()