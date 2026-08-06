# ./scripts/tools/check_seasonal_distribution.py
# Diagnosis: apakah chronological_split() menyisakan bulan-bulan tertentu
# (terutama musim hujan/konvektif seperti Januari) dengan representasi
# minim/nol di test set -- dugaan akar masalah kenapa `reliable` flag
# gagal mendeteksi error besar di kondisi konvektif ekstrem (lihat
# briefing sesi Kyoto Research, temuan #4).
#
# TIDAK perlu rerun 03a/03b -- cukup baca features_{interval}min_ar.csv
# yang sudah ada.
#
# CARA PAKAI:
#   cd scripts
#   python tools/check_seasonal_distribution.py
#   python tools/check_seasonal_distribution.py --interval 30
#   python tools/check_seasonal_distribution.py --test-frac 0.20

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset, chronological_split


def parse_args():
    p = argparse.ArgumentParser(
        description="Cek representasi bulan di train/test split (chronological_split)."
    )
    p.add_argument(
        "--interval", type=int, default=10, choices=[10, 30, 60],
        help="Interval menit yang mau dicek (default: 10, dipakai produksi 06/07/08).",
    )
    p.add_argument(
        "--test-frac", type=float, default=0.15,
        help="test_frac yang sama seperti dipakai 04_train_models.py & 05_recursive_evaluation.py (default 0.15).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    src_path = os.path.join(dataset_dir, f"features_{args.interval}min_ar.csv")

    if not os.path.exists(src_path):
        print(f"[ERROR] File tidak ditemukan: {src_path}")
        return

    print("=" * 90)
    print(f"DIAGNOSIS DISTRIBUSI MUSIMAN -- interval {args.interval} menit, test_frac={args.test_frac}")
    print("=" * 90)

    df = load_ar_dataset(src_path)
    train_df, test_df, cutoff_time = chronological_split(df, test_frac=args.test_frac)

    print(f"\nCutoff waktu split : {cutoff_time}")
    print(f"Total baris         : {len(df)}  (train={len(train_df)}, test={len(test_df)})")

    # --- Breakdown per bulan (berdasarkan base_time) ---
    for label, d in [("df", df), ("train_df", train_df), ("test_df", test_df)]:
        d["__bulan"] = d["base_time"].dt.to_period("M")

    all_months = sorted(df["__bulan"].unique())

    train_counts = train_df.groupby("__bulan")["base_time"].nunique()
    test_counts = test_df.groupby("__bulan")["base_time"].nunique()
    total_counts = df.groupby("__bulan")["base_time"].nunique()

    rows = []
    for m in all_months:
        n_train = int(train_counts.get(m, 0))
        n_test = int(test_counts.get(m, 0))
        n_total = int(total_counts.get(m, 0))
        test_pct = (n_test / n_total * 100) if n_total > 0 else 0.0
        rows.append({
            "bulan": str(m),
            "n_base_time_total": n_total,
            "n_base_time_train": n_train,
            "n_base_time_test": n_test,
            "pct_test": round(test_pct, 1),
        })

    summary = pd.DataFrame(rows)
    print("\n" + "-" * 90)
    print("BREAKDOWN PER BULAN (dihitung dari unique base_time, bukan baris/piksel)")
    print("-" * 90)
    print(summary.to_string(index=False))

    # --- Highlight bulan bermasalah ---
    zero_test = summary[summary["n_base_time_test"] == 0]
    low_test = summary[(summary["n_base_time_test"] > 0) & (summary["pct_test"] < 5.0)]

    print("\n" + "-" * 90)
    print("HASIL DIAGNOSIS")
    print("-" * 90)
    if not zero_test.empty:
        print(f"[MASALAH] {len(zero_test)} bulan TIDAK PUNYA representasi test sama sekali (0%):")
        print("  " + ", ".join(zero_test["bulan"].tolist()))
        print("  -> Model TIDAK PERNAH dievaluasi untuk kondisi di bulan-bulan ini,")
        print("     walau datanya mungkin ada penuh di training set.")
    else:
        print("[OK] Semua bulan yang ada di data punya representasi test > 0%.")

    if not low_test.empty:
        print(f"\n[PERHATIAN] {len(low_test)} bulan representasi test-nya < 5% dari data bulan itu:")
        print("  " + ", ".join(low_test["bulan"].tolist()))
        print("  -> Evaluasi untuk bulan ini secara statistik lemah (n kecil),")
        print("     mae_expected bisa tidak representatif untuk kondisi bulan tsb.")

    print(
        "\n[CATATAN] chronological_split() ambil test_frac dari EKOR KRONOLOGIS MURNI "
        "(bukan stratified). Kalau rentang data mencakup banyak bulan, wajar kalau "
        "bulan-bulan awal (terutama yang jauh dari ekor, mis. Januari kalau data "
        "berakhir di Juni) representasinya 0% di test -- itu bukan bug, tapi memang "
        "cara kerja split kronologis biasa. Ini yang jadi dasar usulan "
        "stratified_monthly_split() di Prioritas 2."
    )


if __name__ == "__main__":
    main()