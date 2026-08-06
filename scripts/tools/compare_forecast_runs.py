# compare_forecast_runs.py
# Bandingkan 2+ hasil run 06_run_inference.py (mis. sebelum vs sesudah
# retrain model, atau model terbaik lama vs baru) berdasarkan full10min.csv
# masing-masing -- MAE per step, MAE keseluruhan, dan MAE dipecah horizon
# pendek (<=90 menit) vs horizon jauh (>90 menit).
#
# CARA PAKAI:
#   cd scripts
#   python tools/compare_forecast_runs.py <tag1> <tag2> [<tag3> ...]
#
# <tag> = nama folder di dalam forecast_output/ (mis. "20260103_1010_catboost"),
# ATAU path lengkap ke folder itu kalau lokasinya di luar forecast_output/.
#
# Contoh:
#   python tools/compare_forecast_runs.py 20260103_1010_catboost 20260110_0800_catboost

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.config import load_config

HORIZON_SPLIT_MINUTES = 90  # titik pemisah "horizon pendek" vs "horizon jauh" di ringkasan


def resolve_run_path(tag, forecast_output_dir):
    """Terima nama folder (tag) ATAU path lengkap, kembalikan path ke full10min.csv."""
    candidate = tag if os.path.isabs(tag) or os.path.exists(tag) else os.path.join(forecast_output_dir, tag)
    csv_path = os.path.join(candidate, "full10min.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Tidak ditemukan: {csv_path}")
    return csv_path


def load_run(tag, forecast_output_dir):
    csv_path = resolve_run_path(tag, forecast_output_dir)
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["actual_tbb13"]).copy()
    df["abs_error"] = (df["predicted_tbb13"] - df["actual_tbb13"]).abs()
    df["menit_ke_depan"] = df["step"] * 10  # full10min.csv selalu resolusi 10 menit
    return df


def per_step_mae(df):
    g = df.groupby(["step", "menit_ke_depan"]).agg(
        n=("abs_error", "size"),
        mae=("abs_error", "mean"),
    ).reset_index()
    return g


def overall_mae(df, max_minutes=None, min_minutes=None):
    sub = df
    if max_minutes is not None:
        sub = sub[sub["menit_ke_depan"] <= max_minutes]
    if min_minutes is not None:
        sub = sub[sub["menit_ke_depan"] > min_minutes]
    if sub.empty:
        return np.nan, 0
    return sub["abs_error"].mean(), len(sub)


def main():
    parser = argparse.ArgumentParser(description="Bandingkan beberapa hasil run 06_run_inference.py")
    parser.add_argument("tags", nargs="+", help="Nama folder (tag) di forecast_output/, atau path lengkap")
    parser.add_argument("--save", action="store_true", help="Simpan tabel perbandingan per-step ke CSV")
    args = parser.parse_args()

    cfg = load_config()
    forecast_output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")

    runs = {}
    for tag in args.tags:
        try:
            runs[tag] = load_run(tag, forecast_output_dir)
        except FileNotFoundError as e:
            print(f"[ERROR] {e}")
            return

    # ---- Tabel MAE per step, side-by-side ----
    merged = None
    for tag, df in runs.items():
        step_df = per_step_mae(df).rename(columns={"mae": f"mae__{tag}", "n": f"n__{tag}"})
        merged = step_df if merged is None else merged.merge(
            step_df, on=["step", "menit_ke_depan"], how="outer"
        )
    merged = merged.sort_values("step")

    mae_cols = [c for c in merged.columns if c.startswith("mae__")]
    merged["mode_terbaik"] = merged[mae_cols].idxmin(axis=1).str.replace("mae__", "", regex=False)

    print("=" * 100)
    print("PERBANDINGAN MAE PER STEP")
    print("=" * 100)
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print(merged.round(3).to_string(index=False))

    # ---- Ringkasan keseluruhan + dipecah horizon pendek vs jauh ----
    print()
    print("=" * 100)
    print(f"RINGKASAN (titik pisah horizon pendek/jauh: {HORIZON_SPLIT_MINUTES} menit)")
    print("=" * 100)
    summary_rows = []
    for tag, df in runs.items():
        mae_all, n_all = overall_mae(df)
        mae_short, n_short = overall_mae(df, max_minutes=HORIZON_SPLIT_MINUTES)
        mae_long, n_long = overall_mae(df, min_minutes=HORIZON_SPLIT_MINUTES)
        summary_rows.append({
            "run": tag,
            "mae_keseluruhan": round(mae_all, 4) if mae_all == mae_all else None,
            "n": n_all,
            f"mae_<={HORIZON_SPLIT_MINUTES}menit": round(mae_short, 4) if mae_short == mae_short else None,
            "n_pendek": n_short,
            f"mae_>{HORIZON_SPLIT_MINUTES}menit": round(mae_long, 4) if mae_long == mae_long else None,
            "n_panjang": n_long,
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    best_overall = summary_df.loc[summary_df["mae_keseluruhan"].idxmin(), "run"]
    print(f"\n>> MAE keseluruhan terendah: {best_overall}")

    if args.save:
        out_path = os.path.join(forecast_output_dir, "comparison_" + "_vs_".join(args.tags) + ".csv")
        merged.to_csv(out_path, index=False)
        print(f"\n[OK] Tabel per-step disimpan: {out_path}")


if __name__ == "__main__":
    main()
