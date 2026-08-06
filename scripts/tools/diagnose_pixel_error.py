# ./scripts/tools/diagnose_pixel_error.py
# Diagnosa anomali MAE tinggi di visualisasi forecast: dump error per-piksel untuk satu
# step/t0 tertentu, lalu cross-check piksel outlier terhadap features_10min_ar.csv untuk
# mendeteksi nilai di luar rentang fisik wajar atau baris duplikat/inkonsisten.

import argparse
import os
import sys

# 'pipeline' & 'ui' ada di scripts/, sedangkan file ini ada di scripts/tools/ --
# tambahkan scripts/ ke sys.path biar importnya jalan dari mana pun skrip ini dipanggil.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from pipeline.config import load_config
from pipeline.model_training import load_ar_dataset

# Rentang fisik wajar untuk Cloud Top Temperature (Kelvin). Di luar rentang ini
# hampir pasti bukan cuaca asli -- indikasi fill value / data korup.
CTT_PLAUSIBLE_MIN = 180.0
CTT_PLAUSIBLE_MAX = 310.0


def parse_args():
    p = argparse.ArgumentParser(description="Dump error per-piksel untuk satu step forecast.")
    p.add_argument("--t0", default=None, help="t0 (UTC), format 'YYYY-MM-DD HH:MM:SS'. Default: dari last_run_state.json")
    p.add_argument("--forecast-csv", default=None, help="Path relatif ke forecast_output/, default dari last_run_state.json")
    p.add_argument("--step", type=int, required=True, help="Step yang mau didiagnosa (mis. 6 = +60 menit)")
    p.add_argument("--top", type=int, default=35, help="Berapa baris teratas yang ditampilkan (default: semua/35)")
    return p.parse_args()


def load_last_run_state(root_output_dir):
    state_path = os.path.join(root_output_dir, "last_run_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"{state_path} tidak ditemukan. Isi --t0 dan --forecast-csv manual.")
    import json
    with open(state_path) as f:
        return json.load(f)


def main():
    args = parse_args()
    cfg = load_config()
    dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")
    root_output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")

    t0_str, forecast_csv = args.t0, args.forecast_csv
    if t0_str is None or forecast_csv is None:
        state = load_last_run_state(root_output_dir)
        t0_str = t0_str or state["t0"]
        forecast_csv = forecast_csv or state["forecast_csv_full10min"]
        print(f"[info] Otomatis pakai run terakhir: t0={t0_str}, file={forecast_csv}")

    t0 = pd.to_datetime(t0_str)
    forecast_path = os.path.join(root_output_dir, forecast_csv)
    if not os.path.exists(forecast_path):
        raise FileNotFoundError(forecast_path)

    forecast_df = pd.read_csv(forecast_path, parse_dates=["forecast_time"])
    frame_df = forecast_df[forecast_df["step"] == args.step].copy()
    if frame_df.empty:
        raise ValueError(f"Tidak ada data untuk step={args.step}. Step tersedia: {sorted(forecast_df['step'].unique())}")

    target_time = frame_df["forecast_time"].iloc[0]

    # ---- 1) Tabel error per-piksel, terurut descending ----
    frame_df["abs_error"] = (frame_df["predicted_tbb13"] - frame_df["actual_tbb13"]).abs()
    frame_df = frame_df.sort_values("abs_error", ascending=False)

    print("=" * 90)
    print(f"STEP {args.step}  |  t0={t0}  ->  target_time={target_time}")
    print("=" * 90)
    cols = ["pixel_row", "pixel_col", "lat", "lon", "predicted_tbb13", "actual_tbb13", "abs_error"]
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(frame_df[cols].head(args.top).to_string(index=False))

    mae = frame_df["abs_error"].mean()
    print(f"\nMAE agregat (nanmean 35 piksel): {mae:.3f}K")

    # ---- 2) Duplikat/inkonsistensi di dalam frame itu sendiri ----
    dup_in_frame = frame_df[frame_df.duplicated(subset=["pixel_row", "pixel_col"], keep=False)]
    if not dup_in_frame.empty:
        print("\n[WARNING] Ada baris duplikat (pixel_row, pixel_col) di dalam frame step ini:")
        print(dup_in_frame[cols].to_string(index=False))

    # ---- 3) Piksel di luar rentang fisik wajar (indikasi fill value / data korup) ----
    implausible = frame_df[
        (frame_df["actual_tbb13"] < CTT_PLAUSIBLE_MIN) | (frame_df["actual_tbb13"] > CTT_PLAUSIBLE_MAX)
    ]
    if not implausible.empty:
        print(f"\n[WARNING] Nilai actual_tbb13 di luar rentang fisik wajar ({CTT_PLAUSIBLE_MIN}-{CTT_PLAUSIBLE_MAX}K):")
        print(implausible[cols].to_string(index=False))

    # ---- 4) Cross-check piksel dengan error terbesar ke features_10min_ar.csv ----
    worst = frame_df.iloc[0]
    pr, pc = int(worst["pixel_row"]), int(worst["pixel_col"])
    print(f"\n{'-'*90}\nCross-check piksel terburuk (row={pr}, col={pc}) ke features_10min_ar.csv")
    print(f"{'-'*90}")

    df10 = load_ar_dataset(os.path.join(dataset_dir, "features_10min_ar.csv"))
    px = df10[(df10["pixel_row"] == pr) & (df10["pixel_col"] == pc)]

    # baris di mana base_time == target_time forecast (tbb_13_t harus == actual_tbb13 di atas)
    match_base = px[px["base_time"] == target_time]
    # baris di mana target_time == target_time forecast (target_tbb_13 juga harus == actual_tbb13)
    match_target = px[px["target_time"] == target_time]

    print(f"Baris dengan base_time == {target_time} (tbb_13_t seharusnya = actual_tbb13 di atas):")
    print(match_base[["base_time", "target_time", "tbb_13_t", "target_tbb_13"]].to_string(index=False) if not match_base.empty else "  (tidak ada)")
    print(f"\nBaris dengan target_time == {target_time} (target_tbb_13 seharusnya = actual_tbb13 di atas):")
    print(match_target[["base_time", "target_time", "tbb_13_t", "target_tbb_13"]].to_string(index=False) if not match_target.empty else "  (tidak ada)")

    if len(match_base) > 1 or len(match_target) > 1:
        print("\n[WARNING] Lebih dari satu baris cocok -- ada DUPLIKAT (pixel_row, pixel_col, base_time) "
              "atau (pixel_row, pixel_col, target_time) di features_10min_ar.csv.")
        print("Cek lebih lanjut dengan:")
        print('  df10.duplicated(subset=["base_time","pixel_row","pixel_col"], keep=False)')

    # ---- 5) Cek duplikat global di seluruh dataset (bukan cuma piksel ini) ----
    dup_global = df10.duplicated(subset=["base_time", "pixel_row", "pixel_col"], keep=False)
    if dup_global.any():
        print(f"\n[WARNING] Ditemukan {int(dup_global.sum())} baris duplikat (base_time, pixel_row, pixel_col) "
              f"di SELURUH features_10min_ar.csv (bukan cuma piksel terburuk).")
    else:
        print("\n[OK] Tidak ada duplikat (base_time, pixel_row, pixel_col) di features_10min_ar.csv.")


if __name__ == "__main__":
    main()