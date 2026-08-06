# scripts/tools/diagnose_nan_target.py
# Diagnostik NaN pada target_tbb_13 (dan kolom terkait) di dataset hasil 03a/03b.
# TIDAK perlu rerun 03a -- cukup baca ulang CSV yang sudah ada.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from pipeline.config import load_config


def parse_args():
    p = argparse.ArgumentParser(description="Diagnostik NaN pada target_tbb_13.")
    p.add_argument(
        "--dataset-dir", default=None,
        help="Folder dataset (default: <PROJECT_ROOT>/dataset, dari pipeline/config.py).",
    )
    p.add_argument("--interval", type=int, default=10, choices=[10, 30, 60])
    p.add_argument(
        "--source", default="full", choices=["full", "ar"],
        help="'full' = features_{interval}min.csv (semua kanal, buat cek korelasi antar-kanal). "
             "'ar' = features_{interval}min_ar.csv (lebih kecil/cepat, tapi cuma tbb_13).",
    )
    return p.parse_args()


def pct(n, total):
    return f"{n} ({100 * n / total:.4f}%)" if total else f"{n} (0%)"


def main():
    args = parse_args()
    dataset_dir = args.dataset_dir
    if dataset_dir is None:
        cfg = load_config()
        dataset_dir = os.path.join(cfg.PROJECT_ROOT, "dataset")

    if args.source == "full":
        path = os.path.join(dataset_dir, f"features_{args.interval}min.csv")
        usecols = [
            "base_time", "target_time", "pixel_row", "pixel_col", "lat", "lon",
            "target_tbb_13",
            "tbb_13_t", "tbb_13_tm1", "tbb_13_tm2",
            "tbb13_neighbor_mean", "tbb13_neighbor_diff",
            "tbb_13_last_real_obs", "minutes_since_last_real_obs",
            # sample beberapa kanal lain buat cek apakah NaN-nya cuma tbb_13 atau nyebar
            "tbb_07_t", "tbb_10_t", "tbb_16_t",
        ]
    else:
        path = os.path.join(dataset_dir, f"features_{args.interval}min_ar.csv")
        usecols = None  # ar file sudah ramping, baca semua kolom

    if not os.path.exists(path):
        print(f"[ERROR] File tidak ditemukan: {path}")
        return

    print(f"Membaca: {path}")
    if usecols:
        # baca header dulu buat filter usecols yang benar-benar ada (jaga-jaga beda versi kolom)
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [c for c in usecols if c in header]
    df = pd.read_csv(path, usecols=usecols, parse_dates=["base_time", "target_time"] if usecols is None or "target_time" in usecols else ["base_time"])

    total = len(df)
    print(f"Total baris: {total}\n")
    print("=" * 70)
    print("1) NaN COUNT PER KOLOM RELEVAN")
    print("=" * 70)
    check_cols = [c for c in [
        "target_tbb_13", "tbb_13_t", "tbb_13_tm1", "tbb_13_tm2",
        "tbb13_neighbor_mean", "tbb13_neighbor_diff",
        "tbb_13_last_real_obs", "tbb_07_t", "tbb_10_t", "tbb_16_t",
    ] if c in df.columns]
    for c in check_cols:
        n_nan = df[c].isna().sum()
        print(f"  {c:28s}: {pct(n_nan, total)}")

    if "target_tbb_13" not in df.columns:
        print("\n[WARN] Kolom target_tbb_13 tidak ada di file ini, stop di sini.")
        return

    nan_mask = df["target_tbb_13"].isna()
    n_nan_target = nan_mask.sum()
    print(f"\nTotal baris dengan target_tbb_13 NaN: {pct(n_nan_target, total)}")

    if n_nan_target == 0:
        print("\nTidak ada NaN di target_tbb_13 pada file ini. (Cek juga source lain / interval lain kalau perlu.)")
        return

    df_nan = df[nan_mask]

    print("\n" + "=" * 70)
    print("2) SEBARAN NaN PER PIXEL (pixel_row, pixel_col)")
    print("=" * 70)
    print("-> Kalau NaN nge-cluster di baris/kolom pinggir grid, indikasi piksel di tepi region crop.")
    pixel_counts = df_nan.groupby(["pixel_row", "pixel_col"]).size().sort_values(ascending=False)
    print(pixel_counts.head(20).to_string())
    print(f"\nJumlah pixel unik yang pernah kena NaN: {pixel_counts.shape[0]} dari total kombinasi pixel di grid")

    print("\n" + "=" * 70)
    print("3) SEBARAN NaN PER JAM (hour of base_time, waktu file = UTC)")
    print("=" * 70)
    print("-> Kalau NaN terkonsentrasi di jam tertentu, indikasi terkait siklus siang/malam matahari.")
    hour_counts = df_nan["base_time"].dt.hour.value_counts().sort_index()
    for h, c in hour_counts.items():
        print(f"  jam {h:02d} UTC: {c}")

    print("\n" + "=" * 70)
    print("4) SEBARAN NaN SEPANJANG WAKTU (per bulan/tanggal)")
    print("=" * 70)
    print("-> Kalau NaN menumpuk di rentang tanggal tertentu, indikasi ada batch file korup/gap data, bukan pola fisik.")
    date_counts = df_nan["base_time"].dt.date.value_counts().sort_index()
    print(date_counts.to_string())

    if "tbb_13_t" in df.columns:
        print("\n" + "=" * 70)
        print("5) APAKAH tbb_13_t (observasi 'sekarang') DI BARIS YANG SAMA JUGA NaN?")
        print("=" * 70)
        print("-> Kalau IYA mayoritas: kemungkinan piksel itu memang rusak/selalu kosong (bukan random).")
        print("-> Kalau TIDAK (tbb_13_t valid tapi target NaN): kemungkinan besar terkait momen spesifik target_time saja")
        print("   (misal storage/observasi di 1 frame itu yang bermasalah, atau representasi 'tidak ada awan' pada momen itu).")
        also_nan = df_nan["tbb_13_t"].isna().sum()
        print(f"  tbb_13_t juga NaN pada baris yang target-nya NaN: {pct(also_nan, n_nan_target)}")

    other_channel_cols = [c for c in ["tbb_07_t", "tbb_10_t", "tbb_16_t"] if c in df.columns]
    if other_channel_cols:
        print("\n" + "=" * 70)
        print("6) APAKAH KANAL LAIN (bukan tbb_13) DI BARIS/PIXEL/WAKTU YANG SAMA JUGA NaN?")
        print("=" * 70)
        print("-> Kalau kanal lain di baris yang sama JUGA banyak NaN: indikasi seluruh piksel/frame bermasalah (data korup).")
        print("-> Kalau kanal lain VALID sementara tbb_13 doang NaN: indikasi spesifik ke kanal 13 / produk turunannya.")
        for c in other_channel_cols:
            also_nan = df_nan[c].isna().sum()
            print(f"  {c:12s} juga NaN pada baris target NaN: {pct(also_nan, n_nan_target)}")

    print("\n" + "=" * 70)
    print("RINGKASAN")
    print("=" * 70)
    print(f"- {pct(n_nan_target, total)} baris di interval {args.interval} menit punya target_tbb_13 = NaN.")
    print("- Lihat bagian 2 & 3 di atas untuk tahu apakah ini pola spasial (piksel tertentu),")
    print("  pola temporal (jam/tanggal tertentu), atau tersebar acak.")
    print("- Lihat bagian 5 & 6 untuk tahu apakah ini masalah 1 kanal spesifik (tbb_13 doang)")
    print("  atau seluruh frame ikut rusak di piksel itu.")


if __name__ == "__main__":
    main()