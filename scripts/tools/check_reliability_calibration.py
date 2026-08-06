# ./scripts/tools/check_reliability_calibration.py
# Verifikasi kalibrasi reliability: bandingkan `mae_expected` (ditempel
# annotate_reliability() dari recursive_evaluation.csv Tahap 5) vs MAE AKTUAL
# yang benar-benar terjadi di satu run 06_run_inference.py -- per step.
#
# Konteks: sebelum split/t0 selection diperbaiki (stratified_monthly), run
# Januari (kondisi konvektif) punya MAE aktual 15.59K padahal mae_expected
# cuma ~2.5K (rasio 6x) -- karena recursive_evaluation.csv lama dihitung dari
# test set yang bias ke bulan tenang. Script ini dipakai untuk cek apakah
# recursive_evaluation.csv YANG BARU (representatif lintas bulan) sudah
# memperbaiki kalibrasi ini, TANPA perlu ubah kode reliability lagi kalau
# ternyata sudah cukup baik.
#
# CARA PAKAI:
#   cd scripts
#   python 06_run_inference.py --t0 "2026-01-03 10:10:00"
#   python tools/check_reliability_calibration.py
#   (atau kalau mau tunjuk run tertentu, bukan run terakhir:)
#   python tools/check_reliability_calibration.py --tag 20260103_1010_xgboost
#
# Fix #7 (Prioritas 1, opsi A): kalau forecast_df hasil 06 punya kolom
# `mae_expected_conservative` (persentil p75/p90 antar-t0, lihat
# pipeline/inference.py::annotate_reliability), script ini otomatis ikut
# bandingkan rasio aktual/conservative di samping rasio aktual/expected
# (mean) yang lama -- buat verifikasi apakah versi konservatif berhasil
# mengecilkan rasio yang tadinya 2-2.3x (residual issue di briefing v2,
# poin 7) mendekati 1x, tanpa perlu re-run recursive_evaluation ulang lagi
# tiap kali mau cek.

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from pipeline.config import load_config

# Rasio aktual/expected di luar rentang ini dianggap "kalibrasi buruk" --
# sama semangatnya dengan COLLAPSE_RATIO_THRESHOLD di 05_recursive_evaluation.py,
# tapi ini mengecek AKURASI (MAE), bukan sebaran spasial (collapse_ratio).
RATIO_WARNING_LOW = 0.5
RATIO_WARNING_HIGH = 2.0


def parse_args():
    p = argparse.ArgumentParser(
        description="Bandingkan mae_expected vs MAE aktual per step dari satu run 06_run_inference.py."
    )
    p.add_argument(
        "--tag", default=None,
        help="Nama folder di forecast_output/ (mis. '20260103_1010_xgboost'). Default: run terakhir (last_run_state.json).",
    )
    return p.parse_args()


def resolve_run(tag, forecast_output_dir):
    if tag is not None:
        forecast_csv = os.path.join(forecast_output_dir, tag, "full10min.csv")
        if not os.path.exists(forecast_csv):
            raise FileNotFoundError(forecast_csv)
        return forecast_csv, tag

    state_path = os.path.join(forecast_output_dir, "last_run_state.json")
    if not os.path.exists(state_path):
        raise FileNotFoundError(f"{state_path} tidak ditemukan. Isi --tag manual.")
    import json
    with open(state_path) as f:
        state = json.load(f)
    return os.path.join(forecast_output_dir, state["forecast_csv_full10min"]), state.get("t0_tag", "?")


def main():
    args = parse_args()
    cfg = load_config()
    forecast_output_dir = os.path.join(cfg.PROJECT_ROOT, "forecast_output")

    forecast_csv, tag = resolve_run(args.tag, forecast_output_dir)
    print(f"Run: {tag}\nFile: {forecast_csv}\n")

    df = pd.read_csv(forecast_csv)
    required_cols = {"step", "predicted_tbb13", "actual_tbb13"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"[ERROR] File tidak punya kolom {missing} -- pastikan ini output 06_run_inference.py (full10min.csv).")
        return

    has_reliability_cols = {"mae_expected", "reliable"}.issubset(df.columns)
    if not has_reliability_cols:
        print(
            "[PERINGATAN] Kolom 'mae_expected'/'reliable' tidak ada di file ini "
            "(kemungkinan recursive_evaluation.csv belum ada saat run 06 dijalankan). "
            "Tetap tampilkan MAE aktual per step saja."
        )

    has_conservative_col = (
        "mae_expected_conservative" in df.columns
        and df["mae_expected_conservative"].notna().any()
    )
    if has_reliability_cols and not has_conservative_col:
        print(
            "[INFO] Kolom 'mae_expected_conservative' belum ada/kosong -- "
            "recursive_evaluation.csv kemungkinan belum di-rerun setelah fix #7 "
            "(Prioritas 1 opsi A). Hanya bandingkan mae_expected (mean) seperti biasa."
        )

    df["abs_error"] = (df["predicted_tbb13"] - df["actual_tbb13"]).abs()
    df_valid = df.dropna(subset=["actual_tbb13"])

    if df_valid.empty:
        print("[INFO] Tidak ada baris dengan actual_tbb13 terisi -- t0 ini di luar rentang ground-truth yang ada.")
        return

    rows = []
    for step, g in df_valid.groupby("step"):
        actual_mae = g["abs_error"].mean()
        menit = int(g["menit_ke_depan"].iloc[0]) if "menit_ke_depan" in g.columns else int(step) * 10
        row = {"step": int(step), "menit_ke_depan": menit, "mae_aktual": actual_mae}
        if has_reliability_cols:
            mae_expected = g["mae_expected"].iloc[0]
            reliable = bool(g["reliable"].iloc[0])
            ratio = actual_mae / mae_expected if mae_expected and mae_expected > 0 else np.nan
            row.update({"mae_expected": mae_expected, "rasio_aktual/expected": ratio, "reliable_flag": reliable})
            if has_conservative_col:
                mae_cons = g["mae_expected_conservative"].iloc[0]
                ratio_cons = actual_mae / mae_cons if mae_cons and mae_cons > 0 else np.nan
                row.update({"mae_expected_conservative": mae_cons, "rasio_aktual/conservative": ratio_cons})
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("step")

    print("=" * 100)
    print("MAE AKTUAL vs MAE_EXPECTED PER STEP")
    print("=" * 100)
    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 140):
        print(summary.to_string(index=False))

    if has_reliability_cols:
        overall_actual = df_valid["abs_error"].mean()
        overall_expected = summary["mae_expected"].mean()
        print(f"\nMAE aktual keseluruhan   : {overall_actual:.3f}K")
        print(f"MAE expected rata-rata   : {overall_expected:.3f}K")
        print(f"Rasio keseluruhan        : {overall_actual / overall_expected:.2f}x" if overall_expected else "")

        badly_calibrated = summary[
            (summary["rasio_aktual/expected"] < RATIO_WARNING_LOW)
            | (summary["rasio_aktual/expected"] > RATIO_WARNING_HIGH)
        ]
        flagged_unreliable_but_ok = summary[(~summary["reliable_flag"]) & (summary["rasio_aktual/expected"] <= RATIO_WARNING_HIGH)]
        missed_by_flag = summary[(summary["reliable_flag"]) & (
            (summary["rasio_aktual/expected"] < RATIO_WARNING_LOW) | (summary["rasio_aktual/expected"] > RATIO_WARNING_HIGH)
        )]

        print("\n" + "-" * 100)
        if badly_calibrated.empty:
            print(
                f"[OK] Semua step rasio aktual/expected dalam rentang wajar "
                f"({RATIO_WARNING_LOW}x-{RATIO_WARNING_HIGH}x) -- kalibrasi reliability sudah cukup baik "
                "untuk run ini."
            )
        else:
            print(
                f"[PERHATIAN] {len(badly_calibrated)} step rasio aktual/expected di luar "
                f"{RATIO_WARNING_LOW}x-{RATIO_WARNING_HIGH}x:"
            )
            print(badly_calibrated[["step", "menit_ke_depan", "rasio_aktual/expected", "reliable_flag"]].to_string(index=False))

        if not missed_by_flag.empty:
            print(
                f"\n[MASALAH SERIUS] {len(missed_by_flag)} step kalibrasinya buruk TAPI flag 'reliable' "
                "masih True -- annotate_reliability() gagal deteksi (kasus persis seperti temuan Januari "
                "di briefing). Step-step ini:"
            )
            print(missed_by_flag[["step", "menit_ke_depan", "rasio_aktual/expected"]].to_string(index=False))

        if has_conservative_col:
            print("\n" + "-" * 100)
            print("[Fix #7] mae_expected_conservative (persentil antar-t0) vs mae_expected (mean pooled):")
            badly_calibrated_cons = summary[
                (summary["rasio_aktual/conservative"] < RATIO_WARNING_LOW)
                | (summary["rasio_aktual/conservative"] > RATIO_WARNING_HIGH)
            ]
            if badly_calibrated_cons.empty:
                print(
                    f"[OK] Semua step rasio aktual/conservative dalam rentang wajar "
                    f"({RATIO_WARNING_LOW}x-{RATIO_WARNING_HIGH}x) -- versi konservatif cukup aman untuk run ini."
                )
            else:
                print(
                    f"[PERHATIAN] {len(badly_calibrated_cons)} step MASIH di luar rentang wajar "
                    "walau sudah pakai mae_expected_conservative:"
                )
                print(badly_calibrated_cons[["step", "menit_ke_depan", "rasio_aktual/conservative"]].to_string(index=False))

            fixed_by_conservative = badly_calibrated[~badly_calibrated["step"].isin(badly_calibrated_cons["step"])]
            if not fixed_by_conservative.empty:
                compare_cols = ["step", "menit_ke_depan", "rasio_aktual/expected", "rasio_aktual/conservative"]
                print(
                    f"\n[MEMBAIK] {len(fixed_by_conservative)} step yang tadinya di luar rentang wajar "
                    "(pakai mean) sekarang masuk rentang wajar pakai mae_expected_conservative:"
                )
                print(fixed_by_conservative[compare_cols].to_string(index=False))
            elif not badly_calibrated.empty:
                print(
                    "\n[PERHATIAN] Tidak ada step yang membaik dengan mae_expected_conservative -- "
                    "coba ganti CONSERVATIVE_MAE_COLUMN di pipeline/inference.py dari 'mae_p75' ke "
                    "'mae_p90' untuk versi yang lebih konservatif lagi."
                )


if __name__ == "__main__":
    main()