# ./scripts/tools/diagnose_bias_by_hour.py
#
# Diagnostik: apakah error rollout (terutama di step-step akhir) terkonsentrasi
# di jam-jam tertentu (WIB) -- indikasi model "kalah" ngikutin trend diurnal
# asli (mis. transisi konvektif siang->sore), BUKAN sekadar noise acak yang
# harusnya diredam damping.
#
# Bias di sini didefinisikan signed: (y_pred - y_true).
#   bias > 0 -> model OVER-predict (TBB diprediksi lebih tinggi/hangat dari
#               aktual -- kemungkinan model "telat" ngikutin penurunan TBB
#               nyata saat konvektif berkembang).
#   bias < 0 -> model UNDER-predict.
#
# Kalau |bias| membesar tajam di jam-jam transisi konvektif (siang->sore WIB)
# DAN kolom run yang pakai damping_rate lebih tinggi punya |bias| yang LEBIH
# BESAR lagi di jam yang sama -- itu konfirmasi kuat kalau damping (reference
# = rata-rata window statis) menarik prediksi balik ke kondisi SEBELUM
# konvektif berkembang, melawan trend asli, bukan meredam noise.
#
# CARA PAKAI:
#   1. Jalankan 06_evaluate_test.py utk tiap damping_rate yang mau
#      dibandingkan (0.0 vs beberapa nilai > 0 yang udah lo coba), simpan
#      test_evaluation_detail.csv masing-masing dengan nama beda, mis.:
#        evaluation/test_evaluation_detail_damping0.csv
#        evaluation/test_evaluation_detail_damping03.csv
#   2. Jalankan skrip ini per file (atau sekaligus, lihat --files):
#
#        py scripts/tools/diagnose_bias_by_hour.py ^
#            --files evaluation/test_evaluation_detail_damping0.csv:damping=0.0 ^
#                    evaluation/test_evaluation_detail_damping03.csv:damping=0.3 ^
#            --step-min 12 --step-max 18
#
#   3. Baca output: tabel bias rata-rata per jam WIB, dan flag jam dengan
#      |bias| terbesar. Bandingin antar run damping -- kalau bias di jam
#      transisi konvektif makin gede seiring damping naik, itu bukti model
#      ketarik ke reference yang salah arah.
#
# OUTPUT: dicetak ke terminal (tabel ringkas), TIDAK menulis file baru --
# ini alat eksplorasi cepat, bukan bagian pipeline resmi.

import argparse
import sys

import numpy as np
import pandas as pd

WIB_OFFSET_HOURS = 7  # target_time di detail_df masih UTC (Himawari native)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnostik bias signed per jam WIB dari test_evaluation_detail.csv "
            "-- buat cek apakah damping melawan trend diurnal asli."
        )
    )

    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help=(
            "Satu atau lebih path ke detail CSV, format "
            "'path.csv:label' (label opsional, default nama file). "
            "Contoh: evaluation/test_evaluation_detail.csv:damping=0.0"
        ),
    )

    parser.add_argument(
        "--models",
        default=None,
        help="Comma-separated model names buat filter (default: semua model di file).",
    )

    parser.add_argument(
        "--step-min",
        type=int,
        default=12,
        help="Batas bawah step yang dianalisis (default 12 -- horizon panjang, paling relevan).",
    )

    parser.add_argument(
        "--step-max",
        type=int,
        default=18,
        help="Batas atas step yang dianalisis (default 18).",
    )

    return parser.parse_args()


def load_labeled(file_spec):
    if ":" in file_spec and not file_spec.split(":")[0].endswith(("csv",)):
        # jaga-jaga kalau path Windows pakai 'C:\...' -- split dari kanan
        path, label = file_spec.rsplit(":", 1)
    elif file_spec.count(":") >= 1 and not file_spec[1:3] == ":\\":
        path, label = file_spec.rsplit(":", 1)
    else:
        path, label = file_spec, file_spec

    return path.strip(), label.strip()


def main():
    args = parse_args()

    model_filter = (
        [m.strip() for m in args.models.split(",")] if args.models else None
    )

    all_summaries = []

    for spec in args.files:
        path, label = load_labeled(spec)

        print(f"\n{'=' * 70}")
        print(f"FILE  : {path}")
        print(f"LABEL : {label}")
        print(f"{'=' * 70}")

        df = pd.read_csv(path, parse_dates=["target_time"])

        if model_filter is not None:
            df = df[df["model"].isin(model_filter)]

        df = df[(df["step"] >= args.step_min) & (df["step"] <= args.step_max)]

        if len(df) == 0:
            print(
                f"  (kosong setelah filter step {args.step_min}-{args.step_max} "
                f"/ model {model_filter} -- skip)"
            )
            continue

        # target_time asumsinya UTC (native Himawari, lihat time_features.py) --
        # +7 jam ke WIB biar konsisten sama fitur waktu yang dilihat model.
        target_time_wib = df["target_time"] + pd.Timedelta(hours=WIB_OFFSET_HOURS)
        df = df.assign(hour_wib=target_time_wib.dt.hour)

        df = df.assign(signed_error=df["y_pred"] - df["y_true"])

        for model_name, group in df.groupby("model"):
            summary = (
                group.groupby("hour_wib")
                .agg(
                    bias=("signed_error", "mean"),
                    mae=("abs_error", "mean"),
                    n=("signed_error", "size"),
                )
                .reset_index()
                .sort_values("hour_wib")
            )
            summary["label"] = label
            summary["model"] = model_name
            all_summaries.append(summary)

            print(f"\n  --- model={model_name} (step {args.step_min}-{args.step_max}) ---")
            print(
                summary[["hour_wib", "bias", "mae", "n"]]
                .to_string(index=False, float_format=lambda v: f"{v:8.3f}")
            )

            worst = summary.reindex(
                summary["bias"].abs().sort_values(ascending=False).index
            ).head(3)
            print(
                f"\n  Jam dengan |bias| terbesar: "
                + ", ".join(
                    f"{int(r.hour_wib):02d}:00 WIB (bias={r.bias:+.3f}K, n={int(r.n)})"
                    for r in worst.itertuples()
                )
            )

    if len(all_summaries) >= 2:
        combined = pd.concat(all_summaries, ignore_index=True)
        pivot = combined.pivot_table(
            index=["model", "hour_wib"], columns="label", values="bias"
        )
        print(f"\n{'=' * 70}")
        print("PERBANDINGAN BIAS ANTAR RUN (kolom = label file)")
        print(f"{'=' * 70}")
        print(pivot.to_string(float_format=lambda v: f"{v:+7.3f}"))
        print(
            "\nKalau |bias| di jam yang sama membesar seiring damping_rate naik, "
            "itu indikasi damping menarik prediksi ke arah yang salah "
            "(melawan trend diurnal asli), bukan meredam noise."
        )


if __name__ == "__main__":
    sys.exit(main())