# ./scripts/10_atmospheric_analysis.py
#
# Tahap 10 (analisis tambahan, non-ML): mining pola atmosfer dari deret
# CTT mentah -- siklus diurnal, klasifikasi risiko konvektif, deteksi
# event onset konvektif, clustering spasial pixel, uji stasioneritas,
# ACF/PACF. Lihat pipeline/atmospheric_analysis.py untuk detail metodologi
# tiap fungsi.
#
# INPUT:
#   dataset/temporal_split/train_temporal.npz (atau file lain via --cache)
#
# OUTPUT (semua di summary_report/atmospheric/):
#   diurnal_cycle.csv, convective_risk.csv, onset_events.csv,
#   pixel_clusters.csv, pixel_correlation_matrix.csv,
#   stationarity_per_pixel.csv, distribution_summary.csv
#   diurnal_cycle.png, pixel_correlation_heatmap.png, acf_pacf_sample.png

import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import load_config
from pipeline.temporal_dataset import load_temporal_cache
from pipeline.atmospheric_analysis import (
    diurnal_cycle_profile,
    classify_convective_risk,
    detect_convective_onset_events,
    spatial_cluster_pixels,
    stationarity_test,
    acf_pacf,
    distribution_summary,
)
from ui.terminal_display import banner, gap, hr, say_error, say_info, say_ok


def parse_args():
    parser = argparse.ArgumentParser(description="Tahap 10: analisis mining atmosfer.")
    parser.add_argument("--cache", default=None, help="Default: Config.TEMPORAL_TRAIN_FILE.")
    parser.add_argument("--output-dir", default=None, help="Default: Config.SUMMARY_REPORT_DIR/atmospheric.")
    parser.add_argument("--n-clusters", type=int, default=4)
    parser.add_argument("--sample-pixel-idx", type=int, default=0, help="Pixel untuk ACF/PACF & stationarity contoh.")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    cache_path = args.cache or cfg.TEMPORAL_TRAIN_FILE
    output_dir = args.output_dir or os.path.join(cfg.SUMMARY_REPORT_DIR, "atmospheric")
    os.makedirs(output_dir, exist_ok=True)

    banner("TAHAP 10: ATMOSPHERIC MINING ANALYSIS")
    say_info(f"Cache  : {cache_path}")
    say_info(f"Output : {output_dir}")
    hr()

    try:
        data_matrix, timeline, pixel_meta = load_temporal_cache(cache_path)
        say_info(f"Shape data_matrix: {data_matrix.shape} (T x P)")

        # 1. Siklus diurnal ----------------------------------------------
        gap()
        say_info("1/6 Diurnal cycle profile (WIB)...")
        diurnal_df = diurnal_cycle_profile(data_matrix, timeline)
        diurnal_df.to_csv(os.path.join(output_dir, "diurnal_cycle.csv"), index=False)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(diurnal_df["hour_wib"], diurnal_df["mean_ctt"], marker="o", color="#1f6f8b")
        ax.fill_between(
            diurnal_df["hour_wib"],
            diurnal_df["mean_ctt"] - diurnal_df["std_ctt"],
            diurnal_df["mean_ctt"] + diurnal_df["std_ctt"],
            alpha=0.2, color="#1f6f8b",
        )
        ax.set_xlabel("Jam lokal (WIB)")
        ax.set_ylabel("Rata-rata CTT (K)")
        ax.set_title("Siklus Diurnal CTT -- Bandung Raya")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "diurnal_cycle.png"), dpi=150)
        plt.close(fig)
        say_ok(f"  jam CTT terendah (rata2): {int(diurnal_df.loc[diurnal_df['mean_ctt'].idxmin(), 'hour_wib'])}:00 WIB")

        # 2. Klasifikasi risiko konvektif ---------------------------------
        gap()
        say_info("2/6 Convective risk classification...")
        risk = classify_convective_risk(data_matrix)
        pd.DataFrame([risk]).to_csv(os.path.join(output_dir, "convective_risk.csv"), index=False)
        say_ok(
            f"  deep_convective={risk['deep_convective']*100:.2f}% "
            f"moderate_cloud={risk['moderate_cloud']*100:.2f}% "
            f"clear={risk['clear']*100:.2f}%"
        )

        # 3. Deteksi onset konvektif (semua pixel) ------------------------
        gap()
        say_info("3/6 Convective onset event detection...")
        all_events = []
        P = data_matrix.shape[1]
        for p in range(P):
            ev = detect_convective_onset_events(data_matrix[:, p], timeline)
            if len(ev) > 0:
                ev["pixel_id"] = pixel_meta["pixel_id"][p]
                all_events.append(ev)
        events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()
        events_df.to_csv(os.path.join(output_dir, "onset_events.csv"), index=False)
        say_ok(f"  total event onset terdeteksi: {len(events_df)} (semua {P} pixel)")

        # 4. Clustering spasial ------------------------------------------
        gap()
        say_info("4/6 Spatial clustering pixel...")
        clusters_df, corr_matrix = spatial_cluster_pixels(
            data_matrix, pixel_meta, n_clusters=args.n_clusters
        )
        clusters_df.to_csv(os.path.join(output_dir, "pixel_clusters.csv"), index=False)
        pd.DataFrame(corr_matrix).to_csv(
            os.path.join(output_dir, "pixel_correlation_matrix.csv"), index=False
        )

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(corr_matrix, cmap="RdYlBu_r", vmin=-1, vmax=1)
        ax.set_title("Korelasi Temporal Antar-Pixel CTT")
        ax.set_xlabel("Pixel index")
        ax.set_ylabel("Pixel index")
        fig.colorbar(im, ax=ax, label="Korelasi")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "pixel_correlation_heatmap.png"), dpi=150)
        plt.close(fig)
        say_ok(f"  {args.n_clusters} cluster spasial terbentuk dari {P} pixel")

        # 5. Stasioneritas per pixel ---------------------------------------
        gap()
        say_info("5/6 Stationarity test (ADF) per pixel...")
        stat_rows = []
        for p in range(P):
            try:
                res = stationarity_test(data_matrix[:, p])
            except ImportError as exc:
                say_error(f"  {exc}")
                res = {"adf_stat": np.nan, "p_value": np.nan, "is_stationary_5pct": None, "n_obs_used": 0}
            res["pixel_id"] = pixel_meta["pixel_id"][p]
            stat_rows.append(res)
        stat_df = pd.DataFrame(stat_rows)
        stat_df.to_csv(os.path.join(output_dir, "stationarity_per_pixel.csv"), index=False)
        n_stationary = stat_df["is_stationary_5pct"].sum() if stat_df["is_stationary_5pct"].notna().any() else 0
        say_ok(f"  {n_stationary}/{P} pixel stasioner (ADF, p<0.05)")

        # 6. ACF/PACF contoh + distribusi ----------------------------------
        gap()
        say_info("6/6 ACF/PACF (pixel contoh) + distribution summary...")
        sample_y = data_matrix[:, args.sample_pixel_idx]
        try:
            acfp_df = acf_pacf(sample_y, n_lags=24)
            acfp_df.to_csv(os.path.join(output_dir, "acf_pacf_sample.csv"), index=False)

            fig, axes = plt.subplots(1, 2, figsize=(11, 4))
            axes[0].stem(acfp_df["lag"], acfp_df["acf"])
            axes[0].set_title("ACF")
            axes[0].set_xlabel("Lag (x10 menit)")
            axes[0].axhline(0, color="black", linewidth=0.8)
            axes[1].stem(acfp_df["lag"], acfp_df["pacf"])
            axes[1].set_title("PACF")
            axes[1].set_xlabel("Lag (x10 menit)")
            axes[1].axhline(0, color="black", linewidth=0.8)
            fig.suptitle(f"ACF/PACF -- pixel_id={pixel_meta['pixel_id'][args.sample_pixel_idx]}")
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, "acf_pacf_sample.png"), dpi=150)
            plt.close(fig)
        except ImportError as exc:
            say_error(f"  {exc}")

        dist = distribution_summary(data_matrix)
        pd.DataFrame([dist]).to_csv(os.path.join(output_dir, "distribution_summary.csv"), index=False)
        say_ok(
            f"  mean={dist['mean']:.1f}K std={dist['std']:.1f}K "
            f"skew={dist['skewness']:.2f} kurtosis={dist['kurtosis']:.2f}"
        )

        hr()
        say_ok("Tahap 10 selesai.")
        say_info(f"Semua output di: {output_dir}")

    except Exception as exc:
        say_error(f"Tahap 10 gagal: {exc}")
        raise


if __name__ == "__main__":
    main()