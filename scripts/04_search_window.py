# ./scripts/04_search_window.py
#
# Tahap 4: Window search.
#
# Untuk setiap model (xgboost/lightgbm/catboost) x setiap kandidat window
# (Config.WINDOW_CANDIDATES, 1..18):
#   1. Bangun dataset fixed-window (raw lag features) dari periode FIT
#      TRAIN (Des'25-Mar'26), latih SATU model single-step.
#   2. Evaluasi model itu dengan RECURSIVE ROLLOUT 18 step di periode
#      VALIDATION (Apr-Mei'26) -- window awal diambil dari observasi asli,
#      lalu di-rollout pakai prediksi model sendiri, dibandingkan ke
#      observasi asli di tiap step.
#   3. Simpan MAE per step + MAE rata-rata semua step, plus R^2 per step
#      + R^2 rata-rata semua step (lihat pipeline/window_model_training.py
#      r2_score_per_step, dihitung dari rollout yang SAMA -- bukan
#      rollout terpisah).
#
# TEST asli (Jun-Jul'26) SAMA SEKALI TIDAK DISENTUH di tahap ini --
# window terbaik dipilih murni dari VALIDATION, biar gak leak/bias ke
# evaluasi akhir.
#
# Kriteria pemilihan window terbaik per model: MAE rata-rata SEMUA 18
# step (bukan cuma horizon panjang atau step terakhir). R^2 dihitung &
# disimpan sebagai metrik tambahan (BUKAN kriteria seleksi) -- MAE tetap
# jadi satu-satunya kriteria supaya perilaku seleksi window tidak berubah
# dari desain sebelumnya.
#
# INPUT:
#   dataset/temporal_split/train_temporal.npz
#
# OUTPUT:
#   window_search/window_search_results.csv   (semua kombinasi model x window)
#   window_search/best_window_per_model.csv   (window terpilih per model)
#
# CATATAN RUNTIME:
# Ini melatih len(models) x len(windows) model TERPISAH (default 3 x 18 =
# 54 model). Berdasarkan smoke-test (data sintetis, 35 pixel, ~4 bulan
# FIT): ~35 detik per window untuk 3 model (xgboost+lightgbm+catboost),
# jadi full run diperkirakan ~10-15 menit. Bisa beda di data asli lo
# (lebih lama/cepat tergantung spek mesin) -- SELALU smoke-test dulu
# pakai --models dan --windows yang dipersempit sebelum full run:
#
#   python scripts/04_search_window.py --models xgboost --windows 1,6,18
#
# baru kalau itu jalan lancar & angkanya masuk akal, jalankan full run
# tanpa --models/--windows (pakai semua default).

import argparse
import os
import time

import numpy as np
import pandas as pd

from pipeline.config import load_config
from pipeline.temporal_dataset import (
    load_temporal_cache,
    chronological_split,
    slice_temporal_data,
)
from pipeline.window_features import (
    build_window_dataset,
    build_rollout_arrays,
    TARGET_COLUMN,
    lag_column_names,
)
from pipeline.window_model_training import (
    train_one_model,
    recursive_rollout_mae,
)
from ui.terminal_display import (
    banner,
    gap,
    hr,
    say_error,
    say_info,
    say_ok,
    say_skip,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Window search: cari window fixed-size terbaik per model "
            "pakai fit/validation split dari TRAIN (TEST tidak disentuh)."
        )
    )

    parser.add_argument(
        "--train-cache",
        default=None,
        help="Path ke train_temporal.npz. Default: Config.TEMPORAL_TRAIN_FILE.",
    )

    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names (xgboost,lightgbm,catboost). "
            "Default: semua Config.MODEL_NAMES."
        ),
    )

    parser.add_argument(
        "--windows",
        default=None,
        help=(
            "Comma-separated window candidates (mis. 1,6,12,18). "
            "Default: semua Config.WINDOW_CANDIDATES."
        ),
    )

    parser.add_argument(
        "--fit-end",
        default=None,
        help="Batas akhir periode FIT. Default: Config.WINDOW_SEARCH_FIT_END.",
    )

    parser.add_argument(
        "--val-start",
        default=None,
        help="Awal periode VALIDATION. Default: Config.WINDOW_SEARCH_VAL_START.",
    )

    parser.add_argument(
        "--val-end",
        default=None,
        help="Akhir periode VALIDATION. Default: Config.WINDOW_SEARCH_VAL_END.",
    )

    parser.add_argument(
        "--anchor-stride",
        type=int,
        default=None,
        help=(
            "Subsample anchor validation (1 dari N). Default: "
            "Config.WINDOW_SEARCH_ANCHOR_STRIDE."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Path CSV hasil lengkap. Default: Config.WINDOW_SEARCH_RESULTS_FILE.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    train_cache = args.train_cache or cfg.TEMPORAL_TRAIN_FILE

    model_names = (
        [m.strip() for m in args.models.split(",")]
        if args.models
        else list(cfg.MODEL_NAMES)
    )

    windows = (
        [int(w.strip()) for w in args.windows.split(",")]
        if args.windows
        else list(cfg.WINDOW_CANDIDATES)
    )

    fit_end = args.fit_end or cfg.WINDOW_SEARCH_FIT_END
    val_start = args.val_start or cfg.WINDOW_SEARCH_VAL_START
    val_end = args.val_end or cfg.WINDOW_SEARCH_VAL_END

    anchor_stride = (
        args.anchor_stride
        if args.anchor_stride is not None
        else cfg.WINDOW_SEARCH_ANCHOR_STRIDE
    )

    horizon_steps = cfg.HORIZON_STEPS

    output_path = args.output or cfg.WINDOW_SEARCH_RESULTS_FILE
    best_path = cfg.WINDOW_SEARCH_BEST_FILE

    banner("WINDOW SEARCH")

    say_info(f"TRAIN cache     : {train_cache}")
    say_info(f"Model           : {model_names}")
    say_info(f"Window kandidat : {windows}")
    say_info(f"FIT             : s/d {fit_end}")
    say_info(f"VALIDATION      : {val_start} s/d {val_end}")
    say_info(f"Anchor stride   : {anchor_stride} (validation rollout)")
    say_info(f"Horizon steps   : {horizon_steps}")
    say_info(
        f"Total kombinasi : {len(model_names)} model x {len(windows)} "
        f"window = {len(model_names) * len(windows)}"
    )

    hr()

    try:
        data_matrix, timeline, pixel_meta = load_temporal_cache(train_cache)

        say_info(f"TRAIN shape : {data_matrix.shape}")

        fit_mask, val_mask = chronological_split(
            timeline=timeline,
            train_end=fit_end,
            test_start=val_start,
            test_end=val_end,
        )

        fit_data, fit_timeline = slice_temporal_data(
            data_matrix, timeline, fit_mask
        )

        val_data, val_timeline = slice_temporal_data(
            data_matrix, timeline, val_mask
        )

        say_info(
            f"FIT        : {fit_timeline[0]} \u2192 {fit_timeline[-1]} "
            f"({len(fit_timeline)} timestep)"
        )

        say_info(
            f"VALIDATION : {val_timeline[0]} \u2192 {val_timeline[-1]} "
            f"({len(val_timeline)} timestep)"
        )

        hr()

        os.makedirs(cfg.WINDOW_SEARCH_DIR, exist_ok=True)

        results = []

        combo_idx = 0
        n_combos = len(model_names) * len(windows)

        for window in windows:

            gap()
            say_info(
                f"=== Window={window} : bangun dataset FIT & VALIDATION ==="
            )

            t0 = time.time()

            fit_df = build_window_dataset(
                fit_data, fit_timeline, pixel_meta, window=window
            )

            build_fit_elapsed = time.time() - t0

            if len(fit_df) == 0:
                say_error(
                    f"Window={window}: dataset FIT kosong (window "
                    f"terlalu besar utk periode FIT). Skip."
                )
                continue

            lag_cols = lag_column_names(window)
            X_fit = fit_df[lag_cols].values
            y_fit = fit_df[TARGET_COLUMN].values

            say_info(
                f"  FIT samples: {len(fit_df):,} "
                f"({build_fit_elapsed:.1f}s bangun fitur)"
            )

            t0 = time.time()

            val_windows, val_future, val_pixel_ids = build_rollout_arrays(
                val_data,
                val_timeline,
                pixel_meta,
                window=window,
                horizon_steps=horizon_steps,
                anchor_stride=anchor_stride,
            )

            build_val_elapsed = time.time() - t0

            say_info(
                f"  VALIDATION anchors: {len(val_windows):,} "
                f"({build_val_elapsed:.1f}s bangun rollout array)"
            )

            if len(val_windows) == 0:
                say_error(
                    f"Window={window}: tidak ada anchor VALIDATION valid "
                    f"(window+horizon terlalu panjang utk periode "
                    f"VALIDATION). Skip."
                )
                continue

            for model_name in model_names:
                combo_idx += 1

                say_info(
                    f"[{combo_idx}/{n_combos}] Training {model_name} "
                    f"(window={window})..."
                )

                model, train_elapsed = train_one_model(
                    model_name, X_fit, y_fit
                )

                say_info(
                    f"  Training selesai ({train_elapsed:.1f}s). "
                    f"Rollout recursive {horizon_steps} step..."
                )

                t0 = time.time()

                mae_per_step, _, r2_per_step = recursive_rollout_mae(
                    model, val_windows, val_future, horizon_steps
                )

                rollout_elapsed = time.time() - t0

                mae_avg = float(np.nanmean(mae_per_step))
                r2_avg = float(np.nanmean(r2_per_step))

                say_ok(
                    f"  {model_name} window={window}: "
                    f"MAE avg (18 step)={mae_avg:.4f}K "
                    f"R2 avg (18 step)={r2_avg:.4f} "
                    f"({rollout_elapsed:.1f}s rollout)"
                )

                row = {
                    "model": model_name,
                    "window": window,
                    "n_fit_samples": len(fit_df),
                    "n_val_anchors": len(val_windows),
                    "train_seconds": train_elapsed,
                    "mae_avg_all_steps": mae_avg,
                    "r2_avg_all_steps": r2_avg,
                }

                for step in range(horizon_steps):
                    row[f"mae_step_{step + 1}"] = float(mae_per_step[step])

                for step in range(horizon_steps):
                    row[f"r2_step_{step + 1}"] = float(r2_per_step[step])

                results.append(row)

        if not results:
            say_error("Tidak ada hasil sama sekali. Cek parameter FIT/VALIDATION.")
            raise SystemExit(1)

        results_df = pd.DataFrame(results)
        results_df.to_csv(output_path, index=False)

        best_df = (
            results_df
            .sort_values("mae_avg_all_steps")
            .groupby("model", as_index=False)
            .first()
            .sort_values("model")
        )

        best_df.to_csv(best_path, index=False)

        gap()
        banner("RINGKASAN")

        for _, row in best_df.iterrows():
            say_ok(
                f"{row['model']:<10} -> window terbaik = {int(row['window'])} "
                f"(MAE avg 18 step = {row['mae_avg_all_steps']:.4f}K, "
                f"R2 avg 18 step = {row['r2_avg_all_steps']:.4f})"
            )

        hr()

        say_ok("Tahap 4 (window search) selesai.")
        say_info(f"Hasil lengkap     : {output_path}")
        say_info(f"Window terpilih   : {best_path}")

    except Exception as exc:
        say_error(f"Window search gagal: {exc}")
        raise


if __name__ == "__main__":
    main()
