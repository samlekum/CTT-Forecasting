# ./scripts/06_evaluate_test.py

import argparse
import os
import time

import numpy as np
import pandas as pd

from pipeline.config import load_config
from pipeline.temporal_dataset import load_temporal_cache
from pipeline.window_features import build_rollout_arrays_with_anchors
from pipeline.window_model_training import (
    load_manifest,
    load_model,
    model_path_for,
    recursive_rollout_predict,
)
from pipeline.window_eval import build_detail_df, summarize_by_step
from ui.terminal_display import (
    banner,
    gap,
    hr,
    say_error,
    say_info,
    say_ok,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluasi final model window di TEST asli (Jun-Jul'26), "
            "anchor_stride=1 (semua anchor), plus MAE/RMSE/R2 dan metrik "
            "spatial collapse/correlation."
        )
    )

    parser.add_argument(
        "--test-cache",
        default=None,
        help="Path ke test_temporal.npz. Default: Config.TEMPORAL_TEST_FILE.",
    )

    parser.add_argument(
        "--manifest-file",
        default=None,
        help=(
            "Path ke model_manifest.json (hasil 05_train_final_models.py). "
            "Default: Config.WINDOW_FINAL_MANIFEST_FILE."
        ),
    )

    parser.add_argument(
        "--models",
        default=None,
        help=(
            "Comma-separated model names yang dievaluasi. "
            "Default: semua model di manifest."
        ),
    )

    parser.add_argument(
        "--anchor-stride",
        type=int,
        default=1,
        help=(
            "Stride anchor TEST. Default 1 (SEMUA anchor -- laporan "
            "final). Naikkan (mis. 6) cuma untuk smoke-test cepat."
        ),
    )

    parser.add_argument(
        "--max-anchors",
        type=int,
        default=None,
        help=(
            "Batasi jumlah anchor_t0 UNIK per model (dipotong setelah "
            "anchor_stride, tetap menyertakan SEMUA pixel di tiap "
            "anchor_t0 yang lolos) -- buat smoke-test cepat sebelum full "
            "run. Dipotong per anchor_t0 unik (bukan per baris mentah) "
            "supaya metrik spatial_collapse_ratio/spatial_correlation "
            "tetap bisa dihitung (butuh >=2 pixel per anchor_t0) alih-alih "
            "NaN. Default: tanpa batas."
        ),
    )

    parser.add_argument(
        "--detail-file",
        default=None,
        help="Path output detail CSV. Default: Config.WINDOW_TEST_EVAL_DETAIL_FILE.",
    )

    parser.add_argument(
        "--summary-file",
        default=None,
        help="Path output summary CSV. Default: Config.WINDOW_TEST_EVAL_SUMMARY_FILE.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    test_cache = args.test_cache or cfg.TEMPORAL_TEST_FILE
    manifest_file = args.manifest_file or cfg.WINDOW_FINAL_MANIFEST_FILE
    detail_path = args.detail_file or cfg.WINDOW_TEST_EVAL_DETAIL_FILE
    summary_path = args.summary_file or cfg.WINDOW_TEST_EVAL_SUMMARY_FILE

    banner("EVALUASI FINAL DI TEST (Jun-Jul'26)")

    say_info(f"TEST cache        : {test_cache}")
    say_info(f"Manifest          : {manifest_file}")
    say_info(f"Anchor stride     : {args.anchor_stride}")

    if args.max_anchors is not None:
        say_info(f"Max anchor/model  : {args.max_anchors} (SMOKE-TEST MODE)")

    hr()

    try:
        if not os.path.exists(manifest_file):
            raise FileNotFoundError(
                f"Manifest tidak ditemukan: {manifest_file}\n"
                "Jalankan 05_train_final_models.py dulu."
            )

        manifest = load_manifest(manifest_file)

        model_names = (
            [m.strip() for m in args.models.split(",")]
            if args.models
            else list(manifest.keys())
        )

        missing_models = [m for m in model_names if m not in manifest]
        if missing_models:
            raise ValueError(
                f"Model berikut tidak ada di manifest: {missing_models}. "
                f"Tersedia: {list(manifest.keys())}"
            )

        say_info(f"Model             : {model_names}")

        # models_dir DIREKONSTRUKSI dari lokasi manifest_file itu sendiri
        # (bukan dipercaya dari entry["model_path"] di JSON) -- model_path
        # yang tersimpan di manifest adalah path ABSOLUT hasil
        # os.path.join(PROJECT_ROOT, ...) SAAT 05_train_final_models.py
        # dijalankan. Kalau folder project di-rename/dipindah setelah itu
        # (mis. CTT-Forecasting-Expanding -> CTT-Forecasting), path lama
        # itu jadi basi walau file .joblib-nya sendiri masih ada persis di
        # sebelah model_manifest.json. Rekonstruksi dari manifest_file
        # membuat resolusi path tahan terhadap folder rename/pindah.
        models_dir = os.path.dirname(os.path.abspath(manifest_file))

        data_matrix, timeline, pixel_meta = load_temporal_cache(test_cache)

        say_info(f"TEST shape  : {data_matrix.shape}")
        say_info(f"TEST range  : {timeline[0]} \u2192 {timeline[-1]}")

        hr()

        horizon_steps = cfg.HORIZON_STEPS

        detail_frames = []
        timing_rows = []

        for model_name in model_names:
            entry = manifest[model_name]
            window = entry["window"]
            model_path = model_path_for(model_name, models_dir)

            gap()
            say_info(f"=== {model_name} (window={window}) ===")

            if not os.path.exists(model_path):
                say_error(
                    f"  Model tidak ditemukan: {model_path}. Skip {model_name}."
                )
                continue

            model = load_model(model_path)

            t0 = time.time()

            windows, true_future, pixel_ids, anchor_time = (
                build_rollout_arrays_with_anchors(
                    data_matrix,
                    timeline,
                    pixel_meta,
                    window=window,
                    horizon_steps=horizon_steps,
                    anchor_stride=args.anchor_stride,
                )
            )

            if args.max_anchors is not None:
                unique_anchor_t0 = pd.unique(anchor_time)
                if len(unique_anchor_t0) > args.max_anchors:
                    # np.unique/pd.unique tidak menjamin urutan waktu,
                    # urutkan dulu supaya smoke-test konsisten ambil
                    # anchor_t0 PALING AWAL (bukan acak tergantung urutan
                    # pixel loop di build_rollout_arrays_with_anchors).
                    keep_t0 = np.sort(unique_anchor_t0)[: args.max_anchors]
                    keep_mask = np.isin(anchor_time, keep_t0)

                    windows = windows[keep_mask]
                    true_future = true_future[keep_mask]
                    pixel_ids = pixel_ids[keep_mask]
                    anchor_time = anchor_time[keep_mask]

            build_elapsed = time.time() - t0

            n_anchor = len(windows)

            say_info(
                f"  Anchor TEST valid : {n_anchor:,} "
                f"({build_elapsed:.1f}s bangun array)"
            )

            if n_anchor == 0:
                say_error(
                    f"  {model_name}: tidak ada anchor TEST valid untuk "
                    f"window={window}. Skip."
                )
                continue

            t1 = time.time()

            # damping_rate/cap dibaca dari manifest (Tahap 5) -- default 0.0
            # untuk manifest lama yang belum punya field ini (getattr-style
            # via .get supaya backward-compatible).
            damping_rate = entry.get("damping_rate", 0.0)
            damping_cap = entry.get("damping_cap", 0.6)

            predictions = recursive_rollout_predict(
                model,
                windows,
                horizon_steps=horizon_steps,
                anchor_time=anchor_time,
                damping_rate=damping_rate,
                damping_cap=damping_cap,
            )

            predict_elapsed = time.time() - t1

            say_ok(
                f"  Recursive rollout selesai ({predict_elapsed:.1f}s, "
                f"{horizon_steps} step)."
            )

            model_detail_df = build_detail_df(
                model_name,
                predictions,
                true_future,
                pixel_ids,
                anchor_time,
            )

            detail_frames.append(model_detail_df)

            timing_rows.append({
                "model": model_name,
                "window": window,
                "n_anchor": n_anchor,
                "build_arrays_seconds": build_elapsed,
                "predict_seconds": predict_elapsed,
            })

        if not detail_frames:
            say_error("Tidak ada model berhasil dievaluasi.")
            raise SystemExit(1)

        detail_df = pd.concat(detail_frames, ignore_index=True)
        summary_df = summarize_by_step(detail_df)

        os.makedirs(os.path.dirname(detail_path), exist_ok=True)
        detail_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)

        gap()
        banner("RINGKASAN (rata-rata seluruh step)")

        for model_name in model_names:
            model_summary = summary_df[summary_df["model"] == model_name]
            if len(model_summary) == 0:
                continue
            say_ok(
                f"{model_name:<10} "
                f"MAE={model_summary['mae'].mean():.3f} "
                f"R2={model_summary['r2'].mean():.3f} "
                f"collapse_ratio={model_summary['spatial_collapse_ratio'].mean():.3f} "
                f"correlation={model_summary['spatial_correlation'].mean():.3f}"
            )

        hr()

        say_ok("Tahap 6 (evaluasi TEST final) selesai.")
        say_info(f"Detail   : {detail_path}")
        say_info(f"Summary  : {summary_path}")

        if args.max_anchors is not None:
            say_info(
                "CATATAN: ini SMOKE-TEST (--max-anchors dipakai). "
                "Jalankan ulang tanpa --max-anchors untuk hasil final."
            )

    except Exception as exc:
        say_error(f"Evaluasi TEST gagal: {exc}")
        raise


if __name__ == "__main__":
    main()