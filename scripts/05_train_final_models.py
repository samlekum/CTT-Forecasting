# ./scripts/05_train_final_models.py
#
# Tahap 5: Training model final.
#
# Untuk setiap model (xgboost/lightgbm/catboost), pakai window TERBAIKNYA
# SENDIRI (hasil 04_search_window.py, Config.WINDOW_SEARCH_BEST_FILE),
# lalu latih SATU model single-step final di SELURUH periode TRAIN
# (Des'25-Mei'26 -- bukan cuma potongan FIT Des'25-Mar'26 yang dipakai
# window search).
#
# KENAPA re-train di seluruh TRAIN (bukan reuse model dari window search):
# model window search cuma dilatih di FIT (Des-Mar, ~4 bulan) supaya ada
# sisa VALIDATION (Apr-Mei) buat milih window secara adil. Begitu window
# per model sudah fix, model PRODUKSI selayaknya dilatih pakai data
# sebanyak mungkin yang tersedia (seluruh TRAIN, Des'25-Mei'26) -- window
# search dan training final adalah DUA model yang berbeda walau window-nya
# sama.
#
# TEST asli (Jun-Jul'26) TETAP TIDAK DISENTUH di tahap ini -- evaluasi
# baru dilakukan di 06_evaluate_test.py.
#
# INPUT:
#   dataset/temporal_split/train_temporal.npz
#   window_search/best_window_per_model.csv
#
# OUTPUT:
#   models_window/{model_name}.joblib   (satu file per model)
#   models_window/training_summary.csv
#   models_window/model_manifest.json   (window + feature_columns per model,
#                                         dibaca ulang oleh Tahap 6 & 7)
#
# CATATAN RUNTIME:
# Cuma len(models) model TERPISAH (default 3, BUKAN 3x18 seperti window
# search) -- tapi masing-masing dilatih di seluruh TRAIN (6 bulan, lebih
# banyak sample daripada FIT 4 bulan di window search). SELALU smoke-test
# dulu pakai --models yang dipersempit sebelum full run:
#
#   python scripts/05_train_final_models.py --models xgboost
#
# baru kalau itu jalan lancar & angkanya masuk akal, jalankan full run
# tanpa --models (pakai semua Config.MODEL_NAMES).

import argparse
import os
import time

import pandas as pd

from pipeline.config import load_config
from pipeline.temporal_dataset import load_temporal_cache
from pipeline.window_features import (
    build_window_dataset,
    TARGET_COLUMN,
    lag_column_names,
)
from pipeline.window_model_training import (
    train_one_model,
    save_model,
    model_path_for,
    save_manifest,
)
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
            "Training model final: satu model per Config.MODEL_NAMES, "
            "masing-masing pakai window terbaiknya sendiri, dilatih di "
            "seluruh periode TRAIN (TEST tidak disentuh)."
        )
    )

    parser.add_argument(
        "--train-cache",
        default=None,
        help="Path ke train_temporal.npz. Default: Config.TEMPORAL_TRAIN_FILE.",
    )

    parser.add_argument(
        "--best-window-file",
        default=None,
        help=(
            "Path ke CSV window terpilih per model (hasil 04_search_window.py). "
            "Default: Config.WINDOW_SEARCH_BEST_FILE."
        ),
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
        "--window-override",
        default=None,
        help=(
            "Override window per model, format 'xgboost=5,catboost=7' "
            "(model yang tidak disebut tetap pakai window dari "
            "--best-window-file). Dipakai buat eksperimen/debugging -- "
            "produksi normal TIDAK perlu ini."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Folder output model final. Default: Config.WINDOW_FINAL_MODELS_DIR.",
    )

    return parser.parse_args()


def _load_best_window_map(best_window_file):
    if not os.path.exists(best_window_file):
        raise FileNotFoundError(
            f"File window terpilih tidak ditemukan: {best_window_file}\n"
            "Jalankan 04_search_window.py dulu."
        )

    best_df = pd.read_csv(best_window_file)
    return dict(zip(best_df["model"], best_df["window"].astype(int)))


def _parse_window_override(raw):
    if not raw:
        return {}

    out = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        name, _, value = pair.partition("=")
        if not value:
            raise ValueError(
                f"Format --window-override tidak valid di bagian '{pair}' "
                "(harus 'model=window')."
            )
        out[name.strip()] = int(value.strip())

    return out


def main():
    args = parse_args()
    cfg = load_config()

    train_cache = args.train_cache or cfg.TEMPORAL_TRAIN_FILE
    best_window_file = args.best_window_file or cfg.WINDOW_SEARCH_BEST_FILE
    output_dir = args.output_dir or cfg.WINDOW_FINAL_MODELS_DIR

    model_names = (
        [m.strip() for m in args.models.split(",")]
        if args.models
        else list(cfg.MODEL_NAMES)
    )

    window_override = _parse_window_override(args.window_override)

    banner("TRAINING MODEL FINAL")

    say_info(f"TRAIN cache       : {train_cache}")
    say_info(f"Window terpilih   : {best_window_file}")
    say_info(f"Model             : {model_names}")
    say_info(f"Output            : {output_dir}")

    if window_override:
        say_info(f"Window override   : {window_override}")

    hr()

    try:
        best_window_map = _load_best_window_map(best_window_file)

        missing_models = [
            m for m in model_names
            if m not in best_window_map and m not in window_override
        ]

        if missing_models:
            raise ValueError(
                f"Model berikut tidak punya window terpilih di "
                f"{best_window_file}: {missing_models}. Jalankan "
                "04_search_window.py untuk model tersebut dulu, atau "
                "pakai --window-override."
            )

        data_matrix, timeline, pixel_meta = load_temporal_cache(train_cache)

        say_info(f"TRAIN shape : {data_matrix.shape}")
        say_info(f"TRAIN range : {timeline[0]} \u2192 {timeline[-1]}")

        hr()

        os.makedirs(output_dir, exist_ok=True)

        summary_rows = []
        manifest = {}

        for model_name in model_names:
            window = window_override.get(
                model_name, best_window_map.get(model_name)
            )

            gap()
            say_info(
                f"=== {model_name} (window={window}) : bangun dataset "
                f"TRAIN penuh ==="
            )

            t0 = time.time()

            train_df = build_window_dataset(
                data_matrix, timeline, pixel_meta, window=window
            )

            build_elapsed = time.time() - t0

            if len(train_df) == 0:
                say_error(
                    f"{model_name}: dataset TRAIN kosong untuk "
                    f"window={window}. Skip."
                )
                continue

            lag_cols = lag_column_names(window)
            X_train = train_df[lag_cols].values
            y_train = train_df[TARGET_COLUMN].values

            say_info(
                f"  TRAIN samples: {len(train_df):,} "
                f"({build_elapsed:.1f}s bangun fitur)"
            )

            say_info(f"  Training {model_name}...")

            model, train_elapsed = train_one_model(
                model_name, X_train, y_train
            )

            say_ok(f"  Training selesai ({train_elapsed:.1f}s).")

            model_path = model_path_for(model_name, output_dir)
            save_model(model, model_path)

            say_ok(f"  Model disimpan: {model_path}")

            summary_rows.append({
                "model": model_name,
                "window": window,
                "n_train_samples": len(train_df),
                "train_seconds": train_elapsed,
                "build_features_seconds": build_elapsed,
                "model_path": model_path,
            })

            manifest[model_name] = {
                "window": window,
                "model_path": model_path,
                "feature_columns": lag_cols,
                "target_column": TARGET_COLUMN,
                "n_train_samples": int(len(train_df)),
                "train_seconds": train_elapsed,
                "train_start": str(timeline[0]),
                "train_end": str(timeline[-1]),
            }

        if not summary_rows:
            say_error("Tidak ada model berhasil dilatih.")
            raise SystemExit(1)

        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(output_dir, "training_summary.csv")
        summary_df.to_csv(summary_path, index=False)

        manifest_path = os.path.join(output_dir, "model_manifest.json")
        save_manifest(manifest_path, manifest)

        gap()
        banner("RINGKASAN")

        for _, row in summary_df.iterrows():
            say_ok(
                f"{row['model']:<10} window={int(row['window']):<2} "
                f"n_train={int(row['n_train_samples']):,} "
                f"({row['train_seconds']:.1f}s)"
            )

        hr()

        say_ok("Tahap 5 (training model final) selesai.")
        say_info(f"Model        : {output_dir}/*.joblib")
        say_info(f"Ringkasan    : {summary_path}")
        say_info(f"Manifest     : {manifest_path}")

    except Exception as exc:
        say_error(f"Training model final gagal: {exc}")
        raise


if __name__ == "__main__":
    main()
