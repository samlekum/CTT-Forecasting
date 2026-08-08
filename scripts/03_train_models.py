# ./scripts/03_train_model.py
# Tahap 3: latih model (XGBoost/LightGBM/CatBoost) pada dataset expanding
# window hasil 02_build_expanding_features.py. Split stratified per-bulan
# (CLAUDE.md §6), evaluasi dasar keseluruhan (bukan per-step -- itu di
# 04_recursive_evaluate.py).

import argparse
import os

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_expanding_dataset, train_all_models


def parse_args():
    p = argparse.ArgumentParser(description="Training model CTT forecasting (expanding window).")
    p.add_argument(
        "--dataset", default=None,
        help="Path ke CSV dataset expanding window. Default: Config.EXPANDING_DATASET_FILE.",
    )
    p.add_argument(
        "--test-frac", type=float, default=None,
        help="Fraksi test per-bulan untuk stratified_monthly_split(). Default: Config.TEST_FRAC (0.15).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    dataset_path = args.dataset or cfg.EXPANDING_DATASET_FILE
    models_dir = cfg.EXPANDING_MODELS_DIR

    banner("TRAINING MODEL - CTT FORECASTING (expanding window)")
    say_info(f"Dataset      : {dataset_path}")
    say_info(f"Folder model : {models_dir}")
    say_info(f"Model        : {', '.join(cfg.MODEL_NAMES)}")
    hr()

    if not os.path.exists(dataset_path):
        say_error(f"Dataset tidak ditemukan: {dataset_path}. Jalankan 02_build_expanding_features.py dulu.")
        return

    df = load_expanding_dataset(dataset_path)
    say_info(f"Total baris dataset: {len(df)}")

    summary_df = train_all_models(df, models_dir=models_dir, test_frac=args.test_frac)

    gap()
    banner("RINGKASAN")
    print(summary_df.to_string(index=False))

    for _, row in summary_df.iterrows():
        if row.get("error"):
            say_error(f"{row['model']:10s} | GAGAL: {row['error']}")
        else:
            say_ok(
                f"{row['model']:10s} | {row['waktu_training_detik']:6.1f}s | "
                f"MAE={row['mae']:.4f}K  RMSE={row['rmse']:.4f}K  R2={row['r2']:.4f}"
            )

    hr()
    say_info("Lanjut ke Tahap 4: 04_recursive_evaluate.py (evaluasi MAE per step 1-18).")


if __name__ == "__main__":
    main()