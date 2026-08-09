# ./scripts/03_train_models.py
# Tahap 3: latih model (XGBoost/LightGBM/CatBoost) pada dataset expanding
# window hasil 02_build_expanding_features.py. Split stratified per-bulan
# (CLAUDE.md §6), evaluasi dasar keseluruhan (bukan per-step -- itu di
# 04_recursive_evaluate.py).

import argparse
import os

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.model_training import load_expanding_dataset, train_all_models, build_step_noise_profile


def parse_args():
    p = argparse.ArgumentParser(description="Training model CTT forecasting (expanding window).")
    p.add_argument(
        "--dataset", default=None,
        help="Path ke CSV dataset expanding window. Default: Config.EXPANDING_DATASET_FILE.",
    )
    p.add_argument(
        "--models-dir", default=None,
        help="Folder output model + training_summary.csv. Default: Config.EXPANDING_MODELS_DIR (models/, produksi).",
    )
    p.add_argument(
        "--test-frac", type=float, default=None,
        help="Fraksi test per-bulan untuk stratified_monthly_split(). Default: Config.TEST_FRAC (0.15).",
    )
    p.add_argument(
        "--noise-std", type=float, default=0.0,
        help=(
            "Std deviasi noise Gaussian (Kelvin) yang disuntik ke fitur window "
            "X_train (bukan X_test) untuk anchor step>1, sebelum training -- "
            "redesain dari inject_lag_noise() repo lama untuk skema fitur "
            "closed-form project ini (lihat pipeline/model_training.py::"
            "inject_recursive_style_noise() untuk desain lengkap). Default 0 = "
            "tanpa noise (perilaku lama, backward-compatible). DIABAIKAN kalau "
            "--noise-step-profile diisi. Mulai coba dari ~MAE step-1 aktual "
            "(lihat recursive_mae_summary.csv, ~2.5K), WAJIB divalidasi lewat "
            "scripts/tools/sweep_noise_std.py."
        ),
    )
    p.add_argument(
        "--noise-step-profile", default=None,
        help=(
            "Path ke recursive_mae_summary.csv (biasanya hasil model produksi saat "
            "ini) -- kalau diisi, magnitude noise jadi PER KEDALAMAN ROLLOUT "
            "(mengikuti kurva MAE aktual per step, bukan --noise-std konstan). "
            "Lihat pipeline/model_training.py::build_step_noise_profile() untuk "
            "alasan lengkap. --noise-std DIABAIKAN kalau ini diisi."
        ),
    )
    p.add_argument(
        "--noise-step-profile-scale", type=float, default=1.0,
        help=(
            "Faktor pengali profil per-step SEBELUM dipakai (cuma berlaku kalau "
            "--noise-step-profile diisi). Default 1.0 = pakai kurva MAE mentah "
            "apa adanya. Turunkan (mis. 0.5) kalau profil mentah (scale=1.0) "
            "terbukti kelewat agresif di step awal."
        ),
    )
    p.add_argument(
        "--cache", default=None,
        help="Path cache raw .npz, dibutuhkan kalau --noise-std > 0 atau --noise-step-profile diisi. Default: Config.EXPANDING_RAW_CACHE_FILE.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    dataset_path = args.dataset or cfg.EXPANDING_DATASET_FILE
    models_dir = args.models_dir or cfg.EXPANDING_MODELS_DIR

    banner("TRAINING MODEL - CTT FORECASTING (expanding window)")
    say_info(f"Dataset      : {dataset_path}")
    say_info(f"Folder model : {models_dir}")
    say_info(f"Model        : {', '.join(cfg.MODEL_NAMES)}")
    if args.noise_step_profile:
        say_info(f"Noise inject : AKTIF, PROFIL PER-STEP dari {args.noise_step_profile} (scale={args.noise_step_profile_scale})")
    elif args.noise_std > 0:
        say_info(f"Noise inject : AKTIF, std={args.noise_std}K KONSTAN pada fitur window X_train (step>1)")
    else:
        say_info("Noise inject : nonaktif (--noise-std=0, perilaku lama)")
    hr()

    if not os.path.exists(dataset_path):
        say_error(f"Dataset tidak ditemukan: {dataset_path}. Jalankan 02_build_expanding_features.py dulu.")
        return

    cache_path = args.cache or cfg.EXPANDING_RAW_CACHE_FILE
    needs_cache = args.noise_std > 0 or args.noise_step_profile
    if needs_cache and not os.path.exists(cache_path):
        say_error(
            f"Noise injection butuh cache raw, tapi tidak ditemukan: {cache_path}\n"
            "Jalankan 02_build_expanding_features.py dulu (tanpa --no-cache)."
        )
        return

    step_noise_profile = None
    if args.noise_step_profile:
        if not os.path.exists(args.noise_step_profile):
            say_error(f"--noise-step-profile tidak ditemukan: {args.noise_step_profile}")
            return
        step_noise_profile = build_step_noise_profile(args.noise_step_profile, scale=args.noise_step_profile_scale)
        say_info(
            f"Profil noise: step 1..{int(step_noise_profile.index.max())}, "
            f"std {step_noise_profile.min():.2f}K-{step_noise_profile.max():.2f}K"
        )

    df = load_expanding_dataset(dataset_path)
    say_info(f"Total baris dataset: {len(df)}")

    summary_df = train_all_models(
        df, models_dir=models_dir, test_frac=args.test_frac,
        noise_std=args.noise_std, cache_path=cache_path,
        step_noise_profile=step_noise_profile,
    )

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