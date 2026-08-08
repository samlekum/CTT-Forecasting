# ./scripts/04_recursive_evaluate.py
# Tahap 5: evaluasi RECURSIVE model hasil 03_train_model.py -- window
# di-extend pakai prediksi model sendiri mulai step 2 (bukan observasi real
# seperti evaluate() flat di model_training.py), lalu MAE dihitung per step
# 1..18 terpisah. Ini yang jawab pertanyaan inti kenapa project ini dibuat:
# apakah expanding window + closed-form features berhasil menghindari
# compounding error / spatial collapse yang jadi masalah di repo lama.
# Lihat CLAUDE.md §7 dan pipeline/recursive_eval.py untuk detail metode.

import argparse

from ui.terminal_display import hr, gap, banner, say_info, say_error
from pipeline.config import load_config
from pipeline.recursive_eval import run_all_models


def parse_args():
    p = argparse.ArgumentParser(description="Evaluasi recursive MAE per step (CTT forecasting, expanding window).")
    p.add_argument(
        "--dataset", default=None,
        help="Path ke CSV dataset expanding window (buat nentuin anchor test set). Default: Config.EXPANDING_DATASET_FILE.",
    )
    p.add_argument(
        "--cache", default=None,
        help="Path ke cache raw time series (.npz). Default: Config.EXPANDING_RAW_CACHE_FILE.",
    )
    p.add_argument(
        "--test-frac", type=float, default=None,
        help="Fraksi test per-bulan, HARUS SAMA dengan yang dipakai 03_train_model.py biar anchor test konsisten. Default: Config.TEST_FRAC (0.15).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    banner("EVALUASI RECURSIVE - CTT FORECASTING (expanding window)")
    say_info(f"Dataset   : {args.dataset or cfg.EXPANDING_DATASET_FILE}")
    say_info(f"Cache raw : {args.cache or cfg.EXPANDING_RAW_CACHE_FILE}")
    say_info(f"Model     : {', '.join(cfg.MODEL_NAMES)}")
    say_info(f"Horizon   : step 1..{cfg.HORIZON_STEPS} ({cfg.HORIZON_STEPS * cfg.FREQ_MINUTES} menit)")
    hr()

    try:
        detail_df, summary_df = run_all_models(
            dataset_csv_path=args.dataset,
            cache_path=args.cache,
            test_frac=args.test_frac,
        )
    except FileNotFoundError as e:
        say_error(str(e))
        return

    gap()
    banner("RINGKASAN MAE PER STEP")
    for model_name in cfg.MODEL_NAMES:
        sub = summary_df[summary_df["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} --")
        print(sub[["step", "mae", "rmse", "spatial_collapse_ratio", "spatial_correlation", "n_t0_groups"]].to_string(index=False))

    hr()
    say_info(
        "spatial_collapse_ratio mendekati 0 seiring step naik = model kehilangan "
        "variasi spasial antar pixel (masalah utama repo lama, CLAUDE.md §1). "
        "spatial_correlation rendah = pola spasial prediksi nggak match sama aktual "
        "walau variasinya ada. Idealnya kedua metrik ini tetap tinggi (dekat 1) "
        "sampai step 18, bukan cuma MAE yang rendah."
    )


if __name__ == "__main__":
    main()