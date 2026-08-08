# ./scripts/04_recursive_evaluate.py

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
    p.add_argument(
        "--damping-factor", type=float, default=1.0,
        help=(
            "Redam delta prediksi model terhadap nilai terakhir window, geometris "
            "tiap step (damping_factor ** (step-1)). Default 1.0 = TIDAK ADA REDAMAN "
            "(perilaku lama). Fix untuk spatial_collapse_ratio yang naik >2x di step 18 "
            "(exposure bias -- lihat pipeline/recursive_eval.py::_apply_damping). Coba "
            "mulai dari 0.9 atau 0.8, jangan langsung kecil (mis. 0.5) -- terlalu kecil "
            "bikin forecast jangka panjang nyaris flat/tidak informatif. Harus di (0, 1]. "
            "Kalau < 1.0, output ditulis ke file BERBEDA (suffix _dampXX) supaya baseline "
            "tidak ketimpa."
        ),
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
    if args.damping_factor < 1.0:
        say_info(f"Damping   : AKTIF, damping_factor={args.damping_factor} (delta diredam {1 - args.damping_factor:.0%} tiap step)")
    else:
        say_info("Damping   : nonaktif (damping_factor=1.0, perilaku lama)")
    hr()

    try:
        detail_df, summary_df = run_all_models(
            dataset_csv_path=args.dataset,
            cache_path=args.cache,
            test_frac=args.test_frac,
            damping_factor=args.damping_factor,
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
        "Idealnya spatial_collapse_ratio tetap dekat 1 dan spatial_correlation tetap "
        "tinggi sampai step 18. Ratio mendekati 0 = model kolaps jadi rata/flat "
        "(masalah repo lama, CLAUDE.md §1). Ratio MENJAUH DARI 1 KE ATAS (>1, makin "
        "naik tiap step) = model kebalikannya: prediksi makin melebar dari nilai wajar "
        "(exposure bias/covariate shift saat rollout recursive) -- coba --damping-factor "
        "< 1.0 untuk meredam ini. spatial_correlation rendah di kedua kasus = pola "
        "spasial prediksi nggak match aktual walau variasinya ada/kurang."
    )


if __name__ == "__main__":
    main()