# ./scripts/09_kfold_statistical_analysis.py
#
# Tahap 9 (analisis tambahan, TIDAK menggantikan Tahap 4-6): blocked
# k-fold cross validation (default k=5) di periode TRAIN, per model
# memakai window terbaiknya sendiri (dari Config.WINDOW_SEARCH_BEST_FILE,
# hasil 04_search_window.py) -- untuk mengukur STABILITAS performa lintas
# potongan waktu berbeda (mean +/- std, 95% CI) dan uji signifikansi
# statistik antar model (paired t-test + Wilcoxon), plus waktu training &
# peak memory per model per fold.
#
# Kenapa terpisah dari 04_search_window.py:
# 04 cuma memakai SATU split FIT/VALIDATION untuk memilih window terbaik
# (murni untuk seleksi hyperparameter, cepat). Modul ini dipakai SETELAH
# window terbaik per model sudah ditentukan, untuk menjawab pertanyaan
# yang beda: "seberapa YAKIN kita model A memang lebih baik dari model B,
# bukan cuma kebetulan menang di satu split itu?"
#
# INPUT:
#   dataset/temporal_split/train_temporal.npz
#   window_search/best_window_per_model.csv (opsional, kalau tidak ada
#   pakai Config.WINDOW_CANDIDATES[len//2] sebagai fallback per model --
#   lihat --window)
#
# OUTPUT:
#   window_search/kfold_detail.csv     (satu baris per model x fold)
#   window_search/kfold_summary.csv    (satu baris per model, agregat)
#   window_search/kfold_significance.csv (satu baris per pasangan model)

import argparse
import os

import pandas as pd

from pipeline.config import load_config
from pipeline.temporal_dataset import load_temporal_cache
from pipeline.kfold_eval import (
    run_kfold_statistical_analysis,
    summarize_kfold,
    pairwise_significance,
)
from ui.terminal_display import banner, gap, hr, say_error, say_info, say_ok


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Tahap 9: blocked k-fold CV + timing/memory per model "
            "(analisis statistik tambahan, di atas periode TRAIN)."
        )
    )
    parser.add_argument("--train-cache", default=None)
    parser.add_argument("--models", default=None, help="Comma-separated. Default: semua Config.MODEL_NAMES.")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help=(
            "Window TUNGGAL dipakai untuk SEMUA model (override). Default: "
            "baca per-model dari Config.WINDOW_SEARCH_BEST_FILE; kalau file "
            "itu tidak ada, pakai nilai tengah Config.WINDOW_CANDIDATES."
        ),
    )
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--output-dir", default=None, help="Default: Config.WINDOW_SEARCH_DIR.")
    return parser.parse_args()


def resolve_window_per_model(args, cfg, model_names):
    if args.window is not None:
        return {m: args.window for m in model_names}

    if os.path.exists(cfg.WINDOW_SEARCH_BEST_FILE):
        best_df = pd.read_csv(cfg.WINDOW_SEARCH_BEST_FILE)
        mapping = dict(zip(best_df["model"], best_df["window"]))
        missing = [m for m in model_names if m not in mapping]
        if not missing:
            return {m: int(mapping[m]) for m in model_names}
        say_error(
            f"Model tanpa window terpilih di {cfg.WINDOW_SEARCH_BEST_FILE}: "
            f"{missing}. Jalankan 04_search_window.py dulu, atau pakai --window."
        )

    fallback = cfg.WINDOW_CANDIDATES[len(cfg.WINDOW_CANDIDATES) // 2]
    say_info(
        f"Window best-per-model tidak tersedia -- fallback window={fallback} "
        "untuk semua model (jalankan 04_search_window.py untuk hasil akurat)."
    )
    return {m: fallback for m in model_names}


def main():
    args = parse_args()
    cfg = load_config()

    train_cache = args.train_cache or cfg.TEMPORAL_TRAIN_FILE
    model_names = (
        [m.strip() for m in args.models.split(",")]
        if args.models
        else list(cfg.MODEL_NAMES)
    )
    output_dir = args.output_dir or cfg.WINDOW_SEARCH_DIR
    horizon_steps = cfg.HORIZON_STEPS

    banner(f"TAHAP 9: {args.n_folds}-FOLD STATISTICAL ANALYSIS")
    say_info(f"TRAIN cache : {train_cache}")
    say_info(f"Model       : {model_names}")
    say_info(f"n_folds     : {args.n_folds}")
    hr()

    try:
        data_matrix, timeline, pixel_meta = load_temporal_cache(train_cache)
        window_per_model = resolve_window_per_model(args, cfg, model_names)
        say_info(f"Window per model : {window_per_model}")
        hr()

        all_details = []
        for model_name in model_names:
            window = window_per_model[model_name]
            gap()
            say_info(f"=== {model_name} (window={window}) ===")

            detail_df = run_kfold_statistical_analysis(
                data_matrix,
                timeline,
                pixel_meta,
                model_names=[model_name],
                window=window,
                horizon_steps=horizon_steps,
                n_folds=args.n_folds,
                freq_minutes=cfg.FREQ_MINUTES,
            )
            if len(detail_df) == 0:
                say_error(f"  {model_name}: tidak ada fold valid, di-skip.")
                continue

            for _, row in detail_df.iterrows():
                say_ok(
                    f"  fold {int(row['fold'])}: MAE={row['mae_avg_all_steps']:.3f}K "
                    f"R2={row['r2_avg_all_steps']:.3f} "
                    f"train={row['train_seconds']:.1f}s "
                    f"peak_mem={row['train_peak_memory_mb']:.1f}MB"
                )
            all_details.append(detail_df)

        if not all_details:
            say_error("Tidak ada hasil sama sekali.")
            raise SystemExit(1)

        kfold_df = pd.concat(all_details, ignore_index=True)
        summary_df = summarize_kfold(kfold_df)
        sig_df = pairwise_significance(kfold_df)

        os.makedirs(output_dir, exist_ok=True)
        detail_path = os.path.join(output_dir, "kfold_detail.csv")
        summary_path = os.path.join(output_dir, "kfold_summary.csv")
        sig_path = os.path.join(output_dir, "kfold_significance.csv")

        kfold_df.to_csv(detail_path, index=False)
        summary_df.to_csv(summary_path, index=False)
        sig_df.to_csv(sig_path, index=False)

        gap()
        banner("RINGKASAN (mean +/- std lintas fold)")
        for _, row in summary_df.iterrows():
            say_ok(
                f"{row['model']:<10} "
                f"MAE={row['mae_mean']:.3f}+/-{row['mae_std']:.3f}K "
                f"(CV={row['mae_cv_percent']:.1f}%) "
                f"R2={row['r2_mean']:.3f} "
                f"train={row['train_seconds_mean']:.1f}s "
                f"mem={row['train_peak_memory_mb_mean']:.1f}MB"
            )

        if len(sig_df) > 0:
            gap()
            banner("SIGNIFIKANSI ANTAR MODEL (paired, metric=MAE)")
            for _, row in sig_df.iterrows():
                verdict = "SIGNIFIKAN" if row["significant_5pct"] else "tidak signifikan"
                say_info(
                    f"{row['model_a']} vs {row['model_b']}: "
                    f"diff={row['mean_diff']:+.3f}K "
                    f"p(t-test)={row['p_value_ttest']:.4f} "
                    f"p(wilcoxon)={row['p_value_wilcoxon']:.4f} "
                    f"({verdict} @5%)"
                )

        hr()
        say_ok("Tahap 9 selesai.")
        say_info(f"Detail       : {detail_path}")
        say_info(f"Summary      : {summary_path}")
        say_info(f"Significance : {sig_path}")

    except Exception as exc:
        say_error(f"Tahap 9 gagal: {exc}")
        raise


if __name__ == "__main__":
    main()