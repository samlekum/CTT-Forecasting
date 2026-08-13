# ./scripts/tools/sweep_damping.py

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

import joblib
import pandas as pd

from pipeline.config import Config
from pipeline.dataset_builder import load_raw_cache
from pipeline.recursive_eval import (
    run_recursive_evaluation,
    select_test_anchors,
    summarize_by_step,
)
from pipeline.recursive_eval import _map_anchors_to_indices
from ui.terminal_display import banner, gap, hr, say_info, say_ok, say_error, make_progress_bar


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep beberapa --damping-factor sekaligus, load cache+model cuma sekali."
    )
    p.add_argument(
        "--factors", default="1.0,0.9,0.8,0.7,0.6,0.5",
        help="Daftar damping_factor dipisah koma, urutan bebas (dinormalisasi urut turun buat tampilan). Default: 1.0,0.9,0.8,0.7,0.6,0.5",
    )
    p.add_argument("--dataset", default=None, help="Path CSV dataset expanding window. Default: Config.EXPANDING_DATASET_FILE.")
    p.add_argument("--cache", default=None, help="Path cache raw .npz. Default: Config.EXPANDING_RAW_CACHE_FILE.")
    p.add_argument("--test-frac", type=float, default=None, help="Fraksi test, HARUS SAMA dengan 03_train_models.py. Default: Config.TEST_FRAC.")
    p.add_argument("--models-dir", default=None, help="Direktori model .joblib. Default: Config.EXPANDING_MODELS_DIR.")
    return p.parse_args()


def main():
    args = parse_args()

    try:
        factors = sorted({round(float(f.strip()), 4) for f in args.factors.split(",") if f.strip()}, reverse=True)
    except ValueError:
        say_error(f"--factors tidak valid: {args.factors!r} (harus angka dipisah koma, mis. '1.0,0.9,0.8')")
        return
    if any(f <= 0 or f > 1 for f in factors):
        say_error("Semua damping_factor harus di rentang (0, 1].")
        return

    dataset_csv_path = args.dataset or Config.EXPANDING_DATASET_FILE
    cache_path = args.cache or Config.EXPANDING_RAW_CACHE_FILE
    models_dir = args.models_dir or Config.EXPANDING_MODELS_DIR

    banner("SWEEP DAMPING FACTOR - CTT FORECASTING (expanding window)")
    say_info(f"Dataset      : {dataset_csv_path}")
    say_info(f"Cache raw    : {cache_path}")
    say_info(f"Faktor sweep : {', '.join(str(f) for f in factors)}")
    hr()

    if not os.path.exists(cache_path):
        say_error(
            f"Cache raw time series tidak ditemukan: {cache_path}\n"
            "Jalankan 02_build_expanding_features.py dulu (tanpa --no-cache)."
        )
        return

    # --- Load cache + anchors + model SEKALI, dipakai ulang tiap faktor ---
    say_info(f"Load cache raw time series: {cache_path}")
    data_matrix, timeline, pixel_meta = load_raw_cache(cache_path)

    say_info(f"Load test anchors dari: {dataset_csv_path}")
    anchors_df = select_test_anchors(dataset_csv_path, test_frac=args.test_frac)
    anchors_idx = _map_anchors_to_indices(anchors_df, timeline, pixel_meta)
    say_info(f"Jumlah anchor test unik: {len(anchors_idx)}")

    models = {}
    for model_name in Config.MODEL_NAMES:
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            say_info(f"PERINGATAN: model {model_name} tidak ditemukan di {model_path}, dilewati.")
            continue
        models[model_name] = joblib.load(model_path)

    if not models:
        say_error(f"Tidak ada model ditemukan di {models_dir}. Jalankan 03_train_models.py dulu.")
        return

    hr()

    # --- Loop tiap damping_factor, reuse cache+model yang sudah di-load ---
    # Output KHUSUS ke Config.DAMPING_SWEEP_DIR (evaluation/sweep_damping/)
    # -- TERPISAH dari Config.EXPANDING_EVAL_DIR (output RESMI Tahap 04),
    # supaya evaluation/ nggak kecampur belasan file per-faktor. TANPA
    # retensi/folder-per-run -- ditimpa tiap sweep dijalankan ulang.
    os.makedirs(Config.DAMPING_SWEEP_DIR, exist_ok=True)
    all_summaries = []

    for factor in make_progress_bar(factors, desc="Sweep damping", unit="faktor"):
        detail_frames = []
        for model_name, model in models.items():
            detail_frames.append(run_recursive_evaluation(
                model_name, model, anchors_idx, data_matrix, timeline, damping_factor=factor,
            ))
        detail_df = pd.concat(detail_frames, ignore_index=True)
        summary_df = summarize_by_step(detail_df)
        summary_df.insert(0, "damping_factor", factor)
        all_summaries.append(summary_df)

        # Simpan detail+summary per-faktor (suffix SELALU dipakai, termasuk
        # utk factor=1.0 -- beda dari versi lama yang nulis factor=1.0 ke
        # nama file TANPA suffix, karena sekarang sudah di folder terpisah
        # jadi tidak ada lagi risiko ketimpa file resmi Tahap 04).
        suffix = f"_damp{factor:.2f}".replace(".", "")
        detail_path = os.path.join(Config.DAMPING_SWEEP_DIR, f"recursive_evaluation{suffix}.csv")
        summary_path = os.path.join(Config.DAMPING_SWEEP_DIR, f"recursive_mae_summary{suffix}.csv")
        detail_df.to_csv(detail_path, index=False)
        summary_df.drop(columns=["damping_factor"]).to_csv(summary_path, index=False)

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(Config.DAMPING_SWEEP_DIR, "damping_sweep_comparison.csv")
    combined.to_csv(combined_path, index=False)

    gap()
    banner("RINGKASAN SWEEP (step terakhir per faktor, per model)")
    last_step = combined["step"].max()
    final_step_view = combined[combined["step"] == last_step].sort_values(
        ["model", "damping_factor"], ascending=[True, False]
    )
    for model_name in Config.MODEL_NAMES:
        sub = final_step_view[final_step_view["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} (step {last_step}) --")
        print(sub[["damping_factor", "mae", "rmse", "spatial_collapse_ratio", "spatial_correlation"]].to_string(index=False))

    hr()
    say_ok(f"Detail & summary per faktor disimpan ke: {Config.DAMPING_SWEEP_DIR} (suffix _dampXX)")
    say_ok(f"Perbandingan gabungan (semua faktor, semua step): {combined_path}")
    say_info(
        "Cara baca: cari damping_factor di mana spatial_collapse_ratio sudah "
        "dekat 1 & spatial_correlation cukup tinggi TAPI mae di step terakhir "
        "belum mulai naik lagi dibanding faktor yang lebih besar -- itu titik "
        "infleksi bias-variance (over-damping mulai bikin forecast nempel ke "
        "last_value / kehilangan sinyal tren). Kalau mae MASIH turun terus "
        "sampai faktor terkecil yang dicoba, coba tambah faktor lebih kecil lagi."
    )


if __name__ == "__main__":
    main()