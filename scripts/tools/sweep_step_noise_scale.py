# ./scripts/tools/sweep_step_noise_scale.py
# Sweep beberapa scale factor pada step_noise_profile (magnitude noise
# PER KEDALAMAN ROLLOUT, lihat pipeline/model_training.py::
# build_step_noise_profile()/inject_recursive_style_noise()) -- lanjutan
# tuning setelah eksperimen scale=1.0 (profil MAE mentah apa adanya)
# terbukti signifikan membaikin spatial_collapse_ratio/spatial_correlation
# di step panjang (12-18) TAPI nge-regresi MAE step 1-4 (window "dangkal"
# jadi porsi besar training data yang di-noise berat, geser fit model).
#
# Pola SAMA PERSIS dgn sweep_noise_std.py -- noise itu parameter TRAINING
# (bukan post-hoc kayak damping), jadi WAJIB retrain 3 model per scale.
# Dataset besar + cache + anchor test set di-load SEKALI, dipakai ulang di
# semua nilai sweep. Tiap scale dapet subfolder model SENDIRI
# (Config.STEP_NOISE_SWEEP_MODELS_DIR/scale{XXX}/), TIDAK PERNAH menimpa
# model produksi.

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
from pipeline.model_training import (
    load_expanding_dataset, stratified_monthly_split, train_all_models, build_step_noise_profile,
)
from pipeline.recursive_eval import _map_anchors_to_indices, run_recursive_evaluation, summarize_by_step
from ui.terminal_display import banner, gap, hr, say_info, say_ok, say_error, make_progress_bar


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep scale factor step_noise_profile -- retrain 3 model per nilai, lalu recursive eval."
    )
    p.add_argument(
        "--scales", default="0.3,0.6,0.85",
        help="Daftar scale factor dipisah koma, urutan bebas (dinormalisasi urut naik). Default: 0.3,0.6,0.85.",
    )
    p.add_argument(
        "--profile-source", default=None,
        help="Path recursive_mae_summary.csv sumber profil (kurva MAE per step). Default: Config.RECURSIVE_EVAL_SUMMARY_FILE (produksi, damping_factor=1.0).",
    )
    p.add_argument("--dataset", default=None, help="Path CSV dataset expanding window. Default: Config.EXPANDING_DATASET_FILE.")
    p.add_argument("--cache", default=None, help="Path cache raw .npz. Default: Config.EXPANDING_RAW_CACHE_FILE.")
    p.add_argument("--test-frac", type=float, default=None, help="Fraksi test, HARUS SAMA dengan 03_train_models.py. Default: Config.TEST_FRAC.")
    p.add_argument("--sweep-models-dir", default=None, help="Folder dasar model per nilai sweep. Default: Config.STEP_NOISE_SWEEP_MODELS_DIR.")
    p.add_argument(
        "--damping-factor", type=float, default=1.0,
        help="damping_factor dipakai SAAT EVALUASI recursive (bukan training). Default: 1.0 (tanpa redaman, konfigurasi produksi -- CLAUDE.md §18.9).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    try:
        scales = sorted({round(float(v.strip()), 4) for v in args.scales.split(",") if v.strip()})
    except ValueError:
        say_error(f"--scales tidak valid: {args.scales!r} (harus angka dipisah koma, mis. '0.3,0.6,0.85')")
        return
    if any(s < 0 for s in scales):
        say_error("Semua scale harus >= 0.")
        return

    dataset_path = args.dataset or Config.EXPANDING_DATASET_FILE
    cache_path = args.cache or Config.EXPANDING_RAW_CACHE_FILE
    profile_source = args.profile_source or Config.RECURSIVE_EVAL_SUMMARY_FILE
    sweep_models_dir = args.sweep_models_dir or Config.STEP_NOISE_SWEEP_MODELS_DIR

    banner("SWEEP STEP-NOISE-PROFILE SCALE - CTT FORECASTING (expanding window)")
    say_info(f"Dataset         : {dataset_path}")
    say_info(f"Cache raw       : {cache_path}")
    say_info(f"Sumber profil   : {profile_source}")
    say_info(f"Nilai scale     : {', '.join(str(s) for s in scales)}")
    say_info(f"Model output    : {sweep_models_dir} (TERPISAH dari model produksi)")
    say_info(f"Damping (eval)  : {args.damping_factor}")
    hr()

    if not os.path.exists(dataset_path):
        say_error(f"Dataset tidak ditemukan: {dataset_path}. Jalankan 02_build_expanding_features.py dulu.")
        return
    if not os.path.exists(cache_path):
        say_error(
            f"Cache raw time series tidak ditemukan: {cache_path}\n"
            "Jalankan 02_build_expanding_features.py dulu (tanpa --no-cache)."
        )
        return
    if not os.path.exists(profile_source):
        say_error(f"Sumber profil tidak ditemukan: {profile_source}. Jalankan 04_recursive_evaluate.py dulu.")
        return

    base_profile = build_step_noise_profile(profile_source)
    say_info(f"Profil dasar (scale=1.0): step 1..{int(base_profile.index.max())}, "
             f"std {base_profile.min():.2f}K-{base_profile.max():.2f}K")

    # --- Load dataset (CSV besar) + cache + anchor test SEKALI -- semuanya
    # TIDAK bergantung ke scale, dipakai ulang di semua nilai sweep. ---
    say_info(f"Load dataset: {dataset_path} (bisa makan waktu, CSV besar)...")
    df = load_expanding_dataset(dataset_path)
    say_ok(f"{len(df):,} baris dataset dimuat.")

    say_info(f"Load cache raw time series: {cache_path}")
    data_matrix, timeline, pixel_meta = load_raw_cache(cache_path)

    _train_df, test_df, _cutoffs = stratified_monthly_split(df, test_frac=args.test_frac)
    anchors_df = test_df[["pixel_id", "anchor_t0"]].drop_duplicates().reset_index(drop=True)
    anchors_idx = _map_anchors_to_indices(anchors_df, timeline, pixel_meta)
    say_info(f"Jumlah anchor test unik: {len(anchors_idx)}")
    hr()

    os.makedirs(sweep_models_dir, exist_ok=True)
    os.makedirs(Config.STEP_NOISE_SWEEP_DIR, exist_ok=True)
    all_summaries = []
    all_train_summaries = []

    for scale in make_progress_bar(scales, desc="Sweep scale", unit="nilai"):
        suffix = f"_scale{scale:.2f}".replace(".", "")
        gap()
        say_info(f"=== scale={scale} ===")

        profile = base_profile * scale

        # --- Retrain 3 model dgn profile ini, ke subfolder SENDIRI ---
        value_models_dir = os.path.join(sweep_models_dir, suffix.lstrip("_"))
        train_summary = train_all_models(
            df, models_dir=value_models_dir, test_frac=args.test_frac,
            step_noise_profile=profile, cache_path=cache_path,
        )
        train_summary.insert(0, "step_noise_scale", scale)
        all_train_summaries.append(train_summary)

        # --- Recursive eval model yang BARU DILATIH ini (reuse cache+anchor
        # yang sudah di-load sebelum loop -- TIDAK reload per nilai). ---
        detail_frames = []
        for model_name in Config.MODEL_NAMES:
            model_path = os.path.join(value_models_dir, f"{model_name}.joblib")
            if not os.path.exists(model_path):
                say_info(f"PERINGATAN: model {model_name} tidak ditemukan di {model_path} (kemungkinan gagal training), dilewati.")
                continue
            model = joblib.load(model_path)
            detail_frames.append(run_recursive_evaluation(
                model_name, model, anchors_idx, data_matrix, timeline,
                damping_factor=args.damping_factor,
            ))
        if not detail_frames:
            say_error(f"Semua model gagal di-training utk scale={scale}, nilai ini dilewati dari perbandingan.")
            continue

        detail_df = pd.concat(detail_frames, ignore_index=True)
        summary_df = summarize_by_step(detail_df)
        summary_df.insert(0, "step_noise_scale", scale)
        all_summaries.append(summary_df)

        detail_path = os.path.join(Config.STEP_NOISE_SWEEP_DIR, f"recursive_evaluation{suffix}.csv")
        summary_path = os.path.join(Config.STEP_NOISE_SWEEP_DIR, f"recursive_mae_summary{suffix}.csv")
        detail_df.to_csv(detail_path, index=False)
        summary_df.drop(columns=["step_noise_scale"]).to_csv(summary_path, index=False)
        say_ok(f"scale={scale}: detail & summary tersimpan (suffix {suffix}).")

    if not all_summaries:
        say_error("Tidak ada nilai scale yang berhasil dievaluasi -- cek log di atas.")
        return

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(Config.STEP_NOISE_SWEEP_DIR, "step_noise_scale_sweep_comparison.csv")
    combined.to_csv(combined_path, index=False)

    combined_train = pd.concat(all_train_summaries, ignore_index=True)
    combined_train_path = os.path.join(sweep_models_dir, "step_noise_scale_sweep_training_summary.csv")
    combined_train.to_csv(combined_train_path, index=False)

    gap()
    banner("RINGKASAN SWEEP (step terakhir per scale, per model)")
    last_step = combined["step"].max()
    final_step_view = combined[combined["step"] == last_step].sort_values(
        ["model", "step_noise_scale"], ascending=[True, True]
    )
    for model_name in Config.MODEL_NAMES:
        sub = final_step_view[final_step_view["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} (step {last_step}) --")
        print(sub[["step_noise_scale", "mae", "rmse", "spatial_collapse_ratio", "spatial_correlation"]].to_string(index=False))

    hr()
    say_ok(f"Model per nilai sweep tersimpan di: {sweep_models_dir} (TIDAK menimpa model produksi)")
    say_ok(f"Detail & summary per nilai: {Config.STEP_NOISE_SWEEP_DIR} (suffix _scaleXXX)")
    say_ok(f"Perbandingan gabungan (semua nilai, semua step): {combined_path}")
    say_ok(f"Ringkasan training (waktu/MAE flat) per nilai: {combined_train_path}")


if __name__ == "__main__":
    main()
