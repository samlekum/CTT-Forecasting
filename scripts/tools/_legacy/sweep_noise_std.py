# ./scripts/tools/sweep_noise_std.py
# Sweep beberapa nilai --noise-std sekaligus buat cari titik optimal
# inject_recursive_style_noise() (CLAUDE.md §17 poin 7) -- BEDA dari
# sweep_damping.py yang cuma post-processing rollout (reuse model yang
# sudah dilatih), noise_std itu parameter TRAINING, jadi tool ini WAJIB
# retrain 3 model per nilai yang di-sweep -- jauh lebih mahal (~menit per
# nilai, bukan detik).
#
# Desain: load dataset besar (CSV) + cache raw + anchor test set SEKALI
# (semuanya TIDAK bergantung ke noise_std -- split & anchor test
# ditentukan murni dari test_frac, bukan dari noise), dipakai ulang di
# semua nilai sweep. Tiap nilai noise_std dapet subfolder model SENDIRI
# (Config.NOISE_SWEEP_MODELS_DIR/noise{XXX}/) -- TIDAK PERNAH menimpa
# model produksi di Config.EXPANDING_MODELS_DIR.
#
# Evaluasi recursive per nilai pakai damping_factor=1.0 default (konsisten
# sama keputusan final CLAUDE.md §18.9 -- damping_factor=0.9 sudah usang
# buat use-case horizon panjang), TAPI bisa di-override --damping-factor
# kalau mau eksperimen kombinasi noise x damping.

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
from pipeline.model_training import load_expanding_dataset, stratified_monthly_split, train_all_models
from pipeline.recursive_eval import _map_anchors_to_indices, run_recursive_evaluation, summarize_by_step
from ui.terminal_display import banner, gap, hr, say_info, say_ok, say_error, make_progress_bar


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep beberapa --noise-std sekaligus -- retrain 3 model per nilai, lalu recursive eval."
    )
    p.add_argument(
        "--values", default="0.0,1.5,2.5,3.5,5.0",
        help=(
            "Daftar noise_std (Kelvin) dipisah koma, urutan bebas (dinormalisasi urut naik). "
            "0.0 = baseline tanpa noise (dipakai sbg pembanding, BELUM pernah dites di split "
            "terpurge -- lihat CLAUDE.md §18.6/§18.9). Default: 0.0,1.5,2.5,3.5,5.0."
        ),
    )
    p.add_argument("--dataset", default=None, help="Path CSV dataset expanding window. Default: Config.EXPANDING_DATASET_FILE.")
    p.add_argument("--cache", default=None, help="Path cache raw .npz. Default: Config.EXPANDING_RAW_CACHE_FILE.")
    p.add_argument("--test-frac", type=float, default=None, help="Fraksi test, HARUS SAMA dengan 03_train_models.py. Default: Config.TEST_FRAC.")
    p.add_argument("--sweep-models-dir", default=None, help="Folder dasar model per nilai sweep. Default: Config.NOISE_SWEEP_MODELS_DIR.")
    p.add_argument(
        "--damping-factor", type=float, default=1.0,
        help="damping_factor dipakai SAAT EVALUASI recursive (bukan training). Default: 1.0 (tanpa redaman, konfigurasi produksi -- CLAUDE.md §18.9).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    try:
        values = sorted({round(float(v.strip()), 4) for v in args.values.split(",") if v.strip()})
    except ValueError:
        say_error(f"--values tidak valid: {args.values!r} (harus angka dipisah koma, mis. '0,1.5,2.5,4')")
        return
    if any(v < 0 for v in values):
        say_error("Semua noise_std harus >= 0.")
        return

    dataset_path = args.dataset or Config.EXPANDING_DATASET_FILE
    cache_path = args.cache or Config.EXPANDING_RAW_CACHE_FILE
    sweep_models_dir = args.sweep_models_dir or Config.NOISE_SWEEP_MODELS_DIR

    banner("SWEEP NOISE_STD - CTT FORECASTING (expanding window)")
    say_info(f"Dataset       : {dataset_path}")
    say_info(f"Cache raw     : {cache_path}")
    say_info(f"Nilai sweep   : {', '.join(str(v) for v in values)}")
    say_info(f"Model output  : {sweep_models_dir} (TERPISAH dari model produksi)")
    say_info(f"Damping (eval): {args.damping_factor}")
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

    # --- Load dataset (CSV besar) + cache + anchor test SEKALI -- semuanya
    # TIDAK bergantung ke noise_std, dipakai ulang di semua nilai sweep. ---
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
    # Output eval KHUSUS ke Config.NOISE_STD_SWEEP_DIR (evaluation/sweep_noise_std/)
    # -- TERPISAH dari Config.EXPANDING_EVAL_DIR (output RESMI Tahap 04),
    # supaya evaluation/ nggak kecampur belasan file per-nilai. TANPA
    # retensi/folder-per-run -- ditimpa tiap sweep dijalankan ulang.
    os.makedirs(Config.NOISE_STD_SWEEP_DIR, exist_ok=True)
    all_summaries = []
    all_train_summaries = []

    for noise_std in make_progress_bar(values, desc="Sweep noise_std", unit="nilai"):
        suffix = f"_noise{noise_std:.2f}".replace(".", "")
        gap()
        say_info(f"=== noise_std={noise_std} ===")

        # --- Retrain 3 model dgn noise_std ini, ke subfolder SENDIRI ---
        value_models_dir = os.path.join(sweep_models_dir, suffix.lstrip("_"))
        train_summary = train_all_models(
            df, models_dir=value_models_dir, test_frac=args.test_frac,
            noise_std=noise_std, cache_path=cache_path,
        )
        train_summary.insert(0, "noise_std_sweep", noise_std)
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
            say_error(f"Semua model gagal di-training utk noise_std={noise_std}, nilai ini dilewati dari perbandingan.")
            continue

        detail_df = pd.concat(detail_frames, ignore_index=True)
        summary_df = summarize_by_step(detail_df)
        summary_df.insert(0, "noise_std", noise_std)
        all_summaries.append(summary_df)

        # Simpan detail+summary per-nilai, suffix sama pola sama sweep_damping.py.
        detail_path = os.path.join(Config.NOISE_STD_SWEEP_DIR, f"recursive_evaluation{suffix}.csv")
        summary_path = os.path.join(Config.NOISE_STD_SWEEP_DIR, f"recursive_mae_summary{suffix}.csv")
        detail_df.to_csv(detail_path, index=False)
        summary_df.drop(columns=["noise_std"]).to_csv(summary_path, index=False)
        say_ok(f"noise_std={noise_std}: detail & summary tersimpan (suffix {suffix}).")

    if not all_summaries:
        say_error("Tidak ada nilai noise_std yang berhasil dievaluasi -- cek log di atas.")
        return

    combined = pd.concat(all_summaries, ignore_index=True)
    combined_path = os.path.join(Config.NOISE_STD_SWEEP_DIR, "noise_std_sweep_comparison.csv")
    combined.to_csv(combined_path, index=False)

    combined_train = pd.concat(all_train_summaries, ignore_index=True)
    combined_train_path = os.path.join(sweep_models_dir, "noise_std_sweep_training_summary.csv")
    combined_train.to_csv(combined_train_path, index=False)

    gap()
    banner("RINGKASAN SWEEP (step terakhir per nilai, per model)")
    last_step = combined["step"].max()
    final_step_view = combined[combined["step"] == last_step].sort_values(
        ["model", "noise_std"], ascending=[True, True]
    )
    for model_name in Config.MODEL_NAMES:
        sub = final_step_view[final_step_view["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} (step {last_step}) --")
        print(sub[["noise_std", "mae", "rmse", "spatial_collapse_ratio", "spatial_correlation"]].to_string(index=False))

    hr()
    say_ok(f"Model per nilai sweep tersimpan di: {sweep_models_dir} (TIDAK menimpa model produksi)")
    say_ok(f"Detail & summary per nilai: {Config.NOISE_STD_SWEEP_DIR} (suffix _noiseXXX)")
    say_ok(f"Perbandingan gabungan (semua nilai, semua step): {combined_path}")
    say_ok(f"Ringkasan training (waktu/MAE flat) per nilai: {combined_train_path}")
    say_info(
        "Cara baca: bandingkan MAE step terakhir antar noise_std -- titik minimum "
        "itu kandidat optimal (sama pola kayak damping sweep, CLAUDE.md §17 poin 6), "
        "TAPI cek juga spatial_collapse_ratio/spatial_correlation -- kalau ada nilai "
        "dgn MAE sedikit lebih tinggi TAPI metrik spasial JAUH lebih baik, itu "
        "trade-off yang wajar dipertimbangkan (bukan otomatis kalah). Kalau model "
        "production ini mau dipakai beneran, salin dari "
        f"{sweep_models_dir}/noise<nilai_terpilih>/*.joblib ke {Config.EXPANDING_MODELS_DIR}/ "
        "manual, jangan asumsikan otomatis."
    )


if __name__ == "__main__":
    main()
