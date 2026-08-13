# ./scripts/tools/diagnose_recursive_drift.py
# Diagnostic TAMBAHAN, bukan pengganti 04_recursive_evaluate.py. Dipakai
# buat investigasi kenapa spatial_collapse_ratio NAIK (bukan turun ke 0
# seperti masalah "collapse" di repo lama) sementara spatial_correlation
# ambruk -- lihat diskusi sesi analisis di chat. Tiga pengecekan:
#
#   A. Runaway pixel -- pixel mana yang paling nyumbang ke pelebaran std
#      antar-pixel di step besar, dan apakah prediksinya sering mentok
#      deket batas nilai target training (indikasi model nge-clamp /
#      ekstrapolasi di leaf boundary tree, bukan prediksi wajar).
#   B. Feature distribution shift -- bandingin distribusi 9 fitur yang
#      DIPAKAI ULANG saat rollout recursive (window makin lama makin
#      isi hasil prediksi sendiri) vs distribusi fitur yang sama di
#      training set ASLI (window isi observasi real) untuk n_points yang
#      sama. Overlap persentil 5-95 kecil = sinyal covariate shift /
#      model lagi ekstrapolasi ke wilayah fitur yang jarang dilihat.
#   C. Clamping check -- persentase prediksi per pixel yang nempel di
#      [target_min, target_max] training.
#
# TIDAK re-train / TIDAK menggantikan recursive_evaluation.csv --
# murni tambahan analisis dari artifact yang sudah ada (model .joblib,
# cache raw .npz, dataset CSV). Jalankan SETELAH 02+03+04 selesai, dari
# folder scripts/ (sama seperti script 01-04), taruh file ini di
# scripts/tools/ (folder itu sudah disebut sebagai tempatnya diagnostic
# script di CLAUDE.md §9, sebelumnya masih kosong).
#
# Cara pakai:
#   python .\tools\diagnose_recursive_drift.py
#   python .\tools\diagnose_recursive_drift.py --steps 1,3,6,10,14,18

import argparse
import os
import sys

# Script ini sengaja ditaruh di scripts/tools/ (subfolder), BEDA dari
# 01-04 yang langsung di scripts/. Kalau dijalanin langsung (`py
# .\tools\diagnose_recursive_drift.py`), Python cuma masukin folder SCRIPT
# INI (scripts/tools/) ke sys.path, BUKAN scripts/ -- jadi `pipeline` &
# `ui` (yang posisinya sejajar tools/, satu level di atas) nggak ketemu.
# Fix: masukin manual folder scripts/ (parent dari folder file ini) ke
# sys.path SEBELUM import pipeline/ui, biar script ini jalan dari folder
# manapun dipanggil (baik `python .\tools\diagnose_recursive_drift.py`
# dari scripts/, maupun dari root project).
_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

import joblib
import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.expanding_features import FEATURE_COLUMNS
from pipeline.dataset_builder import load_raw_cache
from pipeline.model_training import load_expanding_dataset
from pipeline.recursive_eval import (
    MIN_WINDOW_SIZE,
    TARGET_COLUMN,
    select_test_anchors,
    _map_anchors_to_indices,
    run_recursive_evaluation,
)
from ui.terminal_display import banner, gap, hr, say_info, say_ok, say_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Diagnostic recursive drift: runaway pixel + feature distribution shift + clamping check."
    )
    p.add_argument("--dataset", default=None, help="Default: Config.EXPANDING_DATASET_FILE")
    p.add_argument("--cache", default=None, help="Default: Config.EXPANDING_RAW_CACHE_FILE")
    p.add_argument("--models-dir", default=None, help="Default: Config.EXPANDING_MODELS_DIR")
    p.add_argument(
        "--test-frac", type=float, default=None,
        help="HARUS SAMA dengan yang dipakai 03_train_model.py/04_recursive_evaluate.py biar anchor test konsisten. Default: Config.TEST_FRAC.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Default: {Config.EXPANDING_EVAL_DIR}/diagnostics/",
    )
    p.add_argument(
        "--steps", default="1,3,6,10,14,18",
        help="Step mana saja yang dicek detail buat runaway pixel & feature shift (comma-separated). Default: 1,3,6,10,14,18.",
    )
    p.add_argument(
        "--clamp-margin", type=float, default=0.5,
        help="Margin (K) dari target_min/target_max training buat dianggap 'mentok batas'. Default: 0.5.",
    )
    return p.parse_args()


def diagnose_runaway_pixels(pred_df, target_min, target_max, clamp_margin, steps_to_check):
    """Poin A + C. Buat tiap step yang diminta: hitung seberapa jauh y_pred
    tiap pixel dari rata-rata y_pred pixel lain DI ANCHOR_T0 YANG SAMA
    (proxy kontribusi pixel itu ke pelebaran std antar-pixel), dirata-rata
    lintas anchor_t0. Sekalian hitung fraksi prediksi yang nempel di
    [target_min, target_max] training (indikasi clamping / ekstrapolasi
    leaf boundary, bukan prediksi wajar)."""
    rows = []
    for step in steps_to_check:
        sub = pred_df[pred_df["step"] == step]
        if sub.empty:
            continue

        anchor_mean = sub.groupby("anchor_t0")["y_pred"].transform("mean")
        dev = (sub["y_pred"] - anchor_mean).abs()
        tmp = sub.assign(dev=dev)

        near_lo = (tmp["y_pred"] <= target_min + clamp_margin).groupby(tmp["pixel_id"]).mean()
        near_hi = (tmp["y_pred"] >= target_max - clamp_margin).groupby(tmp["pixel_id"]).mean()

        per_pixel = tmp.groupby("pixel_id").agg(
            mean_abs_dev_from_anchor_mean=("dev", "mean"),
            n_anchor=("dev", "size"),
            pred_min=("y_pred", "min"),
            pred_max=("y_pred", "max"),
        ).reset_index()
        per_pixel["step"] = step
        per_pixel["frac_near_train_min"] = per_pixel["pixel_id"].map(near_lo).fillna(0.0)
        per_pixel["frac_near_train_max"] = per_pixel["pixel_id"].map(near_hi).fillna(0.0)
        rows.append(per_pixel)

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows, ignore_index=True)
    return out.sort_values(["step", "mean_abs_dev_from_anchor_mean"], ascending=[True, False]).reset_index(drop=True)


def diagnose_feature_shift(rollout_features_df, train_df, steps_to_check):
    """Poin B. Buat tiap step yang diminta, bandingin distribusi tiap kolom
    FEATURE_COLUMNS antara (1) fitur yang dipakai model pas rollout
    recursive vs (2) fitur di training set ASLI dengan n_points yang sama.
    Overlap persentil 5-95 dipakai sebagai ukuran sederhana -- bukan test
    statistik formal, cukup buat sinyal awal seberapa jauh window rollout
    sudah bergeser dari yang pernah dilihat model pas training."""
    rows = []
    for step in steps_to_check:
        n_points_target = MIN_WINDOW_SIZE + step - 1
        roll_sub = rollout_features_df[rollout_features_df["step"] == step]
        train_sub = train_df[train_df["n_points"] == n_points_target]
        if roll_sub.empty or train_sub.empty:
            continue
        for col in FEATURE_COLUMNS:
            if col == "n_points":
                continue  # konstan per step by design, tidak informatif buat shift
            r_lo, r_hi = np.percentile(roll_sub[col], [5, 95])
            t_lo, t_hi = np.percentile(train_sub[col], [5, 95])
            overlap_lo = max(r_lo, t_lo)
            overlap_hi = min(r_hi, t_hi)
            overlap_width = max(0.0, overlap_hi - overlap_lo)
            union_width = max(r_hi, t_hi) - min(r_lo, t_lo)
            overlap_frac = overlap_width / union_width if union_width > 0 else 1.0
            rows.append({
                "step": step,
                "n_points": n_points_target,
                "feature": col,
                "rollout_p5": r_lo, "rollout_p95": r_hi,
                "train_p5": t_lo, "train_p95": t_hi,
                "overlap_frac_5_95": overlap_frac,
            })
    return pd.DataFrame(rows).sort_values(["step", "overlap_frac_5_95"]).reset_index(drop=True)


def main():
    args = parse_args()
    steps_to_check = sorted(set(int(s) for s in args.steps.split(",")))

    dataset_csv_path = args.dataset or Config.EXPANDING_DATASET_FILE
    cache_path = args.cache or Config.EXPANDING_RAW_CACHE_FILE
    models_dir = args.models_dir or Config.EXPANDING_MODELS_DIR
    out_dir = args.out_dir or os.path.join(Config.EXPANDING_EVAL_DIR, "diagnostics")

    banner("DIAGNOSTIC RECURSIVE DRIFT (runaway pixel + feature shift + clamping)")
    say_info(f"Dataset       : {dataset_csv_path}")
    say_info(f"Cache raw     : {cache_path}")
    say_info(f"Models dir    : {models_dir}")
    say_info(f"Step dicek    : {steps_to_check}")
    say_info(f"Clamp margin  : {args.clamp_margin} K")
    hr()

    for path in [dataset_csv_path, cache_path]:
        if not os.path.exists(path):
            say_error(f"File tidak ditemukan: {path}")
            return

    say_info("Load dataset training & cache raw...")
    train_df = load_expanding_dataset(dataset_csv_path)
    data_matrix, timeline, pixel_meta = load_raw_cache(cache_path)

    target_min = float(train_df[TARGET_COLUMN].min())
    target_max = float(train_df[TARGET_COLUMN].max())
    say_info(f"Range target training ({TARGET_COLUMN}): [{target_min:.2f}, {target_max:.2f}] K")

    say_info("Tentuin anchor test set (sama seperti 04_recursive_evaluate.py)...")
    anchors_df = select_test_anchors(dataset_csv_path, test_frac=args.test_frac)
    anchors_idx = _map_anchors_to_indices(anchors_df, timeline, pixel_meta)
    say_info(f"Jumlah anchor test unik: {len(anchors_idx)}")
    hr()

    os.makedirs(out_dir, exist_ok=True)

    all_runaway = []
    all_shift = []

    for model_name in Config.MODEL_NAMES:
        model_path = os.path.join(models_dir, f"{model_name}.joblib")
        if not os.path.exists(model_path):
            say_info(f"PERINGATAN: model {model_name} tidak ditemukan di {model_path}, dilewati.")
            continue

        say_info(f"Rollout ulang + capture fitur: {model_name}")
        model = joblib.load(model_path)
        # damping_factor=1.0 (tanpa redaman) -- diagnostic ini dari sesi
        # SEBELUM damping ada, tujuannya lihat drift MENTAH model, bukan
        # efek setelah diredam. Pakai rollout canonical (recursive_eval.py)
        # dgn capture_features=True, BUKAN implementasi rollout terpisah.
        pred_df, rollout_features_df = run_recursive_evaluation(
            model_name, model, anchors_idx, data_matrix, timeline,
            damping_factor=1.0, capture_features=True,
        )

        runaway = diagnose_runaway_pixels(pred_df, target_min, target_max, args.clamp_margin, steps_to_check)
        runaway["model"] = model_name
        all_runaway.append(runaway)

        shift = diagnose_feature_shift(rollout_features_df, train_df, steps_to_check)
        shift["model"] = model_name
        all_shift.append(shift)

        gap()

    if not all_runaway:
        say_error("Nggak ada model yang ke-load dari models-dir, diagnostic dibatalkan.")
        return

    runaway_df = pd.concat(all_runaway, ignore_index=True)
    shift_df = pd.concat(all_shift, ignore_index=True)

    runaway_path = os.path.join(out_dir, "runaway_pixels.csv")
    shift_path = os.path.join(out_dir, "feature_distribution_shift.csv")
    runaway_df.to_csv(runaway_path, index=False)
    shift_df.to_csv(shift_path, index=False)
    say_ok(f"Detail runaway pixel disimpan ke : {runaway_path}")
    say_ok(f"Detail feature shift disimpan ke : {shift_path}")

    hr()
    banner("RINGKASAN -- TOP 5 PIXEL PALING 'KABUR' PER MODEL/STEP")
    for model_name in Config.MODEL_NAMES:
        sub = runaway_df[runaway_df["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} --")
        for step in steps_to_check:
            step_sub = sub[sub["step"] == step].head(5)
            if step_sub.empty:
                continue
            print(f"  step {step}:")
            print(step_sub[[
                "pixel_id", "mean_abs_dev_from_anchor_mean",
                "frac_near_train_min", "frac_near_train_max",
            ]].to_string(index=False))

    hr()
    banner("RINGKASAN -- FITUR PALING 'GESER' (overlap p5-p95 rollout vs training)")
    for model_name in Config.MODEL_NAMES:
        sub = shift_df[shift_df["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} --")
        worst = sub.sort_values("overlap_frac_5_95").groupby("step", group_keys=False).head(3)
        print(worst[[
            "step", "feature", "overlap_frac_5_95",
            "rollout_p5", "rollout_p95", "train_p5", "train_p95",
        ]].to_string(index=False))

    hr()
    say_info(
        "Cara baca: overlap_frac_5_95 mendekati 0 = distribusi fitur pas recursive rollout "
        "udah beda jauh dari yang dilihat model pas training (window isi observasi real) -- "
        "makin banyak fitur/step dengan overlap rendah, makin kuat indikasi model lagi "
        "ekstrapolasi ke wilayah fitur yang jarang/nggak pernah dilatih. frac_near_train_min/max "
        "tinggi (mis. >0.3) di pixel tertentu = prediksi pixel itu sering mentok di batas nilai "
        "target training -- indikasi clamping di leaf boundary tree model. "
        "mean_abs_dev_from_anchor_mean tinggi & konsisten di pixel yang sama lintas step = pixel "
        "itu yang paling nyumbang ke kenaikan spatial_collapse_ratio."
    )


if __name__ == "__main__":
    main()