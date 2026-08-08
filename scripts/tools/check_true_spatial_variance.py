# ./scripts/tools/check_true_spatial_variance.py
# Follow-up dari compare_interior_vs_edge_spatial_metrics.py. Temuan sesi
# sebelumnya: exclude pixel tepi (20 dari 35 pixel) malah bikin
# spatial_collapse_ratio NAIK & spatial_correlation TURUN -- kebalikan dari
# hipotesis "pixel tepi = sumber distorsi". Hipotesis baru: variasi TBB
# ASLI (bukan prediksi) antar pixel INTERIOR (15 pixel, saling berdekatan,
# area kecil) memang jauh lebih kecil dari variasi antar pixel FULL grid
# (yang mencakup tepi, jarak geografis lebih jauh) -- kalau std(aktual)
# jadi penyebut yang mengecil drastis, rasio std(pred)/std(aktual) otomatis
# membesar meski std(pred) sendiri tidak ikut memburuk.
#
# Script ini ngukur LANGSUNG std(y_true) & std(y_pred) antar pixel (per
# anchor_t0, dirata-rata per step) -- full grid vs interior-only -- buat
# konfirmasi/gugurkan hipotesis itu tanpa perlu nebak dari rasio.
#
# TIDAK re-run model, cuma baca ulang recursive_evaluation.csv yang sudah
# ada (sama seperti compare_interior_vs_edge_spatial_metrics.py).
#
# Cara pakai (dari folder scripts/, sama seperti 01-04):
#   python .\tools\check_true_spatial_variance.py

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.dataset_builder import load_raw_cache
from ui.terminal_display import banner, gap, hr, say_info, say_ok, say_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Ukur langsung std(y_true) & std(y_pred) antar pixel, full grid vs interior-only."
    )
    p.add_argument("--detail", default=None, help="Path recursive_evaluation.csv. Default: Config.RECURSIVE_EVAL_DETAIL_FILE")
    p.add_argument("--cache", default=None, help="Path cache raw .npz. Default: Config.EXPANDING_RAW_CACHE_FILE")
    p.add_argument("--out", default=None, help="Path CSV output. Default: {EXPANDING_EVAL_DIR}/diagnostics/true_spatial_variance.csv")
    return p.parse_args()


def _std_summary_for_subset(df, label):
    """Per (model, step, anchor_t0) dengan >=2 pixel, hitung std(y_true) &
    std(y_pred) antar pixel, lalu rata-ratakan lintas anchor_t0 per
    (model, step). Sama semangatnya dengan _spatial_metrics_per_step di
    pipeline.recursive_eval, tapi ngukur std MENTAH, bukan rasio/korelasi."""
    rows = []
    for (model, step, t0), group in df.groupby(["model", "step", "anchor_t0"]):
        if len(group) < 2:
            continue
        rows.append({
            "model": model,
            "step": step,
            "anchor_t0": t0,
            "true_std": float(np.std(group["y_true"].values)),
            "pred_std": float(np.std(group["y_pred"].values)),
        })
    if not rows:
        return pd.DataFrame(columns=["model", "step", "true_std_mean", "pred_std_mean", "n_t0_groups"])
    per_t0 = pd.DataFrame(rows)
    summary = per_t0.groupby(["model", "step"]).agg(
        true_std_mean=("true_std", "mean"),
        pred_std_mean=("pred_std", "mean"),
        n_t0_groups=("true_std", "size"),
    ).reset_index()
    summary["subset"] = label
    return summary


def main():
    args = parse_args()
    detail_path = args.detail or Config.RECURSIVE_EVAL_DETAIL_FILE
    cache_path = args.cache or Config.EXPANDING_RAW_CACHE_FILE
    out_path = args.out or os.path.join(Config.EXPANDING_EVAL_DIR, "diagnostics", "true_spatial_variance.csv")

    banner("CEK VARIASI SPASIAL ASLI (std y_true) -- FULL vs INTERIOR")
    say_info(f"Detail eval : {detail_path}")
    say_info(f"Cache raw   : {cache_path}")
    hr()

    for path in [detail_path, cache_path]:
        if not os.path.exists(path):
            say_error(f"File tidak ditemukan: {path}")
            return

    say_info("Load pixel_meta (lat_idx/lon_idx) dari cache...")
    _, _, pixel_meta = load_raw_cache(cache_path)

    lat_min, lat_max = pixel_meta["lat_idx"].min(), pixel_meta["lat_idx"].max()
    lon_min, lon_max = pixel_meta["lon_idx"].min(), pixel_meta["lon_idx"].max()
    is_edge = (
        (pixel_meta["lat_idx"] == lat_min) | (pixel_meta["lat_idx"] == lat_max) |
        (pixel_meta["lon_idx"] == lon_min) | (pixel_meta["lon_idx"] == lon_max)
    )
    interior_pixel_ids = set(pixel_meta.loc[~is_edge, "pixel_id"])
    say_info(f"Pixel interior dipakai: {len(interior_pixel_ids)} dari {len(pixel_meta)} total")
    hr()

    say_info("Load recursive_evaluation.csv...")
    detail_df = pd.read_csv(detail_path)
    interior_df = detail_df[detail_df["pixel_id"].isin(interior_pixel_ids)].copy()

    full_summary = _std_summary_for_subset(detail_df, "full_35px")
    interior_summary = _std_summary_for_subset(interior_df, "interior_15px")
    combined = pd.concat([full_summary, interior_summary], ignore_index=True)
    combined = combined.sort_values(["model", "step", "subset"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)
    say_ok(f"Hasil disimpan ke: {out_path}")

    hr()
    banner("RINGKASAN PER MODEL -- true_std_mean & pred_std_mean, full vs interior")
    for model_name in Config.MODEL_NAMES:
        sub = combined[combined["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} --")
        pivot = sub.pivot(index="step", columns="subset", values=["true_std_mean", "pred_std_mean"])
        print(pivot.to_string())
        gap()

    # Ringkasan langsung: berapa persen true_std mengecil vs pred_std pas exclude tepi, step terakhir
    hr()
    banner("RINGKASAN LANGSUNG -- step terakhir, berapa % masing2 std mengecil pas exclude pixel tepi")
    last_step = combined["step"].max()
    for model_name in Config.MODEL_NAMES:
        sub = combined[(combined["model"] == model_name) & (combined["step"] == last_step)]
        if len(sub) < 2:
            continue
        full_row = sub[sub["subset"] == "full_35px"].iloc[0]
        interior_row = sub[sub["subset"] == "interior_15px"].iloc[0]
        true_shrink = 1 - (interior_row["true_std_mean"] / full_row["true_std_mean"])
        pred_shrink = 1 - (interior_row["pred_std_mean"] / full_row["pred_std_mean"])
        print(
            f"{model_name:>10} (step {last_step}): "
            f"true_std mengecil {true_shrink:+.1%}, pred_std mengecil {pred_shrink:+.1%} "
            f"({'true mengecil LEBIH BANYAK -> hipotesis kekonfirmasi' if true_shrink > pred_shrink else 'pred mengecil lebih banyak/setara -> hipotesis TIDAK kekonfirmasi'})"
        )

    hr()
    say_info(
        "Cara baca: kalau true_std_mean interior_15px JAUH lebih kecil dari "
        "full_35px (dibanding penurunan pred_std_mean), itu konfirmasi: variasi "
        "TBB ASLI antar pixel interior memang jauh lebih kecil (area kecil, "
        "saling berdekatan, kondisi awan cenderung smooth) -- rasio "
        "spatial_collapse_ratio naik BUKAN karena model tambah ngaco, tapi "
        "karena penyebutnya (variasi asli) yang menyusut. Kalau true_std & "
        "pred_std mengecil dengan proporsi mirip, berarti bukan itu "
        "penjelasannya, perlu digali arah lain."
    )


if __name__ == "__main__":
    main()