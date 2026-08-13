# ./scripts/tools/compare_interior_vs_edge_spatial_metrics.py
# Follow-up dari diagnose_recursive_drift.py. Temuan sesi diagnostic
# sebelumnya: pixel yang paling "kabur" (top mean_abs_dev_from_anchor_mean)
# semuanya persis di lat_idx/lon_idx EKSTREM grid (lat_idx 0 & 4, lon_idx 0
# & 6) -- yaitu tepi bounding box (Config.BANDUNG_LAT_MIN/MAX,
# BANDUNG_LON_MIN/MAX), bukan pixel acak. Hipotesis: spatial_collapse_ratio
# yang meleset jauh dari 1 di summary 04_recursive_evaluate.py didominasi
# distorsi dari 20 dari 35 pixel yang duduk di tepi grid (kemungkinan
# artefak subsetting NetCDF di batas bbox), bukan kegagalan recursive
# forecasting yang menyeluruh.
#
# Script ini TIDAK re-run model / TIDAK re-predict apapun -- cuma baca
# ulang recursive_evaluation.csv (hasil 04_recursive_evaluate.py yang
# sudah ada) dan recompute spatial_collapse_ratio/spatial_correlation dua
# kali: (1) full 35 pixel (harus persis sama dengan recursive_mae_summary.csv
# yang sudah ada -- dipakai sebagai sanity check bahwa definisi tidak
# berubah), (2) HANYA 15 pixel interior (lat_idx 1..3, lon_idx 1..5).
# Kalau versi interior jauh lebih dekat ke 1 (collapse_ratio) dan lebih
# tinggi (correlation) dibanding versi full, itu konfirmasi kuat: masalah
# utamanya terkonsentrasi di pixel tepi.
#
# Cara pakai (dari folder scripts/, sama seperti 01-04):
#   python .\tools\compare_interior_vs_edge_spatial_metrics.py

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

import pandas as pd

from pipeline.config import Config
from pipeline.dataset_builder import load_raw_cache
from pipeline.recursive_eval import _spatial_metrics_per_step
from ui.terminal_display import banner, gap, hr, say_info, say_ok, say_error


def parse_args():
    p = argparse.ArgumentParser(
        description="Bandingin spatial_collapse_ratio/correlation full grid vs interior-only (exclude pixel tepi bbox)."
    )
    p.add_argument("--detail", default=None, help="Path recursive_evaluation.csv. Default: Config.RECURSIVE_EVAL_DETAIL_FILE")
    p.add_argument("--cache", default=None, help="Path cache raw .npz (buat ambil lat_idx/lon_idx per pixel). Default: Config.EXPANDING_RAW_CACHE_FILE")
    p.add_argument("--out", default=None, help="Path CSV output perbandingan. Default: {EXPANDING_EVAL_DIR}/diagnostics/interior_vs_full_spatial_metrics.csv")
    return p.parse_args()


def _spatial_metrics_for_subset(detail_df, label):
    """Wrapper tipis di atas pipeline.recursive_eval._spatial_metrics_per_step
    (fungsi CANONICAL, satu-satunya tempat logika grouping (model, step,
    anchor_t0) + filter min 2 pixel per grup didefinisikan) -- cuma nambah
    kolom "subset" biar hasil full vs interior bisa dibedain di CSV
    perbandingan. TIDAK reimplementasi logic-nya sendiri (dulu ada duplikat
    persis, sekarang dihapus supaya kalau definisi spatial_collapse_ratio/
    correlation berubah lagi, cuma satu tempat yang perlu diubah)."""
    summary = _spatial_metrics_per_step(detail_df)
    summary["subset"] = label
    return summary


def main():
    args = parse_args()
    detail_path = args.detail or Config.RECURSIVE_EVAL_DETAIL_FILE
    cache_path = args.cache or Config.EXPANDING_RAW_CACHE_FILE
    out_path = args.out or os.path.join(Config.EXPANDING_EVAL_DIR, "diagnostics", "interior_vs_full_spatial_metrics.csv")

    banner("INTERIOR vs FULL GRID -- SPATIAL METRICS COMPARISON")
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
    edge_pixel_ids = set(pixel_meta.loc[is_edge, "pixel_id"])
    interior_pixel_ids = set(pixel_meta.loc[~is_edge, "pixel_id"])

    say_info(
        f"Grid: lat_idx {lat_min}..{lat_max}, lon_idx {lon_min}..{lon_max} "
        f"({len(pixel_meta)} pixel total)"
    )
    say_info(f"Pixel tepi (di-exclude)  : {len(edge_pixel_ids)} pixel -> {sorted(edge_pixel_ids)}")
    say_info(f"Pixel interior (dipakai) : {len(interior_pixel_ids)} pixel -> {sorted(interior_pixel_ids)}")
    hr()

    say_info("Load recursive_evaluation.csv...")
    detail_df = pd.read_csv(detail_path)
    say_info(f"Total baris detail: {len(detail_df)}")

    interior_df = detail_df[detail_df["pixel_id"].isin(interior_pixel_ids)].copy()
    say_info(f"Baris setelah exclude pixel tepi: {len(interior_df)} ({len(interior_df) / len(detail_df):.1%} dari total)")
    hr()

    full_summary = _spatial_metrics_for_subset(detail_df, "full_35px")
    interior_summary = _spatial_metrics_for_subset(interior_df, "interior_15px")

    combined = pd.concat([full_summary, interior_summary], ignore_index=True)
    combined = combined.sort_values(["model", "step", "subset"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    combined.to_csv(out_path, index=False)
    say_ok(f"Hasil perbandingan disimpan ke: {out_path}")

    hr()
    banner("RINGKASAN PER MODEL -- full_35px vs interior_15px")
    for model_name in Config.MODEL_NAMES:
        sub = combined[combined["model"] == model_name]
        if sub.empty:
            continue
        print(f"\n-- {model_name} --")
        pivot = sub.pivot(index="step", columns="subset", values=["spatial_collapse_ratio", "spatial_correlation"])
        print(pivot.to_string())
        gap()

    hr()
    say_info(
        "Cara baca: bandingin kolom (spatial_collapse_ratio, full_35px) vs "
        "(spatial_collapse_ratio, interior_15px) per step. Kalau interior_15px "
        "jauh lebih dekat ke 1.0 (dan spatial_correlation interior_15px jauh "
        "lebih tinggi) dibanding full_35px, itu konfirmasi: distorsi metrik "
        "spasial terkonsentrasi di 20 pixel tepi grid, bukan gagal menyeluruh. "
        "Kalau interior_15px MASIH jauh dari 1 / korelasi MASIH rendah, berarti "
        "pixel tepi cuma memperparah, bukan satu-satunya penyebab -- exposure "
        "bias/covariate shift (temuan diagnose_recursive_drift.py sebelumnya) "
        "tetap relevan buat pixel interior juga."
    )


if __name__ == "__main__":
    main()