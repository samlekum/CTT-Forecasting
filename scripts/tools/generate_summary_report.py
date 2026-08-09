# ./scripts/tools/generate_summary_report.py
# Ringkasan hasil pipeline Stage 02-05, dibuat buat bahan presentasi (PPT) --
# BUKAN generate file .pptx (diputuskan sengaja, lihat CLAUDE.md kalau ada
# sesi depan yang bingung kenapa) -- output-nya tabel Markdown (angka siap
# copy-paste) + chart PNG (siap drag ke slide), user susun sendiri layout
# slide-nya.
#
# Sumber data per stage:
#   02 -- dataset/expanding_features.csv (BESAR, 3GB+ -- dibaca CHUNKED,
#         cuma kolom pixel_id/anchor_t0/target_time/step, TIDAK di-load penuh)
#   03 -- models/training_summary.csv (kecil, langsung di-load)
#   04 -- evaluation/recursive_mae_summary.csv (kecil, summary per model x
#         step -- BUKAN recursive_evaluation.csv yang detail 700MB+)
#   05 -- forecast_output/{model}_t0..._run.../forecast.csv (dipilih via menu
#         CLI -- pola SAMA PERSIS dgn 06_visualize.py: auto-pick kalau cuma 1
#         hasil, navigasi panah/PageUp-PageDown kalau >1, --csv buat skip menu)
#
# Kalau salah satu sumber belum ada (mis. baru sampai Stage 03), bagian itu
# di-skip dengan pesan jelas -- BUKAN crash (biar tool ini tetap kepakai
# progresif, konsisten sama pola graceful-skip di seluruh project ini).
# PENGECUALIAN: kalau user membatalkan menu pilih forecast (Esc), SELURUH
# report dibatalkan (bukan cuma section 05) -- itu tindakan sadar user,
# sama seperti perilaku 06_visualize.py.
#
# Output: summary_report/{model}_t0..._run.../ (summary.md + charts/) --
# nama folder = identitas run forecast yang dipilih, SENGAJA idempotent
# (pola sama persis dgn visualizations/, lihat build_output_paths() di
# pipeline/visualize.py) -- generate ulang report dari run forecast yang
# SAMA akan overwrite folder yang sama, generate dari run forecast yang
# BEDA bikin folder baru (riwayat tetap ada, retensi per-run).

import argparse
import os
import sys
from collections import Counter
from datetime import datetime

_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pipeline.config import Config
from pipeline.inference import select_inference_model
from pipeline.visualize import (
    discover_forecast_files, prompt_select_forecast_file, load_forecast_csv,
    compute_grid_geometry, load_kecamatan_boundaries, pixel_rows_to_grid,
    classify_tbb_grid, render_map_panel, TBB_CMAP, MAE_CMAP, FORECAST_FOLDER_RE,
)
from ui.terminal_display import banner, gap, hr, say_info, say_ok, say_error, make_progress_bar

# --- Palet & style chart (skill dataviz, konsisten sama pipeline/visualize.py) ---
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID_LINE = "#e5e4df"
# Warna kategorikal per model -- urutan TETAP (Config.MODEL_NAMES), TIDAK
# di-cycle ulang berdasar ranking, biar 1 model = 1 warna konsisten di
# semua chart (slot 1/2/3 palette.md: blue/orange/aqua).
MODEL_COLORS = dict(zip(Config.MODEL_NAMES, ["#2a78d6", "#eb6834", "#1baf7a"]))


def _style_ax(ax, ylabel=None, xlabel=None):
    ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(TEXT_SECONDARY)
    ax.yaxis.grid(True, color=GRID_LINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9, color=TEXT_PRIMARY)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9, color=TEXT_PRIMARY)


def _new_fig(figsize=(7, 4.2)):
    fig, ax = plt.subplots(figsize=figsize, facecolor=SURFACE)
    return fig, ax


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def _df_to_markdown(df):
    """Tabel Markdown manual -- hindari dependency `tabulate` (belum
    terinstall di environment ini, dibutuhkan pandas DataFrame.to_markdown())."""
    cols = list(df.columns)
    header = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [
        "| " + " | ".join(str(v) for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([header, sep] + rows)


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate ringkasan Stage 02-05 (tabel + chart) buat bahan presentasi."
    )
    p.add_argument("--output-dir", default=None, help="Folder induk output. Default: Config.SUMMARY_REPORT_DIR (summary_report/).")
    p.add_argument("--dataset", default=None, help="CSV dataset Stage 02. Default: Config.EXPANDING_DATASET_FILE.")
    p.add_argument("--skip-dataset", action="store_true", help="Skip scan Stage 02 (file besar, ~30 detik) kalau cuma mau update bagian 03-05.")
    p.add_argument("--chunksize", type=int, default=2_000_000, help="Ukuran chunk baca CSV Stage 02. Default: 2000000.")
    p.add_argument("--models-dir", default=None, help="Default: Config.EXPANDING_MODELS_DIR.")
    p.add_argument("--eval-summary", default=None, help="Default: Config.RECURSIVE_EVAL_SUMMARY_FILE (baseline damping_factor=1.0).")
    p.add_argument("--forecast-dir", default=None, help="Folder berisi CSV forecast Stage 05. Default: Config.INFERENCE_DIR.")
    p.add_argument(
        "--csv", default=None,
        help=(
            "Path langsung ke 1 file forecast.csv (skip menu pilih) -- pola sama persis dgn "
            "06_visualize.py --csv. Default: None -> tampilkan menu pilih dari --forecast-dir "
            "(auto-pick kalau cuma 1 hasil, berhenti kalau 0)."
        ),
    )
    p.add_argument("--sample-step", type=int, default=None, help="Step yang dipetakan di contoh forecast. Default: HORIZON_STEPS (step terakhir).")
    return p.parse_args()


# ============================================================================
# Stage 02 -- ringkasan dataset (CHUNKED, jangan load penuh)
# ============================================================================

def summarize_dataset(csv_path, chunksize):
    if not os.path.exists(csv_path):
        say_info(f"[02] Dataset tidak ditemukan di {csv_path}, bagian ini di-skip.")
        return None

    say_info(f"[02] Scan dataset (chunked, chunksize={chunksize}): {csv_path}")
    total_rows = 0
    pixel_ids = set()
    min_anchor, max_anchor = None, None
    min_target, max_target = None, None
    rows_per_month = Counter()
    step_counts = Counter()

    reader = pd.read_csv(
        csv_path,
        usecols=["pixel_id", "anchor_t0", "target_time", "step"],
        parse_dates=["anchor_t0", "target_time"],
        chunksize=chunksize,
    )
    for chunk in make_progress_bar(reader, desc="Baca chunk", unit="chunk"):
        total_rows += len(chunk)
        pixel_ids.update(chunk["pixel_id"].unique().tolist())
        c_min_a, c_max_a = chunk["anchor_t0"].min(), chunk["anchor_t0"].max()
        c_min_t, c_max_t = chunk["target_time"].min(), chunk["target_time"].max()
        min_anchor = c_min_a if min_anchor is None else min(min_anchor, c_min_a)
        max_anchor = c_max_a if max_anchor is None else max(max_anchor, c_max_a)
        min_target = c_min_t if min_target is None else min(min_target, c_min_t)
        max_target = c_max_t if max_target is None else max(max_target, c_max_t)
        rows_per_month.update(chunk["anchor_t0"].dt.to_period("M").astype(str).value_counts().to_dict())
        step_counts.update(chunk["step"].value_counts().to_dict())

    horizon = Config.HORIZON_STEPS
    n_anchors = total_rows // horizon
    remainder = total_rows % horizon
    anchors_per_month = {month: count // horizon for month, count in sorted(rows_per_month.items())}

    return {
        "total_rows": total_rows,
        "n_pixels": len(pixel_ids),
        "n_anchors": n_anchors,
        "remainder_rows": remainder,
        "min_anchor_t0": min_anchor,
        "max_anchor_t0": max_anchor,
        "min_target_time": min_target,
        "max_target_time": max_target,
        "anchors_per_month": anchors_per_month,
        "step_counts": dict(sorted(step_counts.items())),
    }


def chart_dataset_overview(stats, out_path):
    months = list(stats["anchors_per_month"].keys())
    counts = list(stats["anchors_per_month"].values())
    fig, ax = _new_fig(figsize=(8, 4.2))
    bars = ax.bar(months, counts, color=MODEL_COLORS[Config.MODEL_NAMES[0]], width=0.6, zorder=3)
    for b, c in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{c:,}",
                 ha="center", va="bottom", fontsize=7.5, color=TEXT_PRIMARY)
    _style_ax(ax, ylabel="Jumlah anchor", xlabel="Bulan (anchor_t0)")
    ax.set_title("Distribusi Anchor per Bulan (Stage 02)", fontsize=11, color=TEXT_PRIMARY, loc="left")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    _save(fig, out_path)


# ============================================================================
# Stage 03 -- perbandingan model (training flat)
# ============================================================================

def summarize_training(models_dir):
    path = os.path.join(models_dir, "training_summary.csv")
    if not os.path.exists(path):
        say_info(f"[03] training_summary.csv tidak ditemukan di {path}, bagian ini di-skip.")
        return None
    df = pd.read_csv(path)
    df["model"] = pd.Categorical(df["model"], categories=Config.MODEL_NAMES, ordered=True)
    return df.sort_values("model").reset_index(drop=True)


def chart_training_mae_rmse(train_df, out_path):
    models = train_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.32
    fig, ax = _new_fig()
    ax.bar(x - width / 2, train_df["mae"], width, label="MAE", color="#2a78d6", zorder=3)
    ax.bar(x + width / 2, train_df["rmse"], width, label="RMSE", color="#eb6834", zorder=3)
    for xi, (mae, rmse) in enumerate(zip(train_df["mae"], train_df["rmse"])):
        ax.text(xi - width / 2, mae, f"{mae:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_PRIMARY)
        ax.text(xi + width / 2, rmse, f"{rmse:.3f}", ha="center", va="bottom", fontsize=7.5, color=TEXT_PRIMARY)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    _style_ax(ax, ylabel="Kelvin (K)")
    ax.set_title("Perbandingan Model -- MAE & RMSE Flat Test Set (Stage 03)", fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    _save(fig, out_path)


def chart_training_time(train_df, out_path):
    models = train_df["model"].tolist()
    times = train_df["waktu_training_detik"].tolist()
    fig, ax = _new_fig()
    bars = ax.bar(models, times, color=[MODEL_COLORS[m] for m in models], width=0.5, zorder=3)
    for b, t in zip(bars, times):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), f"{t:.1f}s",
                 ha="center", va="bottom", fontsize=8, color=TEXT_PRIMARY)
    _style_ax(ax, ylabel="Waktu training (detik)")
    ax.set_title("Perbandingan Model -- Waktu Training (Stage 03)", fontsize=11, color=TEXT_PRIMARY, loc="left")
    _save(fig, out_path)


# ============================================================================
# Stage 04 -- evaluasi recursive per step
# ============================================================================

def summarize_recursive_eval(summary_path):
    if not os.path.exists(summary_path):
        say_info(f"[04] recursive_mae_summary.csv tidak ditemukan di {summary_path}, bagian ini di-skip.")
        return None
    df = pd.read_csv(summary_path)
    df["model"] = pd.Categorical(df["model"], categories=Config.MODEL_NAMES, ordered=True)
    return df.sort_values(["model", "step"]).reset_index(drop=True)


def _line_chart_per_model(eval_df, y_col, ylabel, title, out_path, ref_line=None, ref_label=None):
    fig, ax = _new_fig()
    for model in Config.MODEL_NAMES:
        sub = eval_df[eval_df["model"] == model]
        if sub.empty:
            continue
        ax.plot(sub["step"], sub[y_col], color=MODEL_COLORS[model], linewidth=2,
                marker="o", markersize=3.5, label=model, zorder=3)
    if ref_line is not None:
        ax.axhline(ref_line, color=TEXT_SECONDARY, linewidth=1, linestyle="--", zorder=2)
        ax.text(eval_df["step"].max(), ref_line, f" {ref_label}", fontsize=7.5,
                color=TEXT_SECONDARY, va="center")
    _style_ax(ax, ylabel=ylabel, xlabel="Step (10 menit/step)")
    ax.set_title(title, fontsize=11, color=TEXT_PRIMARY, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="best")
    _save(fig, out_path)


def chart_mae_per_step(eval_df, out_path):
    _line_chart_per_model(eval_df, "mae", "MAE (K)", "MAE per Step -- Recursive Rollout (Stage 04)", out_path)


def chart_spatial_collapse_ratio(eval_df, out_path):
    _line_chart_per_model(
        eval_df, "spatial_collapse_ratio", "spatial_collapse_ratio",
        "Spatial Collapse Ratio per Step (Stage 04)", out_path,
        ref_line=1.0, ref_label="ideal = 1.0",
    )


def chart_spatial_correlation(eval_df, out_path):
    _line_chart_per_model(
        eval_df, "spatial_correlation", "spatial_correlation",
        "Spatial Correlation per Step (Stage 04)", out_path,
    )


def milestone_table(eval_df, steps=(1, 6, 12, 18)):
    horizon = Config.HORIZON_STEPS
    steps = [s for s in steps if s <= horizon]
    if horizon not in steps:
        steps.append(horizon)
    sub = eval_df[eval_df["step"].isin(steps)][
        ["model", "step", "mae", "rmse", "spatial_collapse_ratio", "spatial_correlation"]
    ].copy()
    return sub.sort_values(["model", "step"]).reset_index(drop=True)


# ============================================================================
# Stage 05 -- contoh hasil forecast
# ============================================================================

def select_forecast_run(forecast_dir, csv_override):
    """Pilih 1 run forecast Stage 05 -- pola SAMA PERSIS dgn
    visualize.visualize_forecast(): --csv langsung skip menu (model/t0/run
    diparse dari nama FOLDER induk, bukan nama file forecast.csv yang
    generic), kalau tidak diisi tampilkan menu pilih (reuse
    discover_forecast_files()/prompt_select_forecast_file() -- auto-pick
    kalau cuma 1 hasil, navigasi panah/PageUp-PageDown kalau >1, RAISE
    kalau 0 -- caller yang putuskan itu fatal atau cuma skip section 05).

    Return
    ------
    (detail_df, csv_path, model_name, t0_str, run_str)

    Raise
    -----
    FileNotFoundError / ValueError -- forecast_dir tidak ada, 0 kandidat,
    atau nama folder --csv tidak cocok pola {model}_t0{...}_run{...}.
    KeyboardInterrupt -- user tekan Esc di menu navigasi.
    """
    if csv_override:
        csv_path = csv_override
        folder_name = os.path.basename(os.path.dirname(csv_path))
        m = FORECAST_FOLDER_RE.match(folder_name)
        if not m:
            raise ValueError(
                f"Nama folder induk '{folder_name}' tidak cocok pola {{model}}_t0{{...}}_run{{...}} -- "
                "pastikan --csv menunjuk ke forecast.csv di dalam folder output 05_run_inference.py "
                "(forecast_output/{model}_t0..._run.../forecast.csv)."
            )
        model_name, t0_str, run_str = m.group("model"), m.group("t0"), m.group("run")
    else:
        candidates = discover_forecast_files(forecast_dir)
        selected = prompt_select_forecast_file(candidates)
        csv_path = selected["path"]
        model_name, t0_str, run_str = selected["model_name"], selected["t0_str"], selected["run_str"]

    detail_df = load_forecast_csv(csv_path)
    return detail_df, csv_path, model_name, t0_str, run_str


def chart_sample_forecast_maps(detail_df, sample_step, out_path):
    horizon = Config.HORIZON_STEPS
    if sample_step is None:
        sample_step = horizon
    step_df = detail_df[detail_df["step"] == sample_step]
    if step_df.empty:
        say_info(f"[05] Tidak ada baris untuk step={sample_step}, pakai step terakhir yang tersedia.")
        sample_step = int(detail_df["step"].max())
        step_df = detail_df[detail_df["step"] == sample_step]

    grid_pred = pixel_rows_to_grid(step_df, "y_pred")
    grid_true = pixel_rows_to_grid(step_df, "y_true")
    grid_mae = pixel_rows_to_grid(step_df, "abs_error")
    extent, center_lat = compute_grid_geometry()
    aspect = 1 / np.cos(np.radians(center_lat))
    boundaries = load_kecamatan_boundaries()

    has_actual = not np.all(np.isnan(grid_true))
    tbb_values = [grid_pred]
    if has_actual:
        tbb_values.append(grid_true)
    tbb_vmin = np.nanmin([np.nanmin(g) for g in tbb_values])
    tbb_vmax = np.nanmax([np.nanmax(g) for g in tbb_values])
    mae_vmax = np.nanmax(grid_mae) if not np.all(np.isnan(grid_mae)) else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), facecolor=SURFACE)
    render_map_panel(axes[0], grid_pred, extent, tbb_vmin, tbb_vmax, "Prediksi", TBB_CMAP,
                      "TBB (K)", boundaries=boundaries, aspect=aspect)
    render_map_panel(axes[1], grid_true, extent, tbb_vmin, tbb_vmax, "Aktual", TBB_CMAP,
                      "TBB (K)", boundaries=boundaries,
                      placeholder_text="Tidak ada observasi asli untuk step ini\n(forecast masa depan genuine)",
                      aspect=aspect)
    render_map_panel(axes[2], grid_mae, extent, 0, mae_vmax, "Error Map", MAE_CMAP,
                      "|error| (K)", boundaries=boundaries,
                      placeholder_text="Tidak ada observasi asli untuk step ini",
                      aspect=aspect)
    fig.suptitle(f"Contoh Forecast -- step {sample_step:02d}/{horizon:02d} (Stage 05)",
                 fontsize=12, color=TEXT_PRIMARY)
    _save(fig, out_path)
    return sample_step


def chart_sample_forecast_mae_per_step(detail_df, out_path):
    per_step = detail_df.dropna(subset=["y_true"]).groupby("step")["abs_error"].mean()
    if per_step.empty:
        return False
    fig, ax = _new_fig()
    ax.plot(per_step.index, per_step.values, color=MODEL_COLORS.get(
        detail_df["model_name"].iloc[0], "#2a78d6"), linewidth=2, marker="o", markersize=4, zorder=3)
    _style_ax(ax, ylabel="MAE (K)", xlabel="Step (10 menit/step)")
    ax.set_title("MAE per Step -- Run Forecast Ini Saja (Stage 05, bukan agregat evaluasi)",
                 fontsize=10.5, color=TEXT_PRIMARY, loc="left")
    _save(fig, out_path)
    return True


# ============================================================================
# Markdown report
# ============================================================================

def build_markdown(output_dir, dataset_stats, train_df, eval_df, production_model,
                    forecast_meta, sample_step, forecast_has_mae_chart):
    lines = []
    lines.append("# Ringkasan Hasil Pipeline CTT-Forecasting (Expanding Window)")
    lines.append(f"\n_Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}_\n")
    lines.append(
        "Dokumen ini ringkasan angka + chart dari Stage 02-05, dibuat buat bahan "
        "presentasi -- bukan file .pptx jadi, tinggal copy tabel/gambar ke slide.\n"
    )

    lines.append("## 1. Dataset (Stage 02 -- `expanding_features.csv`)\n")
    if dataset_stats is None:
        lines.append("_Tidak tersedia -- jalankan `02_build_expanding_features.py` dulu, atau tanpa `--skip-dataset`._\n")
    else:
        s = dataset_stats
        lines.append(f"- Total baris: **{s['total_rows']:,}**")
        lines.append(f"- Jumlah pixel: **{s['n_pixels']}/35** (grid Bandung {Config.PIXEL_GRID_SHAPE[0]}x{Config.PIXEL_GRID_SHAPE[1]})")
        lines.append(f"- Jumlah anchor unik: **{s['n_anchors']:,}** ({Config.HORIZON_STEPS} baris/anchor, sisa tak terbagi: {s['remainder_rows']})")
        lines.append(f"- Rentang anchor_t0: **{s['min_anchor_t0']}** s/d **{s['max_anchor_t0']}**")
        lines.append(f"- Rentang target_time (horizon terjauh): **{s['min_target_time']}** s/d **{s['max_target_time']}**")
        lines.append("\n![Distribusi anchor per bulan](charts/01_dataset_overview_anchors_per_month.png)\n")

    lines.append("## 2. Perbandingan Model -- Training Flat (Stage 03 -- `training_summary.csv`)\n")
    if train_df is None:
        lines.append("_Tidak tersedia -- jalankan `03_train_models.py` dulu._\n")
    else:
        lines.append(_df_to_markdown(train_df.round(4)))
        lines.append("\n![MAE & RMSE per model](charts/02_model_comparison_mae_rmse.png)")
        lines.append("\n![Waktu training per model](charts/03_model_comparison_training_time.png)\n")

    lines.append("## 3. Evaluasi Recursive per Step (Stage 04 -- `recursive_mae_summary.csv`, baseline damping_factor=1.0)\n")
    if eval_df is None:
        lines.append("_Tidak tersedia -- jalankan `04_recursive_evaluate.py` dulu._\n")
    else:
        if production_model is not None:
            model_name, avg_mae, ranking = production_model
            lines.append(
                f"**Model production (auto-detect, prioritas step {Config.INFERENCE_PRIORITY_STEP_RANGE}): "
                f"`{model_name}`** (avg MAE={avg_mae:.4f}K)\n"
            )
        lines.append("Milestone (step 1/6/12/18):\n")
        lines.append(_df_to_markdown(milestone_table(eval_df).round(4)))
        lines.append("\n![MAE per step](charts/04_recursive_mae_per_step.png)")
        lines.append("\n![Spatial collapse ratio per step](charts/05_spatial_collapse_ratio_per_step.png)")
        lines.append("\n![Spatial correlation per step](charts/06_spatial_correlation_per_step.png)")
        lines.append(
            "\n_Catatan: `spatial_collapse_ratio` idealnya mendekati 1 (variasi spasial prediksi "
            "= variasi spasial aktual), `spatial_correlation` idealnya mendekati 1. Konfigurasi "
            "produksi saat ini pakai `damping_factor=1.0` (tanpa redaman) -- keputusan final "
            "berdasar prioritas horizon panjang, lihat CLAUDE.md §18.9._\n"
        )

    lines.append("## 4. Contoh Hasil Forecast (Stage 05)\n")
    if forecast_meta is None:
        lines.append("_Tidak tersedia -- jalankan `05_run_inference.py` dulu._\n")
    else:
        lines.append(f"- File sumber: `{forecast_meta['path']}`")
        lines.append(f"- Model: **{forecast_meta['model_name']}**, damping_factor: **{forecast_meta['damping_factor']}**")
        lines.append(f"- t0 (observasi terakhir): **{forecast_meta['t0']}**")
        lines.append(f"- Jumlah pixel ke-forecast: **{forecast_meta['n_pixels']}/35**")
        lines.append(f"\n![Contoh peta forecast]({os.path.relpath(forecast_meta['map_chart'], output_dir).replace(os.sep, '/')})\n")
        if forecast_has_mae_chart:
            lines.append(f"![MAE per step run ini](charts/08_sample_forecast_mae_per_step.png)\n")
        else:
            lines.append(
                "_Run ini forecast ke masa depan genuine -- tidak ada observasi asli buat "
                "dibandingkan (y_true kosong semua), jadi tidak ada chart MAE run-spesifik._\n"
            )

    return "\n".join(lines)


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()

    output_base = args.output_dir or Config.SUMMARY_REPORT_DIR
    dataset_csv_path = args.dataset or Config.EXPANDING_DATASET_FILE
    models_dir = args.models_dir or Config.EXPANDING_MODELS_DIR
    eval_summary_path = args.eval_summary or Config.RECURSIVE_EVAL_SUMMARY_FILE
    forecast_dir = args.forecast_dir or Config.INFERENCE_DIR

    banner("GENERATE SUMMARY REPORT - STAGE 02-05")
    say_info(f"Folder forecast : {args.forecast_dir or '(Config.INFERENCE_DIR)'}")
    say_info(f"CSV langsung    : {args.csv or '(pilih via menu)'}")
    say_info(f"Folder output   : {output_base}")
    hr()

    # --- Stage 05 dulu -- pilih run forecast SEBELUM kerja berat lain,
    # sama seperti visualize_forecast(). Identitas run yang dipilih dipakai
    # sbg nama folder output (§ header file ini). Kalau belum ada forecast
    # SAMA SEKALI (FileNotFoundError/ValueError dari 0 kandidat), section 05
    # di-skip tapi report tetap lanjut -- BEDA dari 06_visualize.py yang
    # forecast-nya wajib. Esc di menu (KeyboardInterrupt) tetap membatalkan
    # SELURUH report, bukan cuma section 05 (tindakan sadar user).
    detail_df = model_name = t0_str = run_str = forecast_path = None
    try:
        detail_df, forecast_path, model_name, t0_str, run_str = select_forecast_run(forecast_dir, args.csv)
    except (FileNotFoundError, ValueError) as e:
        say_info(f"[05] {e}")
    except KeyboardInterrupt:
        say_info("Dibatalkan.")
        return

    if model_name is not None:
        output_dir = os.path.join(output_base, f"{model_name}_t0{t0_str}_run{run_str}")
    else:
        output_dir = os.path.join(output_base, "no_forecast_data")
    charts_dir = os.path.join(output_dir, "charts")
    os.makedirs(charts_dir, exist_ok=True)
    say_info(f"Output run      : {output_dir}")
    hr()

    # --- Stage 02 ---
    dataset_stats = None
    if args.skip_dataset:
        say_info("[02] --skip-dataset dipakai, bagian dataset di-skip.")
    else:
        dataset_stats = summarize_dataset(dataset_csv_path, args.chunksize)
        if dataset_stats:
            chart_dataset_overview(dataset_stats, os.path.join(charts_dir, "01_dataset_overview_anchors_per_month.png"))
            say_ok(f"[02] {dataset_stats['total_rows']:,} baris, {dataset_stats['n_anchors']:,} anchor, {dataset_stats['n_pixels']}/35 pixel.")

    # --- Stage 03 ---
    train_df = summarize_training(models_dir)
    if train_df is not None:
        chart_training_mae_rmse(train_df, os.path.join(charts_dir, "02_model_comparison_mae_rmse.png"))
        chart_training_time(train_df, os.path.join(charts_dir, "03_model_comparison_training_time.png"))
        say_ok(f"[03] {len(train_df)} model dibandingkan.")

    # --- Stage 04 ---
    eval_df = summarize_recursive_eval(eval_summary_path)
    production_model = None
    if eval_df is not None:
        chart_mae_per_step(eval_df, os.path.join(charts_dir, "04_recursive_mae_per_step.png"))
        chart_spatial_collapse_ratio(eval_df, os.path.join(charts_dir, "05_spatial_collapse_ratio_per_step.png"))
        chart_spatial_correlation(eval_df, os.path.join(charts_dir, "06_spatial_correlation_per_step.png"))
        try:
            production_model = select_inference_model(eval_summary_path)
        except (FileNotFoundError, ValueError) as e:
            say_info(f"[04] Tidak bisa tentukan model production otomatis: {e}")
        say_ok(f"[04] {eval_df['model'].nunique()} model x {eval_df['step'].nunique()} step.")

    # --- Stage 05 (detail_df/forecast_path sudah dipilih di atas) ---
    forecast_meta = None
    forecast_has_mae_chart = False
    if detail_df is not None:
        map_chart_path = os.path.join(charts_dir, "07_sample_forecast_maps.png")
        used_step = chart_sample_forecast_maps(detail_df, args.sample_step, map_chart_path)
        forecast_has_mae_chart = chart_sample_forecast_mae_per_step(
            detail_df, os.path.join(charts_dir, "08_sample_forecast_mae_per_step.png")
        )
        t0 = detail_df.loc[detail_df["step"] == detail_df["step"].min(), "target_time"].min() - pd.Timedelta(minutes=Config.FREQ_MINUTES)
        forecast_meta = {
            "path": forecast_path,
            "model_name": detail_df["model_name"].iloc[0],
            "damping_factor": detail_df["damping_factor"].iloc[0],
            "t0": t0,
            "n_pixels": detail_df["pixel_id"].nunique(),
            "map_chart": map_chart_path,
        }
        say_ok(f"[05] Contoh forecast dipetakan (step {used_step}).")

    # --- Markdown ---
    md = build_markdown(output_dir, dataset_stats, train_df, eval_df, production_model,
                         forecast_meta, args.sample_step, forecast_has_mae_chart)
    md_path = os.path.join(output_dir, "summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    gap()
    hr()
    say_ok(f"Ringkasan tersimpan: {md_path}")
    say_ok(f"Chart tersimpan di: {charts_dir}")


if __name__ == "__main__":
    main()
