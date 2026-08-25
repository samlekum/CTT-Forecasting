# ./scripts/pipeline/kfold_eval.py
#
# Analisis statistik tambahan: BLOCKED K-FOLD CROSS VALIDATION (k=5) di atas
# periode TRAIN (Des'25-Mei'26), plus timing & peak-memory per model.
#
# BUKAN pengganti split kronologis utama (03/04/05/06) -- ini analisis
# TAMBAHAN untuk mengukur SEBERAPA STABIL performa tiap model lintas
# potongan waktu yang berbeda (statistical robustness), karena satu split
# tunggal (FIT/VALIDATION di 04_search_window.py) cuma memberi SATU angka
# MAE per model, tidak ada info variansi/confidence interval.
#
# KENAPA "BLOCKED" (bukan KFold acak sklearn biasa):
# Data ini deret waktu dengan autokorelasi tinggi antar timestep yang
# berdekatan (interval 10 menit). KFold acak akan mencampur baris yang
# nyaris identik antara train & validation fold (leakage temporal
# terselubung -- skor jadi optimis palsu). Solusinya: potong TRAIN jadi 5
# blok waktu BERURUTAN (bukan diacak), lalu tiap fold pakai satu blok
# sebagai VALIDATION dan sisanya sebagai TRAINING -- disebut "blocked" /
# "purged" k-fold di literatur time-series CV (mis. Lopez de Prado, 2018).
#
# PURGE GAP:
# Sample training yang dibangun dari lag window bisa "mengintip" ke
# validation fold kalau posisinya persis di perbatasan blok (window
# lag_1..lag_w mencakup beberapa timestep dari fold lain). Untuk itu,
# baris TRAIN yang jatuh dalam radius (window + horizon_steps) langkah
# dari batas fold validation DIBUANG (purge) -- bukan dipakai training
# ataupun validation. Ini analog dengan purge/embargo yang dipakai metode
# LAMA (CTT-Forecasting-Expanding), tapi di sini scope-nya cuma untuk
# analisis k-fold ini, bukan menggantikan split utama TRAIN/TEST.

import time
import tracemalloc

import numpy as np
import pandas as pd
from scipy import stats

from pipeline.window_features import (
    build_window_dataset,
    build_rollout_arrays_with_anchors,
    feature_column_names,
    TARGET_COLUMN,
)
from pipeline.window_model_training import train_one_model, recursive_rollout_mae


def make_blocked_folds(timeline, n_folds=5):
    """
    Bagi `timeline` (sudah terurut ascending, hasil slice periode TRAIN)
    jadi n_folds blok waktu berurutan dengan panjang SAMA (baris terakhir
    yang tidak habis dibagi masuk ke blok terakhir).

    Returns
    -------
    list of (start_idx, end_idx) -- end_idx EKSKLUSIF, dipakai untuk
    slicing timeline/data_matrix. len(list) == n_folds.
    """
    T = len(timeline)
    if T < n_folds:
        raise ValueError(
            f"Timeline terlalu pendek ({T} timestep) untuk {n_folds}-fold."
        )

    block_size = T // n_folds
    bounds = []
    start = 0
    for k in range(n_folds):
        end = T if k == n_folds - 1 else start + block_size
        bounds.append((start, end))
        start = end
    return bounds


def purge_mask_for_fold(T, val_start, val_end, purge_radius):
    """
    Mask boolean shape (T,), True untuk index yang BOLEH dipakai training
    fold ini -- yaitu SEMUA index DI LUAR [val_start, val_end) DAN di luar
    radius purge_radius di kedua sisi batas validation block.
    """
    idx = np.arange(T)
    in_val = (idx >= val_start) & (idx < val_end)
    near_val = (
        (idx >= val_start - purge_radius) & (idx < val_end + purge_radius)
    )
    return ~near_val, in_val


def _peak_memory_mb(fn, *args, **kwargs):
    """Jalankan fn(*args, **kwargs), ukur waktu wall-clock DAN peak memory
    tambahan (bukan RSS total proses) selama fn berjalan, pakai
    tracemalloc (portable, tidak butuh psutil/resource yang beda-beda
    perilakunya antar OS).

    Returns
    -------
    result : apapun yang di-return fn
    elapsed_seconds : float
    peak_mb : float -- puncak memori Python yang dialokasikan SELAMA fn
        berjalan (di atas baseline sebelum fn dipanggil), dalam MB.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    elapsed = time.perf_counter() - t0
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak / (1024 * 1024)


def run_kfold_statistical_analysis(
    data_matrix,
    timeline,
    pixel_meta,
    model_names,
    window,
    horizon_steps,
    n_folds=5,
    freq_minutes=10,
):
    """
    Jalankan blocked k-fold CV untuk SATU window tertentu (biasanya window
    terbaik hasil 04_search_window.py per model -- dipanggil sekali per
    model dengan window masing-masing, bukan window sama untuk semua
    model), catat MAE/RMSE/R2 per fold + waktu training + peak memory.

    Returns
    -------
    pd.DataFrame, satu baris per (model, fold), kolom:
        model, fold, n_train_samples, n_val_anchors,
        mae_avg_all_steps, rmse_avg_all_steps, r2_avg_all_steps,
        train_seconds, train_peak_memory_mb
    """
    T = len(timeline)
    purge_radius = window + horizon_steps

    fold_bounds = make_blocked_folds(timeline, n_folds=n_folds)

    rows = []

    for fold_idx, (val_start, val_end) in enumerate(fold_bounds, start=1):
        train_mask, val_mask = purge_mask_for_fold(
            T, val_start, val_end, purge_radius
        )

        train_data = data_matrix[train_mask]
        train_timeline = timeline[train_mask]

        val_data = data_matrix[val_mask]
        val_timeline = timeline[val_mask]

        if len(val_timeline) < window + horizon_steps:
            # blok validation terlalu pendek buat window+horizon ini -- skip
            # (bisa terjadi kalau n_folds besar relatif ke panjang TRAIN).
            continue

        fit_df = build_window_dataset(
            train_data, train_timeline, pixel_meta, window=window
        )

        if len(fit_df) == 0:
            continue

        feature_cols = feature_column_names(window)
        X_train = fit_df[feature_cols].values
        y_train = fit_df[TARGET_COLUMN].values

        val_windows, val_future, _pixel_ids, val_anchor_time = (
            build_rollout_arrays_with_anchors(
                val_data,
                val_timeline,
                pixel_meta,
                window=window,
                horizon_steps=horizon_steps,
            )
        )

        if len(val_windows) == 0:
            continue

        for model_name in model_names:
            # train_one_model() mengembalikan (model, elapsed_internal) --
            # elapsed_internal-nya DIABAIKAN di sini, dipakai timing dari
            # _peak_memory_mb() (perf_counter di LUAR fungsi) supaya satu
            # sumber waktu yang konsisten dipakai untuk SEMUA baris hasil
            # (termasuk kalau nanti fungsi lain selain train_one_model mau
            # diukur pakai helper yang sama).
            (model, _elapsed_internal), train_elapsed, peak_mb = _peak_memory_mb(
                train_one_model, model_name, X_train, y_train
            )

            mae_per_step, predictions, r2_per_step = recursive_rollout_mae(
                model,
                val_windows,
                val_future,
                horizon_steps,
                anchor_time=val_anchor_time,
                freq_minutes=freq_minutes,
            )

            rmse_per_step = np.sqrt(
                np.mean((predictions - val_future) ** 2, axis=0)
            )

            rows.append({
                "model": model_name,
                "fold": fold_idx,
                "n_train_samples": len(X_train),
                "n_val_anchors": len(val_windows),
                "mae_avg_all_steps": float(np.nanmean(mae_per_step)),
                "rmse_avg_all_steps": float(np.nanmean(rmse_per_step)),
                "r2_avg_all_steps": float(np.nanmean(r2_per_step)),
                "train_seconds": train_elapsed,
                "train_peak_memory_mb": peak_mb,
            })

    return pd.DataFrame(rows)


def summarize_kfold(kfold_df):
    """
    Ringkas hasil kfold_df (satu baris per model x fold) jadi statistik
    per model: mean, std, 95% CI (t-distribution, df=n_folds-1),
    coefficient of variation (CV = std/mean, indikator stabilitas -- makin
    kecil makin stabil lintas fold).

    Returns
    -------
    pd.DataFrame, satu baris per model.
    """
    rows = []
    for model_name, group in kfold_df.groupby("model"):
        n = len(group)
        mae = group["mae_avg_all_steps"].values
        mean_mae = mae.mean()
        std_mae = mae.std(ddof=1) if n > 1 else 0.0
        sem = std_mae / np.sqrt(n) if n > 1 else 0.0
        # t-distribution 95% CI (bukan normal/z) karena n_folds kecil (5)
        t_crit = stats.t.ppf(0.975, df=max(n - 1, 1))
        ci_half_width = t_crit * sem

        rows.append({
            "model": model_name,
            "n_folds": n,
            "mae_mean": mean_mae,
            "mae_std": std_mae,
            "mae_cv_percent": (std_mae / mean_mae * 100.0) if mean_mae else np.nan,
            "mae_ci95_low": mean_mae - ci_half_width,
            "mae_ci95_high": mean_mae + ci_half_width,
            "r2_mean": group["r2_avg_all_steps"].mean(),
            "r2_std": group["r2_avg_all_steps"].std(ddof=1) if n > 1 else 0.0,
            "train_seconds_mean": group["train_seconds"].mean(),
            "train_seconds_std": group["train_seconds"].std(ddof=1) if n > 1 else 0.0,
            "train_peak_memory_mb_mean": group["train_peak_memory_mb"].mean(),
            "train_peak_memory_mb_std": group["train_peak_memory_mb"].std(ddof=1) if n > 1 else 0.0,
        })

    return pd.DataFrame(rows).sort_values("mae_mean").reset_index(drop=True)


def pairwise_significance(kfold_df, metric="mae_avg_all_steps"):
    """
    Uji signifikansi statistik antar PASANGAN model, dari nilai per-fold
    yang SAMA fold-nya (paired test -- setiap model dievaluasi di fold
    yang identik, jadi variasi antar-fold bisa "dihilangkan" lewat
    pairing, meningkatkan power test dibanding unpaired test).

    Dua uji dilaporkan sekaligus:
      - Paired t-test (parametrik, asumsi selisih berdistribusi normal)
      - Wilcoxon signed-rank (non-parametrik, lebih aman untuk n=5 kecil
        yang sulit dipastikan normal)

    H0: tidak ada beda rata-rata performa (metric) antara dua model.
    p < 0.05 -> tolak H0, beda dianggap signifikan secara statistik.

    Returns
    -------
    pd.DataFrame, satu baris per pasangan model.
    """
    pivot = kfold_df.pivot(index="fold", columns="model", values=metric)
    models = list(pivot.columns)

    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            paired = pivot[[a, b]].dropna()
            if len(paired) < 2:
                continue

            diff = paired[a].values - paired[b].values

            t_stat, p_t = stats.ttest_rel(paired[a], paired[b])

            try:
                w_stat, p_w = stats.wilcoxon(paired[a], paired[b])
            except ValueError:
                # semua selisih nol -- wilcoxon tidak terdefinisi
                w_stat, p_w = np.nan, np.nan

            pooled_std = diff.std(ddof=1) if len(diff) > 1 else np.nan
            cohens_d = (
                diff.mean() / pooled_std if pooled_std else np.nan
            )

            rows.append({
                "model_a": a,
                "model_b": b,
                "mean_diff": diff.mean(),
                "t_stat": t_stat,
                "p_value_ttest": p_t,
                "wilcoxon_stat": w_stat,
                "p_value_wilcoxon": p_w,
                "cohens_d": cohens_d,
                "significant_5pct": bool(p_t < 0.05),
            })

    return pd.DataFrame(rows)