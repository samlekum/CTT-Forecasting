# ./scripts/pipeline/window_eval.py
#
# Evaluasi FINAL di TEST set (Jun-Jul'26) untuk metode window search
# (fixed-size window per model, lihat pipeline/window_features.py &
# pipeline/window_model_training.py).
#
# PENTING: modul ini SENGAJA dibangun dari NOL, TIDAK mengimpor apapun
# dari pipeline/recursive_eval.py atau pipeline/model_training.py --
# keduanya versi expanding-window LAMA yang sudah rusak di level import
# (Config.MIN_WINDOW_SIZE / Config.PURGE_STEPS sudah dihapus dari
# config.py, lihat catatan sesi). Definisi & naming spatial_collapse_ratio
# / spatial_correlation di bawah ini disamakan SECARA KONSEP dengan versi
# lama itu (biar angka bisa dibandingkan apple-to-apple), tapi kodenya
# ditulis ulang total supaya tidak ikut ke-import modul yang rusak.

import numpy as np
import pandas as pd


def spatial_collapse_ratio(preds, actuals):
    """std(prediksi) / std(aktual) ANTAR PIXEL, pada satu anchor_t0 & step
    yang sama. Mendekati 0 berarti model kolaps jadi rata secara spasial
    dan kehilangan variasi antar pixel -- metrik ini yang langsung
    menjawab masalah utama exposure bias / spatial collapse yang jadi
    concern utama sesi-sesi sebelumnya. MAE saja TIDAK bisa mendeteksi
    ini (model bisa MAE rendah tapi flat/rata secara spasial).
    """
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    return float(np.std(preds) / (np.std(actuals) + 1e-6))


def spatial_correlation(preds, actuals):
    """Korelasi spasial prediksi vs aktual ANTAR PIXEL pada satu anchor_t0
    & step yang sama. Dipakai bareng spatial_collapse_ratio -- ratio bisa
    tinggi (variasi terjaga) tapi korelasi rendah (variasinya di tempat
    yang salah / pola spasial nggak match).
    """
    preds = np.asarray(preds, dtype=float)
    actuals = np.asarray(actuals, dtype=float)
    if len(preds) < 2 or np.std(preds) < 1e-9 or np.std(actuals) < 1e-9:
        return 0.0
    return float(np.corrcoef(preds, actuals)[0, 1])


def _spatial_metrics_per_step(detail_df):
    """Untuk tiap (model, step, anchor_t0) yang punya >= 2 pixel (butuh
    minimal 2 titik buat hitung std/korelasi antar pixel yang bermakna),
    hitung spatial_collapse_ratio & spatial_correlation, lalu rata-ratakan
    lintas anchor_t0 per (model, step). anchor_t0 dengan cuma 1 pixel
    (gap-skip per-pixel bisa bikin nggak semua pixel valid di anchor_t0
    yang sama) di-skip dari perhitungan ini -- bukan bug, cuma nggak ada
    "antar pixel" buat dibandingin.
    """
    rows = []
    for (model, step, t0), group in detail_df.groupby(
        ["model", "step", "anchor_t0"]
    ):
        if len(group) < 2:
            continue
        rows.append({
            "model": model,
            "step": step,
            "anchor_t0": t0,
            "collapse_ratio": spatial_collapse_ratio(
                group["y_pred"].values, group["y_true"].values
            ),
            "correlation": spatial_correlation(
                group["y_pred"].values, group["y_true"].values
            ),
        })

    if not rows:
        # Nggak ada anchor_t0 dengan >=2 pixel sama sekali (mis. dataset
        # uji sintetis 1 pixel) -- return kolom kosong yang konsisten,
        # jangan crash.
        return pd.DataFrame(
            columns=[
                "model",
                "step",
                "spatial_collapse_ratio",
                "spatial_correlation",
                "n_t0_groups",
            ]
        )

    per_t0 = pd.DataFrame(rows)
    summary = per_t0.groupby(["model", "step"]).agg(
        spatial_collapse_ratio=("collapse_ratio", "mean"),
        spatial_correlation=("correlation", "mean"),
        n_t0_groups=("collapse_ratio", "size"),
    ).reset_index()
    return summary


def summarize_by_step(detail_df):
    """Ringkas MAE per (model, step), plus metrik spasial (lihat
    _spatial_metrics_per_step) -- indikator langsung buat ngecek gejala
    'spatial collapse' selama recursive rollout di TEST set. Murni
    ringkasan deskriptif dari detail_df yang sudah ada.
    """
    grouped = detail_df.groupby(["model", "step"])
    mae_summary = grouped.agg(
        mae=("abs_error", "mean"),
        rmse=("abs_error", lambda s: np.sqrt(np.mean(s ** 2))),
        n_samples=("abs_error", "size"),
        pred_std=("y_pred", "std"),
        true_std=("y_true", "std"),
    ).reset_index()

    spatial_summary = _spatial_metrics_per_step(detail_df)
    summary = mae_summary.merge(
        spatial_summary, on=["model", "step"], how="left"
    )
    return summary.sort_values(["model", "step"]).reset_index(drop=True)


def build_detail_df(
    model_name,
    predictions,
    true_future,
    pixel_ids,
    anchor_time,
    frequency_minutes=10,
):
    """
    Ubah hasil recursive_rollout_predict() (wide: shape (N, horizon_steps))
    jadi detail_df long-format, satu baris per (anchor, step) -- format
    sama seperti versi lama (recursive_eval.py) supaya summarize_by_step()
    & _spatial_metrics_per_step() di atas bisa langsung dipakai.

    Parameters
    ----------
    model_name : str
    predictions : np.ndarray, shape (N, horizon_steps)
        Hasil recursive_rollout_predict() (pipeline/window_model_training.py).
    true_future : np.ndarray, shape (N, horizon_steps)
        true_future dari build_rollout_arrays_with_anchors().
    pixel_ids : np.ndarray, shape (N,)
    anchor_time : np.ndarray, shape (N,), dtype datetime64[ns]
    frequency_minutes : int, default 10
        Resolusi timestep Himawari, dipakai buat hitung target_time
        (anchor_time + step * frequency_minutes).

    Returns
    -------
    detail_df : pd.DataFrame, kolom:
        model, pixel_id, anchor_t0, step, target_time, y_true, y_pred,
        abs_error.
        Satu baris per (anchor, step) -- long-format, konsisten dengan
        recursive_eval.py versi lama (minus y_pred_raw, karena window
        search TIDAK pakai damping -- lihat window_model_training.py).
    """

    predictions = np.asarray(predictions, dtype=float)
    true_future = np.asarray(true_future, dtype=float)

    if predictions.shape != true_future.shape:
        raise ValueError(
            "predictions dan true_future harus punya shape sama: "
            f"{predictions.shape} vs {true_future.shape}"
        )

    n_anchor, horizon_steps = predictions.shape
    pixel_ids = np.asarray(pixel_ids)
    anchor_time = np.asarray(anchor_time, dtype="datetime64[ns]")

    if len(pixel_ids) != n_anchor or len(anchor_time) != n_anchor:
        raise ValueError(
            "pixel_ids/anchor_time harus punya panjang N sama dengan "
            f"predictions: n_anchor={n_anchor}, "
            f"len(pixel_ids)={len(pixel_ids)}, "
            f"len(anchor_time)={len(anchor_time)}"
        )

    steps = np.arange(1, horizon_steps + 1)

    # Long-format lewat broadcasting, bukan loop Python per (anchor, step)
    # -- N bisa besar (semua anchor TEST, anchor_stride=1).
    model_col = np.full(n_anchor * horizon_steps, model_name)
    pixel_col = np.repeat(pixel_ids, horizon_steps)
    anchor_col = np.repeat(anchor_time, horizon_steps)
    step_col = np.tile(steps, n_anchor)
    target_time_col = anchor_col + step_col.astype("timedelta64[m]") * frequency_minutes
    y_true_col = true_future.reshape(-1)
    y_pred_col = predictions.reshape(-1)

    detail_df = pd.DataFrame({
        "model": model_col,
        "pixel_id": pixel_col,
        "anchor_t0": anchor_col,
        "step": step_col,
        "target_time": target_time_col,
        "y_true": y_true_col,
        "y_pred": y_pred_col,
        "abs_error": np.abs(y_pred_col - y_true_col),
    })

    return detail_df
