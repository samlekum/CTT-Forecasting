# ./scripts/pipeline/window_model_training.py
#
# Training model + recursive rollout prediction untuk metode fixed-window
# baru (raw lag features, chronological split).
#
# SENGAJA TIDAK import dari pipeline/model_training.py (versi lama) --
# file itu masih terikat penuh ke metode expanding window & saat ini
# RUSAK di level import (Config.MIN_WINDOW_SIZE / Config.PURGE_STEPS
# sudah tidak ada lagi setelah config.py direvisi ke metode chronological).
# train_xgboost/train_lightgbm/train_catboost di bawah ini adalah
# hyperparameter yang SAMA (di-reuse konsepnya, bukan importnya) dengan
# versi lama supaya perbandingan hasil tetap adil.

import json
import os
import time
import warnings

import numpy as np
import pandas as pd

from pipeline.time_features import time_features_from_timestamps


def train_xgboost(X_train, y_train):
    import xgboost as xgb

    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_lightgbm(X_train, y_train):
    import lightgbm as lgb

    model = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(X_train, y_train)
    return model


def train_catboost(X_train, y_train):
    from catboost import CatBoostRegressor

    model = CatBoostRegressor(
        iterations=300,
        depth=6,
        learning_rate=0.05,
        random_seed=42,
        verbose=False,
    )
    model.fit(X_train, y_train)
    return model


# SVR tetap di-drop, konsisten dengan keputusan final metode lama
# (CLAUDE.md §5) -- terlalu lambat untuk ukuran dataset ini.
TRAINERS = {
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "catboost": train_catboost,
}


def train_one_model(model_name, X_train, y_train):
    """Dispatch ke fungsi training yang sesuai; return (model, detik_training)."""

    if model_name not in TRAINERS:
        raise KeyError(
            f"Model '{model_name}' tidak dikenal di TRAINERS "
            f"({list(TRAINERS.keys())})."
        )

    start = time.time()
    model = TRAINERS[model_name](X_train, y_train)
    elapsed = time.time() - start

    return model, elapsed


def recursive_rollout_predict(
    model,
    initial_windows,
    horizon_steps,
    anchor_time,
    freq_minutes=10,
    damping_rate=0.0,
    damping_cap=0.6,
):
    """
    Rollout recursive BATCHED (semua anchor diprediksi bersamaan per
    step, bukan loop Python per anchor) -- model dipanggil persis
    horizon_steps kali, TIDAK peduli berapa banyak anchor-nya.

    FITUR WAKTU (hour_sin/hour_cos, lihat pipeline/time_features.py):
    di tiap step, target_time = anchor_time + (step+1) * freq_minutes
    dihitung LANGSUNG dari anchor_time asli -- BUKAN dari window_state
    yang sudah bercampur prediksi. Ini yang bikin fitur waktu tetap
    valid persis di step manapun, walau window_state sendiri sudah jauh
    dari observasi asli (exposure bias). Fitur waktu di-concat SETELAH
    lag columns, urutan HARUS sama dengan
    pipeline/window_features.py::feature_column_names() yang dipakai
    saat training -- jangan ubah urutan di sini tanpa ubah di sana juga.

    DAMPING (opsional, default OFF supaya backward-compatible):
    Recursive rollout tanpa exogenous feature gampang "lari liar" makin
    jauh step-nya karena error step sebelumnya ikut jadi input step
    berikutnya (exposure bias). Kalau damping_rate > 0, tiap step prediksi
    ditarik sebagian ke `reference` (rata-rata window OBSERVASI ASLI di
    t0, bukan window yang sudah tercampur prediksi) -- bobot tarikannya
    (`alpha`) membesar linear per step sampai batas `damping_cap`:

        alpha_step  = min(damping_rate * step, damping_cap)
        pred_final  = (1 - alpha_step) * pred_raw + alpha_step * reference

    `reference` dihitung SEKALI dari initial_windows (persisten sepanjang
    rollout), bukan window_state yang terus berubah -- supaya damping
    menarik ke kondisi awal yang benar-benar teramati, bukan ke rata-rata
    prediksi sendiri yang mungkin sudah bias juga.

    damping_rate=0.0 -> identik dengan versi sebelum ada damping (tidak
    ada perubahan perilaku untuk caller yang belum di-update).

    Parameters
    ----------
    model : model ber-method .predict(X) -> (N,)
    initial_windows : np.ndarray, shape (N, window)
        Kolom 0 = lag_1 (observasi paling baru) ... kolom -1 = lag_w.
    horizon_steps : int
    anchor_time : array-like, shape (N,), datetime64[ns] (UTC)
        Timestamp t0 asli per baris -- dipakai buat hitung target_time
        tiap step (anchor_time + (step+1)*freq_minutes), BUKAN opsional,
        karena model sekarang dilatih dengan fitur waktu (lihat
        pipeline/time_features.py). Untuk model lama tanpa fitur waktu,
        panggil dengan window model_manifest yang sesuai.
    freq_minutes : int, default 10
        Resolusi timestep Himawari, dipakai buat hitung target_time.
    damping_rate : float, default 0.0
        Kenaikan alpha per step. 0.0 = tidak ada damping.
    damping_cap : float, default 0.6
        Batas atas alpha (jangan sampai pred_raw diabaikan sepenuhnya).

    Returns
    -------
    predictions : np.ndarray, shape (N, horizon_steps)
        predictions[:, k] = prediksi k+1 step ke depan dari window awal.
    """

    window_state = initial_windows.astype(np.float64).copy()
    n_anchors = window_state.shape[0]

    anchor_time = pd.DatetimeIndex(
        np.asarray(anchor_time, dtype="datetime64[ns]")
    )

    if len(anchor_time) != n_anchors:
        raise ValueError(
            "anchor_time harus punya panjang N sama dengan "
            f"initial_windows: len(anchor_time)={len(anchor_time)}, "
            f"n_anchors={n_anchors}."
        )

    predictions = np.zeros((n_anchors, horizon_steps), dtype=np.float64)

    reference = window_state.mean(axis=1) if damping_rate > 0 else None

    # Suppress "X does not have valid feature names, but LGBMRegressor was
    # fitted with feature names" (sklearn/LightGBM UserWarning, kosmetik --
    # cuma noise di terminal, TIDAK mempengaruhi hasil prediksi. Muncul
    # karena window_state di sini numpy array polos hasil geser-window per
    # step, bukan DataFrame). Scoped di block ini SAJA (bukan global) biar
    # warning lain yang mungkin penting tetap keliatan.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names",
            category=UserWarning,
        )

        for step in range(horizon_steps):
            target_time = anchor_time + pd.Timedelta(
                minutes=freq_minutes * (step + 1)
            )
            time_feats = time_features_from_timestamps(target_time.values)

            X_step = np.concatenate([window_state, time_feats], axis=1)

            pred = np.asarray(model.predict(X_step), dtype=np.float64)

            if damping_rate > 0:
                alpha = min(damping_rate * step, damping_cap)
                if alpha > 0:
                    pred = (1.0 - alpha) * pred + alpha * reference

            predictions[:, step] = pred

            # geser window: prediksi baru jadi lag_1, sisanya geser ke kanan,
            # lag_w yang paling lama dibuang. hour_sin/hour_cos TIDAK ikut
            # digeser -- selalu dihitung ULANG tiap step dari anchor_time,
            # itu justru poin utamanya (lihat docstring di atas).
            window_state = np.concatenate(
                [pred.reshape(-1, 1), window_state[:, :-1]],
                axis=1,
            )

    return predictions


def r2_score_per_step(predictions, true_future):
    """
    R^2 (koefisien determinasi) per step, dihitung ANTAR ANCHOR (semua
    pixel x semua anchor_t0 digabung jadi satu populasi per step) --
    konsisten dengan cara mae_per_step dihitung di recursive_rollout_mae
    (axis=0 = across anchor, per kolom step).

    R^2_step = 1 - SS_res / SS_tot
        SS_res = sum((y_true - y_pred)^2)
        SS_tot = sum((y_true - mean(y_true))^2)

    Tidak pakai sklearn.metrics.r2_score supaya konsisten dengan gaya
    modul ini (metrik lain di pipeline/window_eval.py juga ditulis
    manual pakai numpy) dan menghindari dependency tambahan.

    +1e-12 di penyebut (bukan 0) supaya tidak divide-by-zero kalau
    true_future di satu step kebetulan konstan (SS_tot=0) -- kasus ini
    R^2 secara definisi tidak bermakna, hasilnya jadi 0.0 (bukan NaN/inf)
    kalau prediksi juga persis sama (SS_res=0), atau sangat negatif kalau
    prediksi meleset dari nilai konstan itu.

    Parameters
    ----------
    predictions : np.ndarray, shape (N, horizon_steps)
    true_future : np.ndarray, shape (N, horizon_steps)

    Returns
    -------
    r2_per_step : np.ndarray, shape (horizon_steps,)
    """

    predictions = np.asarray(predictions, dtype=np.float64)
    true_future = np.asarray(true_future, dtype=np.float64)

    ss_res = np.sum((true_future - predictions) ** 2, axis=0)
    ss_tot = np.sum(
        (true_future - true_future.mean(axis=0, keepdims=True)) ** 2,
        axis=0,
    )

    return 1.0 - ss_res / (ss_tot + 1e-12)


def recursive_rollout_mae(
    model,
    initial_windows,
    true_future,
    horizon_steps,
    anchor_time,
    freq_minutes=10,
    damping_rate=0.0,
    damping_cap=0.6,
):
    """
    Jalankan recursive_rollout_predict() lalu hitung MAE per step DAN
    R^2 per step (lihat r2_score_per_step) -- dua-duanya dihitung dari
    hasil rollout yang sama, tidak ada rollout ganda.

    anchor_time/freq_minutes diteruskan apa adanya ke
    recursive_rollout_predict() -- dibutuhkan buat hitung fitur waktu
    (hour_sin/hour_cos) tiap step, lihat pipeline/time_features.py.

    damping_rate/damping_cap diteruskan apa adanya ke
    recursive_rollout_predict() -- default 0.0 = tanpa damping, sama
    seperti perilaku sebelumnya.

    Returns
    -------
    mae_per_step : np.ndarray, shape (horizon_steps,)
    predictions : np.ndarray, shape (N, horizon_steps)
    r2_per_step : np.ndarray, shape (horizon_steps,)
    """

    if initial_windows.shape[0] == 0:
        return (
            np.full(horizon_steps, np.nan),
            np.empty((0, horizon_steps)),
            np.full(horizon_steps, np.nan),
        )

    predictions = recursive_rollout_predict(
        model,
        initial_windows,
        horizon_steps,
        anchor_time=anchor_time,
        freq_minutes=freq_minutes,
        damping_rate=damping_rate,
        damping_cap=damping_cap,
    )

    abs_errors = np.abs(predictions - true_future)
    mae_per_step = abs_errors.mean(axis=0)

    r2_per_step = r2_score_per_step(predictions, true_future)

    return mae_per_step, predictions, r2_per_step


# ============================================================================
# Persist model final + manifest (Tahap 5, dipakai lagi oleh evaluasi TEST
# Tahap 6 & inference Tahap 7 -- satu tempat baca/tulis, jangan duplikat
# format file di script lain).
# ============================================================================

def save_model(model, path):
    """Simpan satu model (xgboost/lightgbm/catboost) ke .joblib.

    Format SAMA dengan pipeline/model_training.py (metode lama) --
    joblib.dump, bukan format native tiap library -- biar loader tetap
    satu (joblib.load) tidak peduli model_name-nya apa.
    """
    import joblib

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    joblib.dump(model, path)


def load_model(path):
    """Load satu model dari .joblib (kebalikan save_model())."""
    import joblib

    if not os.path.exists(path):
        raise FileNotFoundError(f"Model tidak ditemukan: {path}")
    return joblib.load(path)


def model_path_for(model_name, models_dir):
    """Path standar file model final: {models_dir}/{model_name}.joblib."""
    return os.path.join(models_dir, f"{model_name}.joblib")


def save_manifest(path, manifest):
    """Simpan model_manifest.json: {model_name: {window, model_path,
    feature_columns, n_train_samples, train_seconds, train_start,
    train_end}}.

    Manifest ini SATU-SATUNYA sumber kebenaran "model final pakai window
    berapa" untuk tahap sesudahnya (06_evaluate_test.py, 07_run_inference.py)
    -- supaya tidak ada tahap yang harus baca ulang window_search_results.csv
    dan berisiko salah asosiasi model<->window kalau window_search
    dijalankan ulang dengan kandidat window yang beda.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def load_manifest(path):
    """Load model_manifest.json (kebalikan save_manifest())."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model manifest tidak ditemukan: {path}\n"
            "Jalankan 05_train_final_models.py dulu."
        )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Versi feature set saat ini -- lag_1..lag_w + hour_sin/hour_cos (lihat
# pipeline/time_features.py). Manifest lama (sebelum fitur waktu
# ditambahkan) tidak punya field "feature_set" sama sekali -- kalau
# model itu di-load & di-rollout pakai recursive_rollout_predict() versi
# BARU (yang selalu concat time_feats), jumlah kolom X yang dikirim ke
# model.predict() akan MISMATCH dengan jumlah kolom saat model itu
# dilatih dulu -- kebanyakan library (xgboost/lightgbm/catboost) akan
# error jelas soal shape, TAPI beberapa kasus bisa diam-diam salah
# interpretasi kolom. Guard ini bikin errornya eksplisit & jelas
# pesannya, bukan nunggu error shape yang membingungkan dari library ML.
EXPECTED_FEATURE_SET = "lag_time_v1"


def check_feature_set(manifest_entry, model_name):
    """Pastikan entry manifest satu model punya feature_set yang cocok
    dengan versi recursive_rollout_predict() saat ini. Panggil ini
    SEBELUM rollout (06_evaluate_test.py, pipeline/inference.py) --
    bukan opsional, karena mismatch di sini silent-fail di beberapa
    kombinasi library/versi.
    """
    feature_set = manifest_entry.get("feature_set")
    if feature_set != EXPECTED_FEATURE_SET:
        raise ValueError(
            f"Model '{model_name}' punya feature_set="
            f"{feature_set!r} di manifest, tapi kode rollout saat ini "
            f"butuh feature_set={EXPECTED_FEATURE_SET!r} (lag + fitur "
            "waktu hour_sin/hour_cos). Model ini kemungkinan dilatih "
            "SEBELUM fitur waktu ditambahkan -- jalankan ulang "
            "04_search_window.py + 05_train_final_models.py dari nol "
            f"untuk '{model_name}' supaya manifest-nya konsisten."
        )