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


def recursive_rollout_predict(model, initial_windows, horizon_steps):
    """
    Rollout recursive BATCHED (semua anchor diprediksi bersamaan per
    step, bukan loop Python per anchor) -- model dipanggil persis
    horizon_steps kali, TIDAK peduli berapa banyak anchor-nya.

    Parameters
    ----------
    model : model ber-method .predict(X) -> (N,)
    initial_windows : np.ndarray, shape (N, window)
        Kolom 0 = lag_1 (observasi paling baru) ... kolom -1 = lag_w.
    horizon_steps : int

    Returns
    -------
    predictions : np.ndarray, shape (N, horizon_steps)
        predictions[:, k] = prediksi k+1 step ke depan dari window awal.
    """

    window_state = initial_windows.astype(np.float64).copy()
    n_anchors = window_state.shape[0]

    predictions = np.zeros((n_anchors, horizon_steps), dtype=np.float64)

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
            pred = np.asarray(model.predict(window_state), dtype=np.float64)

            predictions[:, step] = pred

            # geser window: prediksi baru jadi lag_1, sisanya geser ke kanan,
            # lag_w yang paling lama dibuang.
            window_state = np.concatenate(
                [pred.reshape(-1, 1), window_state[:, :-1]],
                axis=1,
            )

    return predictions


def recursive_rollout_mae(model, initial_windows, true_future, horizon_steps):
    """
    Jalankan recursive_rollout_predict() lalu hitung MAE per step.

    Returns
    -------
    mae_per_step : np.ndarray, shape (horizon_steps,)
    predictions : np.ndarray, shape (N, horizon_steps)
    """

    if initial_windows.shape[0] == 0:
        return (
            np.full(horizon_steps, np.nan),
            np.empty((0, horizon_steps)),
        )

    predictions = recursive_rollout_predict(
        model, initial_windows, horizon_steps
    )

    abs_errors = np.abs(predictions - true_future)
    mae_per_step = abs_errors.mean(axis=0)

    return mae_per_step, predictions


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
