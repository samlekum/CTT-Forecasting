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

import time

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
