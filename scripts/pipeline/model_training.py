# ./scripts/pipeline/model_training.py
# Menyediakan fungsi-fungsi untuk memuat dataset expanding window, membagi
# data secara stratified per-bulan, melatih model (XGBoost/LightGBM/
# CatBoost), dan evaluasi dasar (MAE/RMSE/R2 keseluruhan, BUKAN per-step --
# evaluasi per-step ada di recursive_eval.py, Tahap 04, lihat CLAUDE.md §7).
#
# stratified_monthly_split() di bawah ini REUSE VERBATIM dari repo lama
# (nugrahsdhka/Bandung-Weather-Forecast-Himawari-09, scripts/pipeline/
# model_training.py) -- logikanya generic (parameterized time_col), jadi
# tidak ada modifikasi, cuma dipanggil dengan time_col="anchor_t0" (bukan
# "base_time" seperti di repo lama). Lihat CLAUDE.md §6 kenapa dipakai
# dari awal (bukan mulai dari chronological_split() biasa).

import time

import numpy as np
import pandas as pd

from pipeline.config import Config
from pipeline.expanding_features import FEATURE_COLUMNS
from ui.terminal_display import make_progress_bar, say_info, say_ok, say_error

TARGET_COLUMN = f"target_{Config.TARGET_CHANNEL}"


def load_expanding_dataset(path):
    """Baca dataset expanding window hasil dataset_builder.py, pastikan
    anchor_t0/target_time bertipe datetime."""
    df = pd.read_csv(path)
    df["anchor_t0"] = pd.to_datetime(df["anchor_t0"])
    df["target_time"] = pd.to_datetime(df["target_time"])
    return df


def stratified_monthly_split(df, test_frac=None, time_col="anchor_t0"):
    """
    Split train/test PER-BULAN: di dalam tiap bulan, ambil test_frac fraksi
    TERAKHIR secara kronologis sebagai test -- bukan ekor kronologis dari
    SELURUH dataset seperti chronological split biasa.

    REUSE VERBATIM dari repo lama (lihat header file). Kenapa dibutuhkan:
    kalau rentang data mencakup banyak bulan, ekor kronologis dari seluruh
    dataset bisa membuat beberapa bulan (terutama musim konvektif) 0%
    representasi di test -- model tidak pernah dievaluasi untuk kondisi
    bulan itu walau datanya penuh di training. Ini persis masalah yang
    ditemukan di repo lama dan sengaja dihindari dari awal di project ini
    (CLAUDE.md §6).

    Tetap kronologis DI DALAM tiap grup bulan (bukan random) supaya tidak
    ada kebocoran temporal dalam bulan yang sama -- cuma unit
    stratifikasinya yang berubah (per-bulan, bukan seluruh dataset).

    Parameters
    ----------
    test_frac : float, optional. Default None -> pakai Config.TEST_FRAC (0.15).
    time_col : str, default "anchor_t0" (beda dari repo lama yang pakai
        "base_time", karena skema kolom expanding window beda -- lihat
        dataset_builder.py).

    Return
    ------
    (train_df, test_df, cutoffs) -- `cutoffs` adalah dict
    {str(bulan): cutoff_time}.
    """
    if test_frac is None:
        test_frac = Config.TEST_FRAC

    df = df.copy()
    df["__bulan"] = df[time_col].dt.to_period("M")

    train_parts, test_parts = [], []
    cutoffs = {}

    for bulan, group in df.groupby("__bulan"):
        unique_times = sorted(group[time_col].unique())
        cutoff_idx = int(len(unique_times) * (1 - test_frac))
        # Guard bulan dengan timestamp sedikit: pastikan minimal ada 1 test
        # (kalau test_frac > 0) dan tidak keluar dari rentang list.
        cutoff_idx = min(max(cutoff_idx, 0), len(unique_times) - 1)
        cutoff_time = unique_times[cutoff_idx]
        cutoffs[str(bulan)] = cutoff_time

        train_parts.append(group[group[time_col] < cutoff_time])
        test_parts.append(group[group[time_col] >= cutoff_time])

    train_df = pd.concat(train_parts).drop(columns="__bulan").sort_values(time_col).reset_index(drop=True)
    test_df = pd.concat(test_parts).drop(columns="__bulan").sort_values(time_col).reset_index(drop=True)
    return train_df, test_df, cutoffs


def get_feature_target(df, feature_columns=None):
    """Ambil X (9 kolom fitur closed-form, lihat expanding_features.FEATURE_COLUMNS)
    dan y (target_tbb_13) dari dataframe."""
    X = df[feature_columns or FEATURE_COLUMNS].copy()
    y = df[TARGET_COLUMN].copy()
    return X, y


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


# SVR SUDAH DI-DROP (keputusan final, CLAUDE.md §5) -- terlalu lambat untuk
# ukuran training set expanding window per-pixel (potensi jutaan baris).
TRAINERS = {
    "xgboost": train_xgboost,
    "lightgbm": train_lightgbm,
    "catboost": train_catboost,
}


def train_one_model(model_name, X_train, y_train):
    """Dispatch ke fungsi training yang sesuai; return (model, detik_training).

    model_name harus salah satu dari Config.MODEL_NAMES (xgboost/lightgbm/
    catboost). Kalau ada nama lain (mis. sisa "svr" dari config lama),
    KeyError eksplisit -- bukan silent skip -- supaya ketauan cepat kalau
    ada inkonsistensi antara Config.MODEL_NAMES dan TRAINERS di sini.
    """
    if model_name not in TRAINERS:
        raise KeyError(
            f"Model '{model_name}' tidak dikenal di TRAINERS ({list(TRAINERS.keys())}). "
            "Cek Config.MODEL_NAMES di pipeline/config.py."
        )
    start = time.time()
    model = TRAINERS[model_name](X_train, y_train)
    elapsed = time.time() - start
    return model, elapsed


def evaluate(model, X_test, y_test):
    """Evaluasi dasar (MAE/RMSE/R2) di SELURUH test set (semua step
    dicampur). BUKAN evaluasi per-step -- itu tugas recursive_eval.py
    (Tahap 04), yang jalanin model secara recursive per t0 dan hitung MAE
    per step 1..18 secara terpisah (lihat CLAUDE.md §7)."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def train_all_models(df, models_dir=None, test_frac=None):
    """Fungsi utama: split (stratified monthly) -> train semua model di
    Config.MODEL_NAMES -> evaluasi dasar -> simpan model (.joblib) +
    ringkasan metrik (training_summary.csv) ke `models_dir`.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset hasil dataset_builder.build_dataset() / load_expanding_dataset().
    models_dir : str, optional. Default None -> Config.EXPANDING_MODELS_DIR.
    test_frac : float, optional. Default None -> Config.TEST_FRAC.

    Return
    ------
    summary_df : pd.DataFrame, satu baris per model, kolom: model, n_train,
        n_test, waktu_training_detik, mae, rmse, r2.
    """
    import os
    import joblib

    if models_dir is None:
        models_dir = Config.EXPANDING_MODELS_DIR
    os.makedirs(models_dir, exist_ok=True)

    train_df, test_df, cutoffs = stratified_monthly_split(df, test_frac=test_frac)
    X_train, y_train = get_feature_target(train_df)
    X_test, y_test = get_feature_target(test_df)
    say_info(f"Split selesai -- train: {len(train_df)} baris, test: {len(test_df)} baris "
             f"({len(cutoffs)} bulan).")

    summary_rows = []
    # Progress bar per-model -- training XGBoost/LightGBM/CatBoost bisa makan
    # waktu beberapa menit tergantung ukuran dataset, jadi WAJIB keliatan
    # progresnya (style sama kayak Tahap 1/2), bukan cuma diem nunggu.
    model_progress = make_progress_bar(Config.MODEL_NAMES, desc="Training model", unit="model")
    for model_name in model_progress:
        model_progress.set_postfix_str(model_name)
        say_info(f"Mulai training: {model_name} ...")
        try:
            model, elapsed = train_one_model(model_name, X_train, y_train)
            metrics = evaluate(model, X_test, y_test)

            model_path = os.path.join(models_dir, f"{model_name}.joblib")
            joblib.dump(model, model_path)

            say_ok(
                f"{model_name} selesai ({elapsed:.1f}s) -- "
                f"MAE={metrics['mae']:.4f}K RMSE={metrics['rmse']:.4f}K R2={metrics['r2']:.4f}"
            )

            summary_rows.append({
                "model": model_name,
                "n_train": len(train_df),
                "n_test": len(test_df),
                "waktu_training_detik": round(elapsed, 1),
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "r2": metrics["r2"],
            })
        except ImportError as e:
            say_error(f"{model_name} dilewati -- library belum terinstall: {e}")
            summary_rows.append({
                "model": model_name, "n_train": len(train_df), "n_test": len(test_df),
                "waktu_training_detik": None, "mae": None, "rmse": None, "r2": None,
                "error": f"Library belum terinstall: {e}",
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(models_dir, "training_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    return summary_df