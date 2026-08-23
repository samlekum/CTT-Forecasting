# ./scripts/pipeline/time_features.py
#
# Fitur waktu siklikal (diurnal) untuk metode window search.
#
# LATAR BELAKANG:
# window_features.py (lag_1..lag_w) cuma berisi riwayat nilai TBB mentah.
# Saat recursive rollout sudah 15-18 step (2.5-3 jam) ke depan, window
# state didominasi prediksi-dari-prediksi sendiri (exposure bias) --
# model kehilangan sinyal "sekarang jam berapa, harusnya siang/malam,
# harusnya konvektif atau tidak". Modul ini nambahin fitur EXOGENOUS yang
# SELALU diketahui pasti di setiap step manapun, terlepas dari akurasi
# rollout -- karena target_time tiap step dihitung langsung dari
# anchor_time + step * FREQ_MINUTES, bukan dari window yang mungkin sudah
# tercemar prediksi.
#
# TIMEZONE -- PENTING:
# Timestamp Himawari (lihat pipeline/netcdf_tools.py::extract_time_from_filename)
# adalah UTC APA ADANYA, TIDAK ada konversi timezone di manapun di
# pipeline ini. Siklus diurnal awan konvektif di Bandung terikat ke jam
# MATAHARI LOKAL (WIB, UTC+7), BUKAN UTC. Encode jam UTC mentah bikin
# sinyalnya salah kaprah (mis. "07:00" di data ini sebenarnya jam 14:00
# siang WIB, bukan pagi) -- makanya WAJIB +7 jam dulu sebelum di-encode.
#
# ENCODING:
# Cyclical sin/cos dari menit-dalam-hari lokal (0..1439), BUKAN integer
# jam mentah -- supaya jam 23:50 dan 00:10 "dekat" secara numerik di
# ruang fitur, bukan terpisah jauh (23 vs 0) seperti kalau pakai angka
# jam linear.

import numpy as np
import pandas as pd


WIB_OFFSET_HOURS = 7  # Himawari timestamp = UTC, Bandung = WIB (UTC+7)

TIME_FEATURE_COLUMNS = ["hour_sin", "hour_cos"]


def time_features_from_timestamps(timestamps):
    """
    Hitung fitur cyclical [hour_sin, hour_cos] dari timestamp UTC.

    Parameters
    ----------
    timestamps : array-like
        Timestamp UTC (native Himawari), bisa berupa np.ndarray
        datetime64[ns], pd.DatetimeIndex, atau pd.Series.

    Returns
    -------
    np.ndarray, shape (N, 2), dtype float64
        Kolom 0 = hour_sin, kolom 1 = hour_cos -- konsisten dengan
        TIME_FEATURE_COLUMNS. Dihitung dari jam LOKAL WIB (UTC+7),
        bukan UTC mentah.
    """
    ts = pd.DatetimeIndex(
        np.asarray(timestamps, dtype="datetime64[ns]")
    ) + pd.Timedelta(hours=WIB_OFFSET_HOURS)

    minute_of_day = (ts.hour * 60 + ts.minute).to_numpy(dtype=np.float64)

    angle = 2.0 * np.pi * minute_of_day / 1440.0

    return np.stack(
        [np.sin(angle), np.cos(angle)],
        axis=1,
    )