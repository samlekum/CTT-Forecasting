# ./scripts/pipeline/netcdf_tools.py
# Menyediakan utilitas untuk memproses file NetCDF, meliputi validasi hasil unduhan, ekstraksi waktu observasi, dan pembuatan subset wilayah Bandung.

import os
import re
import logging
import xarray as xr

from ui.terminal_display import say_ok, say_skip
from pipeline.config import Config
from datetime import datetime

FILENAME_PATTERN = re.compile(r"NC_H\d+_(\d{8})_(\d{4})_")


def subset_bandung(temp_file, target_path):
    """Subset file NetCDF ke area Bandung; return True kalau ada data, raise ValueError kalau struktur lat/lon tidak sesuai (non-retryable)."""
    with xr.open_dataset(temp_file) as ds:
        if "latitude" not in ds.coords and "latitude" not in ds.variables:
            raise ValueError(f"Variabel 'latitude' tidak ditemukan di {temp_file}")
        if "longitude" not in ds.coords and "longitude" not in ds.variables:
            raise ValueError(f"Variabel 'longitude' tidak ditemukan di {temp_file}")

        mask_lat = (ds.latitude >= Config.BANDUNG_LAT_MIN) & (
            ds.latitude <= Config.BANDUNG_LAT_MAX
        )
        mask_lon = (ds.longitude >= Config.BANDUNG_LON_MIN) & (
            ds.longitude <= Config.BANDUNG_LON_MAX
        )
        subset = ds.where(mask_lat & mask_lon, drop=True)

        if subset.latitude.size > 0:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            subset.to_netcdf(target_path)
            logging.info(f"BERHASIL DIPROSES: {os.path.basename(target_path)}")
            say_ok(f"Data Bandung ditemukan → disimpan: {os.path.basename(target_path)}")
            return True
        else:
            logging.info(f"TIDAK ADA DATA BANDUNG: {os.path.basename(target_path)}")
            say_skip("Area Bandung tidak ditemukan di file ini")
            return False


def validate_downloaded_file(filepath, min_size=10000):
    """Validasi file hasil download tidak korup (cek ukuran minimum)."""
    if os.path.getsize(filepath) < min_size:
        raise Exception("Ukuran file terlalu kecil, kemungkinan korup")


def extract_time_from_filename(filename):
    """Ambil waktu observasi dari nama file Himawari."""
    match = FILENAME_PATTERN.search(filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
    return None