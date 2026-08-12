# ./scripts/03_prepare_temporal_dataset.py
#
# Tahap 3:
# Menyiapkan chronological train/test split dari temporal cache.
#
# INPUT:
#   dataset/expanding_raw_cache.npz
#
# OUTPUT:
#   dataset/temporal_split/
#       train_temporal.npz
#       test_temporal.npz
#       split_metadata.json
#
# DESAIN:
#   TRAIN : 2025-12-01 -> 2026-05-31
#   TEST  : 2026-06-01 -> 2026-07-31
#
# Tidak melakukan:
#   - stratified monthly split
#   - purge
#   - embargo
#   - model training
#   - window selection
#
# Window selection dilakukan pada tahap berikutnya.

import argparse
import json
import os

import numpy as np
import pandas as pd

from pipeline.config import load_config
from ui.terminal_display import (
    banner,
    gap,
    hr,
    say_error,
    say_info,
    say_ok,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare chronological train/test dataset."
    )

    parser.add_argument(
        "--cache",
        default=None,
        help=(
            "Path ke temporal cache .npz. "
            "Default: Config.EXPANDING_RAW_CACHE_FILE."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Folder output split. "
            "Default: dataset/temporal_split."
        ),
    )

    parser.add_argument(
        "--train-end",
        default="2026-05-31 23:59:59",
        help="Batas akhir TRAIN.",
    )

    parser.add_argument(
        "--test-start",
        default="2026-06-01 00:00:00",
        help="Awal TEST.",
    )

    parser.add_argument(
        "--test-end",
        default="2026-07-31 23:59:59",
        help="Akhir TEST.",
    )

    return parser.parse_args()


def load_temporal_cache(cache_path):
    if not os.path.exists(cache_path):
        raise FileNotFoundError(
            f"Temporal cache tidak ditemukan:\n{cache_path}\n\n"
            "Jalankan tahap 02 terlebih dahulu."
        )

    cache = np.load(
        cache_path,
        allow_pickle=False,
    )

    required_keys = {
        "data_matrix",
        "timeline_ns",
        "pixel_id",
        "lat_idx",
        "lon_idx",
        "latitude",
        "longitude",
    }

    missing = required_keys.difference(cache.files)

    if missing:
        raise ValueError(
            "Temporal cache tidak lengkap.\n"
            f"Key yang hilang: {sorted(missing)}"
        )

    data_matrix = cache["data_matrix"]
    timeline = pd.to_datetime(cache["timeline_ns"])

    pixel_meta = {
        "pixel_id": cache["pixel_id"],
        "lat_idx": cache["lat_idx"],
        "lon_idx": cache["lon_idx"],
        "latitude": cache["latitude"],
        "longitude": cache["longitude"],
    }

    return data_matrix, timeline, pixel_meta


def validate_temporal_cache(
    data_matrix,
    timeline,
    pixel_meta,
):
    say_info("Validasi temporal cache...")

    if data_matrix.ndim != 2:
        raise ValueError(
            f"data_matrix harus 2D, ditemukan {data_matrix.ndim}D."
        )

    n_timesteps, n_pixels = data_matrix.shape

    if len(timeline) != n_timesteps:
        raise ValueError(
            "Jumlah timestamp tidak sama dengan jumlah timestep."
        )

    if len(pixel_meta["pixel_id"]) != n_pixels:
        raise ValueError(
            "Jumlah pixel_id tidak sama dengan jumlah kolom data_matrix."
        )

    if len(timeline) == 0:
        raise ValueError("Timeline kosong.")

    if not timeline.is_monotonic_increasing:
        raise ValueError(
            "Timeline tidak terurut ascending."
        )

    if timeline.duplicated().any():
        raise ValueError(
            "Timeline memiliki timestamp duplikat."
        )

    if n_timesteps > 1:
        delta_minutes = (
            np.diff(timeline.values)
            .astype("timedelta64[m]")
            .astype(np.int64)
        )

        unique_delta = np.unique(delta_minutes)

        if not (
            len(unique_delta) == 1
            and unique_delta[0] == 10
        ):
            raise ValueError(
                "Interval temporal bukan 10 menit secara konsisten.\n"
                f"Interval ditemukan: {unique_delta}"
            )

    if len(np.unique(pixel_meta["pixel_id"])) != n_pixels:
        raise ValueError(
            "pixel_id tidak unik."
        )

    say_ok("Validasi temporal cache berhasil.")


def create_temporal_split(
    timeline,
    train_end,
    test_start,
    test_end,
):
    train_end = pd.Timestamp(train_end)
    test_start = pd.Timestamp(test_start)
    test_end = pd.Timestamp(test_end)

    if train_end >= test_start:
        raise ValueError(
            "TRAIN dan TEST overlap.\n"
            f"train_end={train_end}\n"
            f"test_start={test_start}"
        )

    if test_start > test_end:
        raise ValueError(
            "test_start lebih besar daripada test_end."
        )

    train_mask = timeline <= train_end

    test_mask = (
        (timeline >= test_start)
        & (timeline <= test_end)
    )

    if not train_mask.any():
        raise ValueError(
            "Tidak ada timestep yang masuk TRAIN."
        )

    if not test_mask.any():
        raise ValueError(
            "Tidak ada timestep yang masuk TEST."
        )

    if np.any(train_mask & test_mask):
        raise ValueError(
            "TRAIN dan TEST overlap."
        )

    return train_mask, test_mask


def save_temporal_split(
    output_dir,
    data_matrix,
    timeline,
    pixel_meta,
    train_mask,
    test_mask,
    train_end,
    test_start,
    test_end,
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    train_data = data_matrix[train_mask]
    train_timeline = timeline[train_mask]

    test_data = data_matrix[test_mask]
    test_timeline = timeline[test_mask]

    common_metadata = {
        "pixel_id": pixel_meta["pixel_id"],
        "lat_idx": pixel_meta["lat_idx"],
        "lon_idx": pixel_meta["lon_idx"],
        "latitude": pixel_meta["latitude"],
        "longitude": pixel_meta["longitude"],
    }

    train_path = os.path.join(
        output_dir,
        "train_temporal.npz",
    )

    test_path = os.path.join(
        output_dir,
        "test_temporal.npz",
    )

    np.savez_compressed(
        train_path,
        data_matrix=train_data.astype(np.float32),
        timeline_ns=(
            train_timeline
            .values
            .astype("datetime64[ns]")
            .astype(np.int64)
        ),
        **common_metadata,
    )

    np.savez_compressed(
        test_path,
        data_matrix=test_data.astype(np.float32),
        timeline_ns=(
            test_timeline
            .values
            .astype("datetime64[ns]")
            .astype(np.int64)
        ),
        **common_metadata,
    )

    metadata = {
        "method": "chronological_holdout",
        "frequency_minutes": 10,

        "requested_train_end": str(
            pd.Timestamp(train_end)
        ),
        "requested_test_start": str(
            pd.Timestamp(test_start)
        ),
        "requested_test_end": str(
            pd.Timestamp(test_end)
        ),

        "train_start": str(train_timeline[0]),
        "train_end": str(train_timeline[-1]),

        "test_start": str(test_timeline[0]),
        "test_end": str(test_timeline[-1]),

        "n_train_timesteps": int(
            len(train_timeline)
        ),
        "n_test_timesteps": int(
            len(test_timeline)
        ),

        "n_pixels": int(
            data_matrix.shape[1]
        ),

        "train_shape": list(
            train_data.shape
        ),
        "test_shape": list(
            test_data.shape
        ),
    }

    metadata_path = os.path.join(
        output_dir,
        "split_metadata.json",
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    return (
        train_path,
        test_path,
        metadata_path,
        metadata,
    )


def main():
    args = parse_args()

    cfg = load_config()

    cache_path = (
        args.cache
        or cfg.EXPANDING_RAW_CACHE_FILE
    )

    output_dir = (
        args.output_dir
        or os.path.join(
            cfg.PROJECT_ROOT,
            "dataset",
            "temporal_split",
        )
    )

    banner(
        "PREPARE TEMPORAL DATASET"
    )

    say_info(
        f"Cache input : {cache_path}"
    )

    say_info(
        f"Output dir  : {output_dir}"
    )

    say_info(
        "Split       : chronological"
    )

    say_info(
        f"Train end   : {args.train_end}"
    )

    say_info(
        f"Test start  : {args.test_start}"
    )

    say_info(
        f"Test end    : {args.test_end}"
    )

    hr()

    try:
        data_matrix, timeline, pixel_meta = (
            load_temporal_cache(cache_path)
        )

        say_info(
            f"Data matrix : {data_matrix.shape}"
        )

        say_info(
            f"Timeline    : "
            f"{timeline[0]} → {timeline[-1]}"
        )

        say_info(
            f"Pixel       : "
            f"{data_matrix.shape[1]}"
        )

        validate_temporal_cache(
            data_matrix,
            timeline,
            pixel_meta,
        )

        train_mask, test_mask = (
            create_temporal_split(
                timeline=timeline,
                train_end=args.train_end,
                test_start=args.test_start,
                test_end=args.test_end,
            )
        )

        (
            train_path,
            test_path,
            metadata_path,
            metadata,
        ) = save_temporal_split(
            output_dir=output_dir,
            data_matrix=data_matrix,
            timeline=timeline,
            pixel_meta=pixel_meta,
            train_mask=train_mask,
            test_mask=test_mask,
            train_end=args.train_end,
            test_start=args.test_start,
            test_end=args.test_end,
        )

        gap()

        banner("RINGKASAN")

        say_info(
            f"TRAIN timestep : "
            f"{metadata['n_train_timesteps']}"
        )

        say_info(
            f"TEST timestep  : "
            f"{metadata['n_test_timesteps']}"
        )

        say_info(
            f"Pixel          : "
            f"{metadata['n_pixels']}"
        )

        say_info(
            f"TRAIN          : "
            f"{metadata['train_start']} → "
            f"{metadata['train_end']}"
        )

        say_info(
            f"TEST           : "
            f"{metadata['test_start']} → "
            f"{metadata['test_end']}"
        )

        say_ok(
            f"TRAIN shape    : "
            f"{metadata['train_shape']}"
        )

        say_ok(
            f"TEST shape     : "
            f"{metadata['test_shape']}"
        )

        hr()

        say_ok(
            "Tahap 3 selesai."
        )

        say_info(
            f"TRAIN : {train_path}"
        )

        say_info(
            f"TEST  : {test_path}"
        )

        say_info(
            f"META  : {metadata_path}"
        )

    except Exception as exc:
        say_error(
            f"Tahap 3 gagal: {exc}"
        )
        raise


if __name__ == "__main__":
    main()