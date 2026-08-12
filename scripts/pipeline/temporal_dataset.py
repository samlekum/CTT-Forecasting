# ./scripts/pipeline/temporal_dataset.py

import json
import os

import numpy as np
import pandas as pd


REQUIRED_CACHE_KEYS = {
    "data_matrix",
    "timeline_ns",
    "pixel_id",
    "lat_idx",
    "lon_idx",
    "latitude",
    "longitude",
}


def load_temporal_cache(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Temporal cache tidak ditemukan: {path}"
        )

    cache = np.load(
        path,
        allow_pickle=False,
    )

    missing = REQUIRED_CACHE_KEYS.difference(
        cache.files
    )

    if missing:
        raise ValueError(
            f"Cache tidak lengkap. Missing: {sorted(missing)}"
        )

    data_matrix = cache["data_matrix"]

    timeline = pd.to_datetime(
        cache["timeline_ns"]
    )

    pixel_meta = {
        "pixel_id": cache["pixel_id"],
        "lat_idx": cache["lat_idx"],
        "lon_idx": cache["lon_idx"],
        "latitude": cache["latitude"],
        "longitude": cache["longitude"],
    }

    return (
        data_matrix,
        timeline,
        pixel_meta,
    )


def validate_temporal_cache(
    data_matrix,
    timeline,
    pixel_meta,
    frequency_minutes=10,
):
    if data_matrix.ndim != 2:
        raise ValueError(
            f"data_matrix harus 2D: {data_matrix.shape}"
        )

    if data_matrix.shape[0] != len(timeline):
        raise ValueError(
            "Jumlah timestep data_matrix dan timeline berbeda."
        )

    if data_matrix.shape[1] != len(
        pixel_meta["pixel_id"]
    ):
        raise ValueError(
            "Jumlah pixel data_matrix dan metadata berbeda."
        )

    if len(timeline) == 0:
        raise ValueError(
            "Timeline kosong."
        )

    if not timeline.is_monotonic_increasing:
        raise ValueError(
            "Timeline tidak ascending."
        )

    if timeline.duplicated().any():
        raise ValueError(
            "Timeline memiliki duplicate timestamp."
        )

    if len(timeline) > 1:
        deltas = (
            np.diff(timeline.values)
            .astype("timedelta64[m]")
            .astype(np.int64)
        )

        unique_deltas = np.unique(
            deltas
        )

        if not (
            len(unique_deltas) == 1
            and unique_deltas[0]
            == frequency_minutes
        ):
            raise ValueError(
                "Interval timeline tidak konsisten: "
                f"{unique_deltas} menit."
            )

    if len(np.unique(
        pixel_meta["pixel_id"]
    )) != data_matrix.shape[1]:
        raise ValueError(
            "pixel_id tidak unik."
        )

    return True


def chronological_split(
    timeline,
    train_end,
    test_start,
    test_end,
):
    train_end = pd.Timestamp(
        train_end
    )

    test_start = pd.Timestamp(
        test_start
    )

    test_end = pd.Timestamp(
        test_end
    )

    if train_end >= test_start:
        raise ValueError(
            "TRAIN dan TEST overlap."
        )

    if test_start > test_end:
        raise ValueError(
            "test_start > test_end."
        )

    train_mask = timeline <= train_end

    test_mask = (
        (timeline >= test_start)
        & (timeline <= test_end)
    )

    if not train_mask.any():
        raise ValueError(
            "TRAIN kosong."
        )

    if not test_mask.any():
        raise ValueError(
            "TEST kosong."
        )

    if np.any(
        train_mask & test_mask
    ):
        raise ValueError(
            "TRAIN dan TEST overlap."
        )

    return (
        train_mask,
        test_mask,
    )


def slice_temporal_data(
    data_matrix,
    timeline,
    mask,
):
    return (
        data_matrix[mask],
        timeline[mask],
    )


def save_temporal_npz(
    path,
    data_matrix,
    timeline,
    pixel_meta,
):
    os.makedirs(
        os.path.dirname(
            os.path.abspath(path)
        ),
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        data_matrix=data_matrix.astype(
            np.float32
        ),
        timeline_ns=(
            timeline.values
            .astype("datetime64[ns]")
            .astype(np.int64)
        ),
        pixel_id=pixel_meta[
            "pixel_id"
        ],
        lat_idx=pixel_meta[
            "lat_idx"
        ],
        lon_idx=pixel_meta[
            "lon_idx"
        ],
        latitude=pixel_meta[
            "latitude"
        ],
        longitude=pixel_meta[
            "longitude"
        ],
    )


def save_metadata(
    path,
    metadata,
):
    os.makedirs(
        os.path.dirname(
            os.path.abspath(path)
        ),
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )