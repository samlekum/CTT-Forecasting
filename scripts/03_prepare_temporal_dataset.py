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
# Window selection dilakukan pada tahap berikutnya (04_search_window.py).
#
# Catatan refactor:
# Seluruh logic load/validate/split/save sekarang di pipeline/temporal_dataset.py
# (dipakai bareng oleh tahap ini dan tahap window-search berikutnya, biar
# gak ada 2 salinan logic yang sama). Script ini cuma CLI wrapper + summary
# terminal.

import argparse
import os

from pipeline.config import load_config
from pipeline.temporal_dataset import (
    load_temporal_cache,
    validate_temporal_cache,
    chronological_split,
    slice_temporal_data,
    save_temporal_npz,
    save_metadata,
)
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
        help="Folder output split. Default: Config.TEMPORAL_SPLIT_DIR.",
    )

    parser.add_argument(
        "--train-end",
        default=None,
        help="Batas akhir TRAIN. Default: Config.TRAIN_END.",
    )

    parser.add_argument(
        "--test-start",
        default=None,
        help="Awal TEST. Default: Config.TEST_START.",
    )

    parser.add_argument(
        "--test-end",
        default=None,
        help="Akhir TEST. Default: Config.TEST_END.",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    cache_path = args.cache or cfg.EXPANDING_RAW_CACHE_FILE
    output_dir = args.output_dir or cfg.TEMPORAL_SPLIT_DIR

    train_end = args.train_end or cfg.TRAIN_END
    test_start = args.test_start or cfg.TEST_START
    test_end = args.test_end or cfg.TEST_END

    banner("PREPARE TEMPORAL DATASET")

    say_info(f"Cache input : {cache_path}")
    say_info(f"Output dir  : {output_dir}")
    say_info("Split       : chronological (tanpa purge/embargo)")
    say_info(f"Train end   : {train_end}")
    say_info(f"Test start  : {test_start}")
    say_info(f"Test end    : {test_end}")

    hr()

    try:
        data_matrix, timeline, pixel_meta = load_temporal_cache(cache_path)

        say_info(f"Data matrix : {data_matrix.shape}")
        say_info(f"Timeline    : {timeline[0]} \u2192 {timeline[-1]}")
        say_info(f"Pixel       : {data_matrix.shape[1]}")

        validate_temporal_cache(
            data_matrix,
            timeline,
            pixel_meta,
            frequency_minutes=cfg.FREQ_MINUTES,
        )

        say_ok("Validasi temporal cache berhasil.")

        train_mask, test_mask = chronological_split(
            timeline=timeline,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )

        train_data, train_timeline = slice_temporal_data(
            data_matrix, timeline, train_mask
        )

        test_data, test_timeline = slice_temporal_data(
            data_matrix, timeline, test_mask
        )

        train_path = os.path.join(output_dir, "train_temporal.npz")
        test_path = os.path.join(output_dir, "test_temporal.npz")
        metadata_path = os.path.join(output_dir, "split_metadata.json")

        save_temporal_npz(train_path, train_data, train_timeline, pixel_meta)
        save_temporal_npz(test_path, test_data, test_timeline, pixel_meta)

        metadata = {
            "method": "chronological_holdout",
            "frequency_minutes": cfg.FREQ_MINUTES,

            "requested_train_end": str(train_end),
            "requested_test_start": str(test_start),
            "requested_test_end": str(test_end),

            "train_start": str(train_timeline[0]),
            "train_end": str(train_timeline[-1]),

            "test_start": str(test_timeline[0]),
            "test_end": str(test_timeline[-1]),

            "n_train_timesteps": int(len(train_timeline)),
            "n_test_timesteps": int(len(test_timeline)),

            "n_pixels": int(data_matrix.shape[1]),

            "train_shape": list(train_data.shape),
            "test_shape": list(test_data.shape),
        }

        save_metadata(metadata_path, metadata)

        gap()
        banner("RINGKASAN")

        say_info(f"TRAIN timestep : {metadata['n_train_timesteps']}")
        say_info(f"TEST timestep  : {metadata['n_test_timesteps']}")
        say_info(f"Pixel          : {metadata['n_pixels']}")

        say_info(
            f"TRAIN          : {metadata['train_start']} \u2192 "
            f"{metadata['train_end']}"
        )

        say_info(
            f"TEST           : {metadata['test_start']} \u2192 "
            f"{metadata['test_end']}"
        )

        say_ok(f"TRAIN shape    : {metadata['train_shape']}")
        say_ok(f"TEST shape     : {metadata['test_shape']}")

        hr()

        say_ok("Tahap 3 selesai.")
        say_info(f"TRAIN : {train_path}")
        say_info(f"TEST  : {test_path}")
        say_info(f"META  : {metadata_path}")

    except Exception as exc:
        say_error(f"Tahap 3 gagal: {exc}")
        raise


if __name__ == "__main__":
    main()
