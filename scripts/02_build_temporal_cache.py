# ./scripts/02_build_expanding_features.py

import argparse
import os
import time

from ui.terminal_display import (
    hr,
    gap,
    banner,
    say_info,
    say_ok,
    say_error,
)

from pipeline.config import load_config
from pipeline.dataset_builder import (
    discover_nc_files,
    build_uniform_timeline,
    load_pixel_grid,
    save_raw_cache,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build raw temporal cache dari file NetCDF Himawari. "
            "Tahap ini tidak membuat sliding-window features."
        )
    )

    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Folder berisi file subset_*.nc. "
            "Default: Config.FINAL_BASE_DIR."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help=(
            "Path output temporal cache (.npz). "
            "Default: Config.EXPANDING_RAW_CACHE_FILE."
        ),
    )

    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help=(
            "SMOKE-TEST: hanya gunakan N file pertama secara kronologis. "
            "Gunakan sebelum full run."
        ),
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=(
            "Jumlah worker untuk membaca NetCDF. "
            "Default: Config.NETCDF_READ_WORKERS."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    data_dir = args.data_dir or cfg.FINAL_BASE_DIR
    output_path = args.output or cfg.EXPANDING_RAW_CACHE_FILE

    banner("BUILD TEMPORAL CACHE")

    say_info(f"Folder data : {data_dir}")
    say_info(f"Output      : {output_path}")
    say_info(
        f"Workers     : "
        f"{args.workers or cfg.NETCDF_READ_WORKERS}"
    )

    if args.max_files is not None:
        say_info(
            f"Mode        : SMOKE-TEST ({args.max_files} file pertama)"
        )
    else:
        say_info("Mode        : FULL RUN")

    hr()

    start_time = time.time()

    try:
        # ------------------------------------------------------------
        # 1. Discover NetCDF files
        # ------------------------------------------------------------
        entries = discover_nc_files(data_dir)

        if not entries:
            raise ValueError(
                f"Tidak ada file subset_*.nc ditemukan di: {data_dir}"
            )

        if args.max_files is not None:
            if args.max_files < 1:
                raise ValueError(
                    "--max-files harus >= 1."
                )

            entries = entries[:args.max_files]

        say_info(
            f"File NetCDF yang diproses: {len(entries)}"
        )

        # ------------------------------------------------------------
        # 2. Build uniform timeline
        # ------------------------------------------------------------
        timeline = build_uniform_timeline(
            entries,
            freq_minutes=cfg.FREQ_MINUTES,
        )

        say_info(
            f"Timeline: {timeline[0]} → {timeline[-1]}"
        )
        say_info(
            f"Jumlah timestep: {len(timeline)}"
        )

        # ------------------------------------------------------------
        # 3. Load raw temporal matrix
        #
        # Shape:
        #   T = timestep
        #   P = pixel
        # ------------------------------------------------------------
        data_matrix, pixel_meta = load_pixel_grid(
            entries,
            timeline,
            n_workers=args.workers,
        )

        say_info(
            f"Data matrix shape: {data_matrix.shape}"
        )

        # ------------------------------------------------------------
        # 4. Save raw temporal cache
        # ------------------------------------------------------------
        os.makedirs(
            os.path.dirname(output_path),
            exist_ok=True,
        )

        save_raw_cache(
            output_path,
            data_matrix,
            timeline,
            pixel_meta,
        )

        elapsed = time.time() - start_time

        # ------------------------------------------------------------
        # Summary
        # ------------------------------------------------------------
        gap()
        banner("RINGKASAN")

        say_info(
            f"Timestep       : {data_matrix.shape[0]}"
        )

        say_info(
            f"Pixel          : {data_matrix.shape[1]}"
        )

        say_info(
            f"Timeline       : "
            f"{timeline[0]} s/d {timeline[-1]}"
        )

        say_info(
            f"Valid values   : "
            f"{(~__import__('numpy').isnan(data_matrix)).sum():,}"
        )

        say_info(
            f"NaN / gaps     : "
            f"{__import__('numpy').isnan(data_matrix).sum():,}"
        )

        say_info(
            f"Waktu proses   : {elapsed:.1f} detik"
        )

        say_ok(
            f"Temporal cache disimpan ke: {output_path}"
        )

        hr()

        if args.max_files is not None:
            say_info(
                "Ini adalah SMOKE-TEST. "
                "Kalau hasilnya masuk akal, jalankan ulang "
                "tanpa --max-files untuk full run."
            )
        else:
            say_ok(
                "Tahap 2 selesai. "
                "Temporal cache siap digunakan oleh Tahap 3."
            )

    except Exception as exc:
        say_error(str(exc))
        raise


if __name__ == "__main__":
    main()