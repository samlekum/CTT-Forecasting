# ./scripts/02_build_expanding_features.py
# Tahap 2: scan semua file subset_*.nc di data_bandung/, generate dataset
# training expanding window (fitur closed-form + gap-skip rule), simpan
# sebagai CSV. Lihat CLAUDE.md §2-4 untuk metode, §12 untuk keputusan
# desain dataset_builder.py.

import argparse
import time

from ui.terminal_display import hr, gap, banner, say_info, say_ok, say_error
from pipeline.config import load_config
from pipeline.dataset_builder import build_dataset


def parse_args():
    p = argparse.ArgumentParser(
        description="Build dataset training expanding window dari file NetCDF Himawari."
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Folder berisi file subset_*.nc. Default: Config.FINAL_BASE_DIR (data_bandung/).",
    )
    p.add_argument(
        "--output", default=None,
        help="Path output CSV. Default: Config.EXPANDING_DATASET_FILE.",
    )
    p.add_argument(
        "--anchor-stride", type=int, default=None,
        help="Jarak antar anchor. Default: Config.ANCHOR_STRIDE_DEFAULT (1).",
    )
    p.add_argument(
        "--max-files", type=int, default=None,
        help=(
            "SMOKE-TEST: cuma pakai N file pertama (kronologis), bukan full run. "
            "WAJIB dicoba dulu sebelum full run tanpa flag ini -- lihat CLAUDE.md §11 "
            "(pelajaran dari 03a_build_features.py yang makan 10 jam+ di repo lama "
            "karena langsung full-run tanpa validasi subset). Contoh: --max-files 200."
        ),
    )
    p.add_argument(
        "--no-cache", action="store_true",
        help=(
            "Skip nyimpen cache raw time series (.npz) yang dibutuhkan "
            "04_recursive_evaluate.py. Default: cache DISIMPAN. Pakai flag ini "
            "cuma buat smoke-test cepat yang nggak perlu lanjut ke recursive eval."
        ),
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help=(
            "Jumlah proses paralel untuk baca file NetCDF. Default: "
            "Config.NETCDF_READ_WORKERS (semua core kecuali 1). Set --workers 1 "
            "untuk fallback sekuensial murni (mis. debug, atau kalau I/O disk "
            "sudah jadi bottleneck duluan sebelum CPU sehingga paralel tidak "
            "nambah kecepatan)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    data_dir = args.data_dir or cfg.FINAL_BASE_DIR
    output_path = args.output or cfg.EXPANDING_DATASET_FILE

    banner("BUILD DATASET - EXPANDING WINDOW FEATURES")
    say_info(f"Folder data   : {data_dir}")
    say_info(f"Output CSV    : {output_path}")
    say_info(f"Anchor stride : {args.anchor_stride or cfg.ANCHOR_STRIDE_DEFAULT}")
    say_info(f"Workers baca  : {args.workers or cfg.NETCDF_READ_WORKERS}")
    if args.max_files is not None:
        say_info(f"Mode          : SMOKE-TEST, hanya {args.max_files} file pertama (kronologis)")
    else:
        say_info("Mode          : FULL RUN (semua file) -- pastikan sudah smoke-test dulu dengan --max-files")
    hr()

    start = time.time()
    try:
        df = build_dataset(
            data_dir=data_dir,
            output_path=output_path,
            anchor_stride=args.anchor_stride,
            max_files=args.max_files,
            cache_path=False if args.no_cache else None,
            n_workers=args.workers,
        )
    except ValueError as e:
        say_error(str(e))
        return
    elapsed = time.time() - start

    gap()
    banner("RINGKASAN")
    say_info(f"Total baris        : {len(df)}")
    say_info(f"Jumlah pixel unik  : {df['pixel_id'].nunique()}")
    say_info(f"Jumlah anchor unik : {df.groupby('pixel_id')['anchor_t0'].nunique().sum()}")
    say_info(f"Rentang waktu      : {df['anchor_t0'].min()} s/d {df['target_time'].max()}")
    say_info(f"Waktu proses       : {elapsed:.1f} detik")
    say_ok(f"Dataset disimpan ke: {output_path}")

    hr()
    if args.max_files is not None:
        say_info("Ini hasil smoke-test. Kalau hasilnya masuk akal, jalankan ulang TANPA --max-files untuk full run.")
    else:
        say_info("Lanjut ke Tahap 3: 03_train_models.py")


if __name__ == "__main__":
    main()