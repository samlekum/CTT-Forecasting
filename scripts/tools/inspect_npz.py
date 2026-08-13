# ./scripts/tools/inspect_npz.py
#
# Utility buat ngintip isi file .npz (raw temporal cache hasil
# 02_build_temporal_cache.py, atau .npz lain yang formatnya mirip --
# lihat pipeline/dataset_builder.py::save_raw_cache()).
#
# .npz itu arsip biner (bukan teks kayak .csv), jadi gak bisa dibuka
# langsung di Excel/text editor. Script ini nampilin ringkasan tiap
# array di dalamnya (key, shape, dtype), preview beberapa baris, dan
# opsional convert ke .csv kalau mau di-eyeball manual.
#
# TIDAK mengubah file .npz asli -- read-only, cuma baca dan (kalau
# diminta) nulis file .csv preview terpisah.
#
# Cara pakai (dari folder scripts/, sama seperti 01-06):
#   python tools/inspect_npz.py --path ../dataset/expanding_raw_cache.npz
#   python tools/inspect_npz.py --path ../dataset/expanding_raw_cache.npz --rows 20
#   python tools/inspect_npz.py --path ../dataset/expanding_raw_cache.npz --to-csv preview.csv
#   python tools/inspect_npz.py --path ../dataset/expanding_raw_cache.npz --to-csv preview.csv --max-cols 50

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="Ringkasan isi file .npz (khususnya raw temporal cache Stage 02)."
    )
    p.add_argument(
        "--path", required=True,
        help="Path ke file .npz yang mau diinspeksi.",
    )
    p.add_argument(
        "--rows", type=int, default=10,
        help="Jumlah baris preview data_matrix yang ditampilkan (default: 10).",
    )
    p.add_argument(
        "--max-cols", type=int, default=10,
        help="Jumlah kolom/pixel pertama yang ditampilkan di preview terminal (default: 10). "
             "Tidak berlaku untuk --to-csv (CSV selalu berisi SEMUA kolom).",
    )
    p.add_argument(
        "--to-csv", default=None,
        help="Kalau diisi, export data_matrix (lengkap, semua baris & kolom) jadi file "
             "CSV di path ini -- baru bisa dibuka Excel/spreadsheet biasa.",
    )
    return p.parse_args()


def describe_keys(npz):
    """Tampilkan key, shape, dtype tiap array di dalam .npz."""
    print("=== Isi file (key -> shape, dtype) ===")
    for key in npz.files:
        arr = npz[key]
        size_mb = arr.nbytes / (1024 * 1024)
        print(f"  {key:15s} shape={str(arr.shape):15s} dtype={str(arr.dtype):10s} ~{size_mb:.2f} MB")
    print()


def describe_timeline(npz):
    """Kalau ada timeline_ns, tampilkan rentang waktu & jumlah gap NaN."""
    if "timeline_ns" not in npz.files:
        return

    timeline = pd.to_datetime(npz["timeline_ns"])
    print("=== Timeline ===")
    print(f"  mulai       : {timeline.min()}")
    print(f"  selesai     : {timeline.max()}")
    print(f"  jumlah titik: {len(timeline)}")
    print()


def describe_missing(npz):
    """Ringkasan NaN di data_matrix -- penting buat cek gap data."""
    if "data_matrix" not in npz.files:
        return

    dm = npz["data_matrix"]
    n_total = dm.size
    n_nan = int(np.isnan(dm).sum())
    pct = (n_nan / n_total * 100) if n_total else 0.0

    print("=== Data hilang (NaN) di data_matrix ===")
    print(f"  total sel   : {n_total}")
    print(f"  NaN         : {n_nan} ({pct:.2f}%)")
    print()


def preview_table(npz, n_rows, max_cols):
    """Tampilkan potongan data_matrix sebagai tabel (mirip buka CSV)."""
    if "data_matrix" not in npz.files:
        print("(tidak ada key 'data_matrix', skip preview tabel)")
        return

    dm = npz["data_matrix"]
    n_cols_show = min(max_cols, dm.shape[1])

    index = None
    if "timeline_ns" in npz.files:
        index = pd.to_datetime(npz["timeline_ns"][:n_rows])

    columns = None
    if "pixel_id" in npz.files:
        columns = npz["pixel_id"][:n_cols_show]

    df = pd.DataFrame(
        dm[:n_rows, :n_cols_show],
        index=index,
        columns=columns,
    )

    print(
        f"=== Preview {n_rows} baris x {n_cols_show} kolom pertama "
        f"(dari total {dm.shape[0]} x {dm.shape[1]}) ==="
    )
    print(df)
    print()


def export_csv(npz, out_path):
    """Export SELURUH data_matrix (semua baris & kolom) jadi CSV."""
    if "data_matrix" not in npz.files:
        print("(tidak ada key 'data_matrix', tidak bisa export CSV)")
        return

    dm = npz["data_matrix"]

    index = None
    if "timeline_ns" in npz.files:
        index = pd.to_datetime(npz["timeline_ns"])

    columns = None
    if "pixel_id" in npz.files:
        columns = npz["pixel_id"]

    df = pd.DataFrame(dm, index=index, columns=columns)
    if index is not None:
        df.index.name = "timestamp"

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    df.to_csv(out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"Diekspor ke: {out_path} ({size_mb:.2f} MB, shape={df.shape})")


def main():
    args = parse_args()

    if not os.path.exists(args.path):
        print(f"File tidak ditemukan: {args.path}")
        sys.exit(1)

    npz = np.load(args.path, allow_pickle=False)

    print(f"File: {args.path}\n")
    describe_keys(npz)
    describe_timeline(npz)
    describe_missing(npz)
    preview_table(npz, n_rows=args.rows, max_cols=args.max_cols)

    if args.to_csv:
        export_csv(npz, args.to_csv)


if __name__ == "__main__":
    main()
