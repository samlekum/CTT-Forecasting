# ./scripts/tools/test_window_features.py
#
# Smoke-test manual untuk pipeline/window_features.py di data ASLI
# (bukan sintetis). Jalankan setelah 03_prepare_temporal_dataset.py
# berhasil.
#
# Cara pakai:
#   python scripts/tools/test_window_features.py
#
# Ini BUKAN bagian dari pipeline resmi (04/05/dst) -- cuma buat validasi
# manual sebelum lanjut ke tahap window search.

import os
import sys
import time

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)

from pipeline.config import load_config
from pipeline.temporal_dataset import load_temporal_cache
from pipeline.window_features import (
    build_window_dataset,
    find_full_horizon_anchors_single_pixel,
)


def main():
    cfg = load_config()

    train_path = cfg.TEMPORAL_TRAIN_FILE

    print(f"Load TRAIN dari: {train_path}")

    data_matrix, timeline, pixel_meta = load_temporal_cache(train_path)

    print(f"TRAIN shape       : {data_matrix.shape}")
    print(f"Timeline          : {timeline[0]} -> {timeline[-1]}")
    print(f"Jumlah pixel      : {data_matrix.shape[1]}")
    print()

    for window in [1, 6, 18]:
        t0 = time.time()

        df = build_window_dataset(
            data_matrix,
            timeline,
            pixel_meta,
            window=window,
        )

        elapsed = time.time() - t0

        print(f"--- window={window} ---")
        print(f"  Jumlah training sample : {len(df):,}")
        print(f"  Jumlah kolom           : {df.shape[1]}")
        print(f"  Kolom                  : {list(df.columns)}")
        print(f"  Waktu proses           : {elapsed:.2f} detik")
        print(f"  Contoh baris pertama   :")
        print(df.head(2).to_string())
        print()

    print("--- cek anchor recursive-eval (window=6, horizon=18) ---")

    y_pixel0 = data_matrix[:, 0]

    anchors = find_full_horizon_anchors_single_pixel(
        y_pixel0,
        window=6,
        horizon_steps=cfg.HORIZON_STEPS,
    )

    print(f"  Jumlah anchor valid (pixel pertama) : {len(anchors):,}")

    print()
    print("SELESAI. Kalau semua angka di atas masuk akal (jumlah sample")
    print("berkurang seiring window makin besar, gak ada error, waktu")
    print("proses wajar/cepat), modul window_features.py siap dipakai")
    print("untuk tahap window search berikutnya.")


if __name__ == "__main__":
    main()
