# ./scripts/_legacy/06_visualize.py
#
# *** LEGACY -- metode expanding window lama, BUKAN bagian pipeline aktif
# *** (01-06 fixed-window). Disimpan cuma buat referensi/perbandingan.
# *** Rantai import ini TERBUKTI RUSAK (pipeline._legacy.inference ->
# *** pipeline.recursive_eval yang tidak ada di repo) -- jangan
# *** disentuh/dipakai.
#
# Tahap 7 (lama): render hasil forecast Stage 05 (CSV di forecast_output/) jadi
# animasi GIF -- 6 panel PETA per step (Input/Prediksi/Aktual/Kelas Awan/
# Risiko Banjir/Error Map). Trigger manual/on-demand, pemilihan file
# forecast via CLI select menu (auto-pick kalau cuma 1, list+prompt kalau
# >1, berhenti bersih kalau 0). Lihat pipeline/_legacy/visualize.py untuk detail.

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from ui.terminal_display import hr, gap, banner, say_info, say_error
from pipeline.config import load_config
from pipeline._legacy.visualize import visualize_forecast


def parse_args():
    p = argparse.ArgumentParser(
        description="Render hasil forecast Stage 05 jadi animasi GIF 6-panel peta."
    )
    p.add_argument(
        "--forecast-dir", default=None,
        help="Folder berisi CSV forecast Stage 05. Default: Config.INFERENCE_DIR (forecast_output/).",
    )
    p.add_argument(
        "--csv", default=None,
        help=(
            "Path langsung ke 1 file CSV forecast (skip menu pilih). Default: None -> tampilkan "
            "menu pilih dari --forecast-dir (auto-pick kalau cuma 1 hasil, berhenti kalau 0)."
        ),
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Folder output visualisasi. Default: Config.VISUALIZATION_DIR (visualizations/).",
    )
    p.add_argument(
        "--frame-duration-ms", type=int, default=None,
        help="Durasi tiap frame GIF (ms). Default: Config.VISUALIZATION_FRAME_DURATION_MS (600).",
    )
    p.add_argument(
        "--mask-outside-bandung", action="store_true",
        help=(
            "Mask (jadi abu-abu) pixel yang cell-nya TIDAK beririsan dgn batas administratif "
            "Kota Bandung (scripts/geojson/KotaBandung.geojson) -- hasilnya cuma 13/35 pixel "
            "yang tampil. Default: NONAKTIF, tampilkan semua 35 pixel grid seperti biasa."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    load_config()

    banner("VISUALISASI - RENDER FORECAST CTT JADI GIF")
    say_info(f"Folder forecast : {args.forecast_dir or '(Config.INFERENCE_DIR)'}")
    say_info(f"CSV langsung    : {args.csv or '(pilih via menu)'}")
    say_info(f"Folder output   : {args.output_dir or '(Config.VISUALIZATION_DIR)'}")
    say_info(f"Mask luar Bandung: {'AKTIF (--mask-outside-bandung)' if args.mask_outside_bandung else 'NONAKTIF (default)'}")
    hr()

    try:
        result = visualize_forecast(
            csv_path=args.csv,
            forecast_dir=args.forecast_dir,
            output_dir=args.output_dir,
            frame_duration_ms=args.frame_duration_ms,
            mask_outside_bandung=args.mask_outside_bandung,
        )
    except (FileNotFoundError, ValueError) as e:
        say_error(str(e))
        return
    except KeyboardInterrupt:
        say_info("Dibatalkan.")
        return

    gap()
    banner("RINGKASAN VISUALISASI")
    say_info(f"Model           : {result['model_name']}")
    say_info(f"t0              : {result['t0_str']}")
    say_info(f"Run forecast    : {result['run_str']}")
    say_info(f"Jumlah frame    : {result['n_frames']}")
    say_info(f"Folder frame    : {result['frames_dir']}")
    say_info(f"GIF             : {result['gif_path']}")
    if result["missing_pixels"]:
        say_info(f"Pixel hilang    : {len(result['missing_pixels'])} (lihat log di atas)")
    if not result["has_actual_values"]:
        say_info("Panel Aktual/Error Map: NaN semua (forecast murni masa depan, belum ada observasi asli).")
    hr()


if __name__ == "__main__":
    main()
