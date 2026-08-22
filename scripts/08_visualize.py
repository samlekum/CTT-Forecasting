# ./scripts/08_visualize.py
#
# Tahap 8: render hasil forecast Tahap 7 (forecast_output/{model}_w{window}_
# t0{...}_run{...}/forecast.csv) jadi animasi GIF 6-panel per step (Input,
# Prediksi, Aktual, Kelas Awan, Risiko Banjir, Error Map).
#
# Tanpa --csv: tampil menu pilih dari semua folder forecast yang ada
# (terurut run terbaru duluan). Dengan --csv: langsung render folder itu,
# tanpa menu.
#
# Contoh pakai:
#   python scripts/08_visualize.py
#   python scripts/08_visualize.py --csv forecast_output/xgboost_w6_t0.../forecast.csv
#   python scripts/08_visualize.py --mask-outside-bandung

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from ui.terminal_display import hr, gap, banner, say_info, say_error, say_ok
from pipeline.config import load_config
from pipeline.visualize import visualize_forecast


def parse_args():
    p = argparse.ArgumentParser(
        description="Render forecast produksi CTT (Tahap 7) jadi animasi GIF 6-panel."
    )
    p.add_argument(
        "--csv", default=None,
        help=(
            "Path langsung ke forecast.csv (di dalam folder "
            "forecast_output/{model}_w{window}_t0..._run.../). "
            "Default: None -> tampilkan menu pilih dari semua folder yang ada."
        ),
    )
    p.add_argument(
        "--forecast-dir", default=None,
        help="Folder berisi hasil Tahap 7. Default: Config.INFERENCE_DIR (forecast_output/).",
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Folder output visualisasi. Default: Config.VISUALIZATION_DIR (visualizations/).",
    )
    p.add_argument(
        "--frame-duration-ms", type=int, default=None,
        help="Durasi tiap frame GIF (ms). Default: Config.VISUALIZATION_FRAME_DURATION_MS.",
    )
    p.add_argument(
        "--mask-outside-bandung", action="store_true",
        help=(
            "Mask (abu-abu) pixel yang cell-nya tidak beririsan dengan batas "
            "administratif Kota Bandung. Default: nonaktif (tampilkan semua pixel grid)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    banner("VISUALISASI FORECAST CTT (Tahap 8)")
    say_info(f"CSV           : {args.csv or '(menu pilih interaktif)'}")
    say_info(f"Folder forecast: {args.forecast_dir or cfg.INFERENCE_DIR}")
    say_info(f"Folder output : {args.output_dir or cfg.VISUALIZATION_DIR}")
    say_info(f"Mask Bandung  : {args.mask_outside_bandung}")
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
        raise SystemExit(1)
    except KeyboardInterrupt:
        say_error("Dibatalkan user.")
        raise SystemExit(1)

    gap()
    banner("RINGKASAN VISUALISASI")
    say_info(f"Model         : {result['model_name']} (window={result['window']})")
    say_info(f"t0            : {result['t0_str']}")
    say_info(f"Jumlah frame  : {result['n_frames']}")
    say_info(f"Observasi asli: {'ada' if result['has_actual_values'] else 'belum ada (forecast murni ke masa depan)'}")
    if result["missing_pixels"]:
        say_info(f"Pixel hilang  : {len(result['missing_pixels'])} (lihat log di atas)")
    say_ok(f"Frame PNG     : {result['frames_dir']}")
    say_ok(f"GIF           : {result['gif_path']}")

    hr()
    say_ok("Tahap 8 (visualisasi) selesai.")


if __name__ == "__main__":
    main()