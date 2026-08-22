# ./scripts/07_run_inference.py
#
# Tahap 7: forecast produksi CTT (metode window, recursive rollout dari
# data terbaru). Model production dipilih OTOMATIS dari hasil
# 06_evaluate_test.py (avg MAE step 12-18), atau manual via --model.
# Window model dibaca dari model_manifest.json -- TIDAK bisa dioverride
# manual (window sudah ditetapkan sejak 04_search_window.py + 05, ganti
# window di sini akan mismatch dengan model yang sudah dilatih).
#
# CATATAN: pipeline ini TIDAK memakai damping (beda dari
# scripts/_legacy/05_run_inference.py) -- konsisten dengan
# 04_search_window.py/06_evaluate_test.py yang juga tanpa damping.
#
# Contoh pakai:
#   python scripts/07_run_inference.py
#   python scripts/07_run_inference.py --model lightgbm
#   python scripts/07_run_inference.py --t0 "2026-07-15 08:00" --tail-files 50

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from ui.terminal_display import hr, gap, banner, say_info, say_error
from pipeline.config import load_config
from pipeline.inference import run_inference


def parse_args():
    p = argparse.ArgumentParser(
        description="Forecast produksi CTT (metode window, recursive rollout dari data terbaru)."
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Folder berisi file subset_*.nc. Default: Config.FINAL_BASE_DIR (data_bandung/).",
    )
    p.add_argument(
        "--manifest-file", default=None,
        help=(
            "Path model_manifest.json (hasil 05_train_final_models.py). "
            "Default: Config.WINDOW_FINAL_MANIFEST_FILE."
        ),
    )
    p.add_argument(
        "--model", default=None,
        help=(
            "Nama model production (xgboost/lightgbm/catboost). Default: None -> "
            "AUTO-DETECT dari test_evaluation_summary.csv (rata-rata MAE step "
            "Config.INFERENCE_PRIORITY_STEP_RANGE, prioritas horizon panjang). "
            "Window model diambil dari manifest, TIDAK bisa di-override manual di sini."
        ),
    )
    p.add_argument(
        "--t0", default=None,
        help=(
            "Titik waktu observasi TERAKHIR yang jadi basis window forecast "
            "(format bebas yang bisa di-parse pandas, mis. '2026-07-02 08:00'). "
            "Forecast dimulai dari t0 + 10 menit. Default: None -> pakai data "
            "TERBARU yang tersedia di --data-dir."
        ),
    )
    p.add_argument(
        "--tail-files", type=int, default=None,
        help=(
            "Jumlah file .nc terakhir (mundur dari data terbaru, atau dari --t0 "
            "kalau diisi) yang dibaca untuk cari window observasi per pixel. "
            "Default: Config.INFERENCE_TAIL_FILES (100)."
        ),
    )
    p.add_argument(
        "--output-dir", default=None,
        help="Folder output forecast. Default: Config.INFERENCE_DIR (forecast_output/).",
    )
    p.add_argument(
        "--workers", type=int, default=None,
        help="Jumlah proses paralel baca NetCDF. Default: Config.NETCDF_READ_WORKERS. --workers 1 = sekuensial.",
    )
    p.add_argument(
        "--skip-actual", action="store_true",
        help=(
            "Lewati pencarian observasi asli untuk y_true/abs_error (lebih cepat). "
            "Berguna kalau memang tahu forecast ini murni ke masa depan yang belum "
            "terjadi (y_true pasti NaN semua)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    manifest_file = args.manifest_file or cfg.WINDOW_FINAL_MANIFEST_FILE

    banner("INFERENCE - FORECAST PRODUKSI CTT (metode window)")
    say_info(f"Folder data   : {args.data_dir or cfg.FINAL_BASE_DIR}")
    say_info(f"Manifest      : {manifest_file}")
    say_info(f"Model         : {args.model or '(auto-detect dari 06_evaluate_test.py)'}")
    say_info(f"t0            : {args.t0 or '(data terbaru yang tersedia)'}")
    say_info(f"Tail files    : {args.tail_files if args.tail_files is not None else cfg.INFERENCE_TAIL_FILES}")
    say_info(f"Skip actual   : {args.skip_actual}")
    hr()

    try:
        result = run_inference(
            data_dir=args.data_dir,
            manifest_file=manifest_file,
            model_name=args.model,
            tail_files=args.tail_files,
            target_t0=args.t0,
            output_dir=args.output_dir,
            n_workers=args.workers,
            skip_actual=args.skip_actual,
        )
    except (FileNotFoundError, ValueError, KeyError) as e:
        say_error(str(e))
        raise SystemExit(1)

    gap()
    banner("RINGKASAN INFERENCE")
    say_info(
        f"Model dipakai      : {result['model_name']} "
        f"({'auto-detect' if result['model_auto_detected'] else 'manual via --model'}), "
        f"window={result['window']}"
    )
    say_info(f"Pixel di-forecast  : {result['n_used']}/{result['n_total']}")
    say_info(f"Folder output      : {result['run_dir']}")
    say_info(f"Output CSV         : {result['csv_path']}")
    say_info(f"Output GeoJSON     : {result['geojson_path']}")

    anchors_df = result["anchors_df"].sort_values("pixel_id")
    say_info("Window observasi per pixel (window_end_time = titik terakhir yang dipakai):")
    print(anchors_df[["pixel_id", "latitude", "longitude", "window_end_time"]].to_string(index=False))

    detail_df = result["detail_df"]
    step1 = detail_df[detail_df["step"] == 1][["pixel_id", "target_time", "y_pred"]].rename(
        columns={"y_pred": "y_pred_step1"}
    )
    last_step = detail_df["step"].max()
    step_last = detail_df[detail_df["step"] == last_step][["pixel_id", "y_pred"]].rename(
        columns={"y_pred": f"y_pred_step{last_step}"}
    )
    preview = step1.merge(step_last, on="pixel_id").sort_values("pixel_id")

    gap()
    say_info("Preview forecast step 1 vs step terakhir per pixel:")
    print(preview.to_string(index=False))

    with_actual = detail_df.dropna(subset=["y_true"])
    gap()
    if with_actual.empty:
        say_info(
            "Tidak ada observasi asli (y_true) yang ketemu untuk rentang target_time "
            "forecast ini -- kemungkinan forecast produksi murni ke masa depan yang "
            "belum terjadi/didownload, atau --skip-actual dipakai. Kolom y_true/"
            "abs_error di output semuanya NaN."
        )
    else:
        mae_per_step = with_actual.groupby("step")["abs_error"].agg(["mean", "size"]).reset_index()
        mae_per_step.columns = ["step", "mae", "n_pixel_dgn_observasi_asli"]
        say_info(
            f"Observasi asli ketemu untuk {len(with_actual)}/{len(detail_df)} baris -- "
            "MAE aktual per step:"
        )
        print(mae_per_step.to_string(index=False))

    hr()
    say_info("Tahap 7 (inference produksi) selesai.")


if __name__ == "__main__":
    main()