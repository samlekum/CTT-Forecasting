# ./scripts/_legacy/05_run_inference.py
#
# *** LEGACY -- metode expanding window lama, BUKAN bagian pipeline aktif
# *** (01-06 fixed-window). Disimpan cuma buat referensi/perbandingan.
# *** Rantai import ini TERBUKTI RUSAK kalau dijalankan (pipeline.recursive_eval
# *** tidak ada di repo, Config.MIN_WINDOW_SIZE sudah dihapus) -- jangan
# *** disentuh/dipakai.
#
# Tahap 6 (lama): forecast produksi -- baca window observasi terbaru (atau di
# sekitar --t0) dari file .nc yang sudah ada di data_bandung/, rollout
# recursive pakai model yang sudah dilatih (auto-detect model terbaik dari
# hasil 04_recursive_evaluate.py, atau manual via --model), simpan CSV +
# GeoJSON. Lihat pipeline/_legacy/inference.py untuk detail desain.

import argparse
import os
import sys

_SCRIPTS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SCRIPTS_ROOT)

from ui.terminal_display import hr, gap, banner, say_info, say_error
from pipeline.config import load_config
from pipeline._legacy.inference import run_inference


def parse_args():
    p = argparse.ArgumentParser(
        description="Forecast produksi CTT (expanding window, recursive rollout dari data terbaru)."
    )
    p.add_argument(
        "--data-dir", default=None,
        help="Folder berisi file subset_*.nc. Default: Config.FINAL_BASE_DIR (data_bandung/).",
    )
    p.add_argument(
        "--models-dir", default=None,
        help="Folder model .joblib. Default: Config.EXPANDING_MODELS_DIR.",
    )
    p.add_argument(
        "--model", default=None,
        help=(
            "Nama model production (xgboost/lightgbm/catboost). Default: None -> AUTO-DETECT dari "
            "evaluation/recursive_mae_summary.csv (rata-rata MAE step Config.INFERENCE_PRIORITY_STEP_RANGE, "
            "prioritas horizon panjang -- lihat CLAUDE.md §18.9), BUKAN hardcode. Isi manual buat override."
        ),
    )
    p.add_argument(
        "--t0", default=None,
        help=(
            "Titik waktu observasi TERAKHIR yang jadi basis window forecast (format bebas yang bisa "
            "di-parse pandas, mis. '2026-07-02 08:00'). Forecast dimulai dari t0 + 10 menit. "
            "Default: None -> pakai data TERBARU yang tersedia di --data-dir (bukan t0 tetap)."
        ),
    )
    p.add_argument(
        "--tail-files", type=int, default=None,
        help=(
            "Jumlah file .nc terakhir (mundur dari data terbaru, atau dari --t0 kalau diisi) yang "
            "dibaca untuk cari window observasi per pixel. Default: Config.INFERENCE_TAIL_FILES (100). "
            "WAJIB kecil dibanding total riwayat data_bandung/ -- lihat CLAUDE.md §16 soal bottleneck I/O."
        ),
    )
    p.add_argument(
        "--damping-factor", type=float, default=1.0,
        help=(
            "Redam delta prediksi model terhadap nilai terakhir window, geometris tiap step. "
            "Default 1.0 = TIDAK ADA REDAMAN, konfigurasi final CLAUDE.md §18.9 (prioritas horizon "
            "panjang step 12-18). JANGAN otomatis pakai 0.9 -- itu rekomendasi lama yang sudah usang "
            "untuk model saat ini (noise_std=2.5 + split terpurge)."
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
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()

    banner("INFERENCE - FORECAST PRODUKSI CTT (expanding window)")
    say_info(f"Folder data   : {args.data_dir or cfg.FINAL_BASE_DIR}")
    say_info(f"Folder model  : {args.models_dir or cfg.EXPANDING_MODELS_DIR}")
    say_info(f"Model         : {args.model or '(auto-detect dari hasil 04_recursive_evaluate.py)'}")
    say_info(f"t0            : {args.t0 or '(data terbaru yang tersedia)'}")
    # PENTING: pakai "is not None", BUKAN "or" -- args.tail_files=0 itu
    # falsy jadi "0 or cfg.INFERENCE_TAIL_FILES" bakal salah nampilin "100"
    # padahal yang beneran dipakai itu 0 (nilai berbahaya, lihat validasi
    # eksplisit di pipeline.inference.load_recent_window_data()).
    say_info(f"Tail files    : {args.tail_files if args.tail_files is not None else cfg.INFERENCE_TAIL_FILES}")
    say_info(f"Damping factor: {args.damping_factor}")
    hr()

    try:
        result = run_inference(
            data_dir=args.data_dir,
            models_dir=args.models_dir,
            model_name=args.model,
            tail_files=args.tail_files,
            target_t0=args.t0,
            damping_factor=args.damping_factor,
            output_dir=args.output_dir,
            n_workers=args.workers,
        )
    except (FileNotFoundError, ValueError) as e:
        say_error(str(e))
        return

    gap()
    banner("RINGKASAN INFERENCE")
    say_info(
        f"Model dipakai      : {result['model_name']} "
        f"({'auto-detect' if result['model_auto_detected'] else 'manual via --model'})"
    )
    say_info(f"Pixel di-forecast  : {result['n_used']}/{result['n_total']}")
    say_info(f"Folder output      : {result['run_dir']}")
    say_info(f"Output CSV         : {result['csv_path']}")
    say_info(f"Output GeoJSON     : {result['geojson_path']}")

    anchors_df = result["anchors_df"].sort_values("pixel_id")
    say_info("Rentang window observasi per pixel (window_end_time = titik terakhir yang dipakai):")
    print(anchors_df[["pixel_id", "latitude", "longitude", "anchor_t0", "window_end_time"]].to_string(index=False))

    detail_df = result["detail_df"]
    step1 = detail_df[detail_df["step"] == 1][["pixel_id", "target_time", "y_pred"]].rename(columns={"y_pred": "y_pred_step1"})
    step_last = detail_df[detail_df["step"] == detail_df["step"].max()][["pixel_id", "y_pred"]].rename(
        columns={"y_pred": f"y_pred_step{detail_df['step'].max()}"}
    )
    preview = step1.merge(step_last, on="pixel_id").sort_values("pixel_id")

    gap()
    say_info("Preview forecast step 1 vs step terakhir per pixel:")
    print(preview.to_string(index=False))

    with_actual = detail_df.dropna(subset=["y_true"])
    gap()
    if with_actual.empty:
        say_info(
            "Tidak ada observasi asli (y_true) yang ketemu untuk rentang target_time forecast ini -- "
            "kemungkinan forecast produksi murni ke masa depan yang belum terjadi/didownload. "
            "Kolom y_true/abs_error di output semuanya NaN."
        )
    else:
        mae_per_step = with_actual.groupby("step")["abs_error"].agg(["mean", "size"]).reset_index()
        mae_per_step.columns = ["step", "mae", "n_pixel_dgn_observasi_asli"]
        say_info(
            f"Observasi asli ketemu untuk {len(with_actual)}/{len(detail_df)} baris -- MAE aktual per step:"
        )
        print(mae_per_step.to_string(index=False))

    hr()


if __name__ == "__main__":
    main()
