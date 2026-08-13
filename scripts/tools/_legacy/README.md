# _legacy/

File di folder ini dipindahkan dari `scripts/tools/` pada komit
"Rekonstruksi Metode" (yang menghapus `pipeline/model_training.py` dan
`pipeline/recursive_eval.py` untuk pindah ke metode fixed-window baru).

Semua file di sini **RUSAK di level import** karena masih bergantung pada
salah satu atau kedua modul yang sudah dihapus tersebut:

| File | Import yang hilang |
|---|---|
| `compare_interior_vs_edge_spatial_metrics.py` | `pipeline.recursive_eval._spatial_metrics_per_step` |
| `diagnose_recursive_drift.py` | `pipeline.model_training.load_expanding_dataset`, `pipeline.recursive_eval.*` |
| `sweep_damping.py` | `pipeline.recursive_eval.*` |
| `sweep_noise_std.py` | `pipeline.model_training.*`, `pipeline.recursive_eval.*` |
| `sweep_step_noise_scale.py` | `pipeline.model_training.*`, `pipeline.recursive_eval.*` |
| `generate_summary_report.py` | `pipeline.inference.select_inference_model` (transitif -- `pipeline/inference.py` sendiri gagal import karena baris `from pipeline.recursive_eval import _apply_damping`) |

Selain rusak secara import, isi file-file ini juga dirancang untuk
artefak metode LAMA (expanding window) yang sudah tidak diproduksi
pipeline baru: `dataset/expanding_features.csv`,
`evaluation/recursive_evaluation.csv`,
`evaluation/recursive_mae_summary.csv`, dan script `04_recursive_evaluate.py`
(sudah dihapus, diganti `04_search_window.py` + `06_evaluate_test.py`).

Dibiarkan di sini (bukan dihapus) sebagai referensi historis kalau ada
logic diagnostic yang mau diporting ke metode window baru. Jangan
dijalankan langsung tanpa di-porting dulu.
