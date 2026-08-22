# _legacy/

Entry point dari metode LAMA (expanding window, purge/embargo split),
dipindah keluar dari `scripts/` biar gak ketuker sama pipeline aktif
01-06 (fixed-window, chronological split).

| File | Import yang hilang |
|---|---|
| `05_run_inference.py` | `pipeline._legacy.inference` (transitif -- `from pipeline.recursive_eval import _apply_damping`, modul ini tidak ada di repo) |
| `06_visualize.py` | `pipeline._legacy.visualize` -> `pipeline._legacy.inference` (rantai broken yang sama) |

Modul intinya (`inference.py`, `visualize.py`, `validate_no_leakage.py`)
ada di `scripts/pipeline/_legacy/` -- lihat README di sana.

Disimpan sebagai referensi/perbandingan desain (window observasi terbaru,
rollout recursive produksi, render GIF 6-panel), BUKAN bagian dari
pipeline aktif. Jangan dijalankan langsung.

Lihat juga `scripts/tools/_legacy/` untuk diagnostic scripts dari metode
yang sama.
