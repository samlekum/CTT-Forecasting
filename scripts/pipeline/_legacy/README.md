# pipeline/_legacy/

Modul inti dari metode LAMA (expanding window, purge/embargo split),
dipindah keluar dari `pipeline/` biar gak ketuker sama modul aktif
(`config.py`, `temporal_dataset.py`, `window_features.py`,
`window_model_training.py`, `window_eval.py`, dst).

| File | Import yang hilang |
|---|---|
| `inference.py` | `pipeline.recursive_eval._apply_damping` (modul `pipeline.recursive_eval` tidak ada di repo ini), `Config.MIN_WINDOW_SIZE` (sudah dihapus dari `config.py`, diganti `Config.LEGACY_MIN_WINDOW_SIZE`) |
| `visualize.py` | `pipeline._legacy.inference` (rantai broken yang sama seperti di atas) |
| `validate_no_leakage.py` | `pipeline.model_training`, `pipeline.recursive_eval` (dua-duanya tidak ada di repo ini) |

Semua file di sini **RUSAK di level import** kalau dijalankan sekarang.
Dibiarkan (bukan dihapus) sebagai referensi historis desain expanding
window -- kalau ada logic yang mau diporting ke metode fixed-window baru
(mis. rancangan Stage 7 -- inference produksi baca `model_manifest.json`,
belum ada versi barunya per audit terakhir), porting dulu logic-nya,
jangan jalanin file ini langsung.

`pipeline/expanding_features.py` dan
`pipeline/validate_expanding_features.py` SENGAJA TIDAK dipindah ke sini
-- keduanya murni numpy/pandas, gak nyentuh `Config` yang sudah dihapus,
jadi masih bisa di-import (dipertahankan sebagai validator matematika
metode lama, bukan bagian pipeline 01-06 baru).
