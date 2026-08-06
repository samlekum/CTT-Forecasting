# ./scripts/pipeline/utils.py
# Menyediakan fungsi-fungsi utilitas umum yang digunakan oleh berbagai modul dalam pipeline.

import os


def format_size(num_bytes):
    """Format ukuran byte jadi string yang mudah dibaca."""
    size = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024:
            return f"{size:.2f}{unit}"
        size /= 1024
    return f"{size:.2f}TB"


def get_paths_for_remote_dir(remote_dir, final_base_dir, metadata_base_dir):
    """Generate path lokal (final & metadata) berdasarkan remote_dir, menyerupai struktur FTP."""
    clean_remote = remote_dir.lstrip("/")

    final_dir = os.path.join(final_base_dir, clean_remote)
    meta_dir = os.path.join(metadata_base_dir, clean_remote)
    checked_log_path = os.path.join(meta_dir, "checked_files.log")

    return final_dir, checked_log_path