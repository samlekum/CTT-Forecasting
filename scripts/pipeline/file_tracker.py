# ./scripts/pipeline/file_tracker.py
# Mengelola pencatatan file yang telah diproses menggunakan checked_files.log untuk mencegah pemrosesan ulang.

import os

class FileTracker:

    """Lacak file yang sudah diproses lewat checked_files.log."""
    def __init__(self, metadata_base_dir):
        self.metadata_base_dir = metadata_base_dir

    def get_log_path(self, remote_dir):
        """Ambil path checked_files.log untuk suatu direktori remote."""
        clean_remote = remote_dir.lstrip("/")
        meta_dir = os.path.join(self.metadata_base_dir, clean_remote)
        return os.path.join(meta_dir, "checked_files.log")

    def load_checked(self, remote_dir):
        """Muat daftar nama file yang sudah pernah dicek."""
        log_path = self.get_log_path(remote_dir)
        if not os.path.exists(log_path):
            return set()
        with open(log_path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())

    def append_checked(self, remote_dir, filename):
        """Tambahkan nama file ke checked log."""
        log_path = self.get_log_path(remote_dir)
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(filename + "\n")