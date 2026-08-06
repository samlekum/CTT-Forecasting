# ./scripts/pipeline/ftp_client.py
# Menyediakan antarmuka untuk berinteraksi dengan server FTP, meliputi manajemen koneksi, penelusuran direktori, pengunduhan file, serta mekanisme reconnect otomatis.

import ftplib
import time
import logging

from ui.terminal_display import say_login, say_ok, say_error, make_download_progress_bar
from pipeline.utils import format_size


class FTPClient:
    
    """Klien FTP dengan manajemen koneksi dan kemampuan download."""
    def __init__(self, host, user, password, timeout=60):
        self.host = host
        self.user = user
        self.password = password
        self.timeout = timeout
        self._ftp = None
        self._dirty = False

    def connect(self):
        """Buat koneksi FTP baru."""
        say_login(f"Menghubungkan ke {self.host} ...")
        try:
            self._ftp = ftplib.FTP(self.host, timeout=self.timeout)
            self._ftp.login(user=self.user, passwd=self.password)
        except ftplib.error_perm as e:
            say_error(f"Username/password ditolak server: {e}")
            logging.error(f"Login gagal (error_perm): {e}")
            raise
        except Exception as e:
            say_error(f"Tidak bisa terhubung ke {self.host}: {e}")
            logging.error(f"Login gagal (koneksi): {e}")
            raise
        else:
            self._dirty = False
            say_ok(f"Terhubung sebagai '{self.user}'")
            logging.info(f"Login berhasil sebagai {self.user}")
            return self

    def mark_dirty(self):
        self._dirty = True

    def is_alive(self):
        if self._ftp is None:
            return False

        if getattr(self, "_dirty", False):
            return False

        try:
            self._ftp.voidcmd("NOOP")
            return True
        except Exception:
            return False

    def ensure_connected(self):
        """Sambung ulang kalau koneksi sudah mati."""
        if not self.is_alive():
            say_error("Koneksi FTP terputus, mencoba menyambung ulang ...")
            logging.info("Koneksi FTP terputus, reconnect.")
            self.connect()

    def cwd(self, remote_dir):
        """Ganti direktori kerja."""
        self._ftp.cwd(remote_dir)

    def list_directory(self, remote_dir):
        """Daftar isi direktori FTP; return (files, subdirs), pakai MLSD kalau didukung, kalau tidak fallback ke NLST + trial-cwd pada basename."""
        files = []
        subdirs = []

        try:
            self._ftp.cwd(remote_dir)
        except Exception as e:
            say_error(f"Gagal masuk direktori {remote_dir}: {e}")
            logging.error(f"Gagal masuk direktori {remote_dir}: {e}")
            return files, subdirs

        # --- Preferred path: MLSD ---
        try:
            for name, facts in self._ftp.mlsd():
                if name in (".", ".."):
                    continue
                entry_type = facts.get("type", "")
                if entry_type == "dir":
                    subdirs.append(name)
                elif entry_type == "file":
                    if name.endswith(".nc"):
                        files.append(name)
                # other types (cdir, pdir, link, etc.) are ignored
            return files, subdirs
        except Exception:
            logging.info(f"MLSD tidak didukung/gagal untuk {remote_dir}, fallback ke NLST+CWD")

        # --- Fallback: NLST + trial cwd ---
        try:
            items = self._ftp.nlst()
        except Exception as e:
            say_error(f"Gagal list direktori {remote_dir}: {e}")
            logging.error(f"Gagal list direktori {remote_dir}: {e}")
            return files, subdirs

        for item in items:
            item_name = item.rstrip("/").split("/")[-1]
            if not item_name or item_name in (".", ".."):
                continue
            try:
                self._ftp.cwd(item_name)
                self._ftp.cwd("..")
                subdirs.append(item_name)
            except ftplib.error_perm:
                if item_name.endswith(".nc"):
                    files.append(item_name)
            except Exception:
                if item_name.endswith(".nc"):
                    files.append(item_name)

        return files, subdirs

    def download_with_progress(self, filename, dest_path):
        """Download file from FTP with progress bar."""
        total_size = None
        try:
            self._ftp.voidcmd("TYPE I")
            total_size = self._ftp.size(filename)
        except Exception:
            total_size = None

        downloaded = {"n": 0}
        progress_bar = make_download_progress_bar(total_size)

        with open(dest_path, "wb") as f:
            def write_chunk(data):
                f.write(data)
                downloaded["n"] += len(data)
                progress_bar.update(len(data))

            self._ftp.retrbinary(f"RETR {filename}", write_chunk, blocksize=8192)

        progress_bar.close()

        total_downloaded = downloaded["n"]
        if total_size:
            percent = (total_downloaded / total_size) * 100
            say_ok(
                f"Unduh selesai: {format_size(total_downloaded)} / "
                f"{format_size(total_size)} ({percent:.1f}%)"
            )
        else:
            say_ok(
                f"Unduh selesai: {format_size(total_downloaded)} "
                f"(ukuran total tidak diketahui)"
            )

        return total_downloaded

    def quit(self):
        """Tutup koneksi FTP dengan baik."""
        try:
            if self._ftp:
                self._ftp.quit()
        except Exception:
            pass

    def reconnect(self):
        """Paksa tutup & bikin koneksi FTP baru dari nol.
        Dipakai setelah kegagalan (timeout, error_reply, dll), karena
        is_alive()/NOOP saja tidak cukup untuk mendeteksi socket kontrol
        yang 'desync' akibat balasan server yang belum sempat dibaca.
        """
        say_error("Menutup koneksi lama dan menyambung ulang dari nol ...")
        logging.info("Reconnect paksa: menutup socket lama.")
        try:
            if self._ftp:
                self._ftp.close()
        except Exception:
            pass
        self._ftp = None
        self.connect()