# ./scripts/01_download_data.py
# Mengunduh data NetCDF dari FTP secara rekursif, melakukan subset wilayah Bandung, memvalidasi hasil, serta mencatat status proses dan mengirimkan notifikasi Telegram.

import os
import time
import logging

from ui.terminal_display import (
    hr, gap, banner,
    say_info, say_download, say_ok, say_skip, say_error, say_fatal,
    make_total_progress_bar,
)
from pipeline.config import load_config
from pipeline.ftp_client import FTPClient
from pipeline.netcdf_tools import subset_bandung, validate_downloaded_file
from pipeline.file_tracker import FileTracker
from pipeline.utils import get_paths_for_remote_dir
from pipeline.telegram_notifier import TelegramNotifier


def setup_logging(log_file):
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def process_one_file(ftp_client, file, remote_dir, tracker, checked_files, tally, failed_files, cfg, notifier):
    """Mengunduh, memproses subset, memvalidasi, dan mencatat status satu file."""
    final_dir, _ = get_paths_for_remote_dir(
        remote_dir, cfg.FINAL_BASE_DIR, cfg.METADATA_BASE_DIR
    )
    target_path = os.path.join(final_dir, f"subset_{file}")

    # Already processed?
    if os.path.exists(target_path):
        logging.info(f"FILE SUDAH ADA DI DATASET: {file}")
        if file not in checked_files:
            tracker.append_checked(remote_dir, file)
            checked_files.add(file)
        tally["dilewati"] += 1
        return

    # Already checked but no data?
    if file in checked_files:
        logging.info(f"SDH DIPERIKSA, TDK ADA BANDUNG: {file}")
        tally["dilewati"] += 1
        return

    gap()
    say_download(f"[{remote_dir}] {file}  (percobaan 1/{cfg.MAX_RETRY})")

    success = False
    last_error = None
    already_notified = False
    for attempt in range(1, cfg.MAX_RETRY + 1):
        try:
            ftp_client.ensure_connected()
            ftp_client.cwd(remote_dir)

            if attempt > 1:
                say_download(f"[{remote_dir}] {file}  (percobaan {attempt}/{cfg.MAX_RETRY})")

            downloaded_bytes = ftp_client.download_with_progress(file, cfg.TEMP_FILE)
            logging.info(f"DOWNLOAD SELESAI: {file}")

            validate_downloaded_file(cfg.TEMP_FILE)

            has_data = subset_bandung(cfg.TEMP_FILE, target_path)
            tally["berhasil_ada_data" if has_data else "berhasil_tanpa_data"] += 1

            tracker.append_checked(remote_dir, file)
            checked_files.add(file)
            success = True
            break

        except ValueError as e:
            say_error(f"File tidak valid, dilewati: {e}")
            logging.error(f"Struktur data tidak sesuai untuk {file}: {e}")
            last_error = str(e)
            notifier.notify_file_error(remote_dir, file, f"Struktur data tidak valid: {e}")
            already_notified = True
            break

        except Exception as e:
            say_error(f"Percobaan {attempt} gagal: {e}")
            logging.warning(f"Percobaan {attempt} gagal untuk {file}: {e}")
            last_error = str(e)
            ftp_client.mark_dirty()

            try:
                ftp_client.reconnect()
            except Exception as reconnect_err:
                logging.error(f"Reconnect juga gagal: {reconnect_err}")
            if attempt < cfg.MAX_RETRY:
                time.sleep(5)

    if not success:
        logging.error(f"GAGAL TOTAL setelah {cfg.MAX_RETRY} percobaan: {file}")
        say_fatal(f"Melewati {file} setelah {cfg.MAX_RETRY}x gagal")
        tally["gagal"] += 1
        failed_files.append(f"{remote_dir}/{file}")
        # Notifikasi real-time -- tapi jangan kirim dobel kalau sudah
        # dinotif sebagai ValueError (struktur data) di atas.
        if not already_notified:
            notifier.notify_file_error(
                remote_dir, file, f"Gagal setelah {cfg.MAX_RETRY}x percobaan: {last_error}"
            )

    if os.path.exists(cfg.TEMP_FILE):
        os.remove(cfg.TEMP_FILE)


def process_directory(ftp_client, remote_dir, tracker, tally, failed_files, cfg, notifier):
    """Recursively process FTP directory."""
    say_info(f"Menelusuri: {remote_dir}")

    final_dir, _ = get_paths_for_remote_dir(
        remote_dir, cfg.FINAL_BASE_DIR, cfg.METADATA_BASE_DIR
    )
    os.makedirs(final_dir, exist_ok=True)

    files, subdirs = ftp_client.list_directory(remote_dir)

    # Filter files
    if cfg.FILENAME_MUST_CONTAIN:
        filtered = [f for f in files if cfg.FILENAME_MUST_CONTAIN in f]
        skipped = len(files) - len(filtered)
        if skipped > 0:
            say_info(f"{skipped} file dilewati (tidak mengandung '{cfg.FILENAME_MUST_CONTAIN}')")
        files = filtered

    say_info(f"Ditemukan {len(files)} file .nc, {len(subdirs)} sub-folder")
    hr()

    # Load checked_files.log ONCE for this directory, not once per file.
    checked_files = tracker.load_checked(remote_dir)

    # Tally/failed-list KHUSUS untuk direktori ini saja (bukan akumulasi
    # global), supaya notifikasi Telegram yang dikirim setelah direktori
    # ini selesai hanya melaporkan hasil direktori ini -- bukan tercampur
    # dengan direktori lain yang sudah diproses sebelumnya.
    dir_tally = {"berhasil_ada_data": 0, "berhasil_tanpa_data": 0, "dilewati": 0, "gagal": 0}
    dir_failed_files = []

    for file in files:
        process_one_file(ftp_client, file, remote_dir, tracker, checked_files, dir_tally, dir_failed_files, cfg, notifier)

    # Gabungkan hasil direktori ini ke tally & failed_files global (dipakai
    # untuk ringkasan akhir di terminal, seperti sebelumnya).
    for key in tally:
        tally[key] += dir_tally[key]
    failed_files.extend(dir_failed_files)

    # Kirim notifikasi Telegram real-time: direktori ini sudah selesai
    # (semua filenya, belum termasuk subdirektori).
    notifier.notify_directory_done(remote_dir, dir_tally, dir_failed_files)

    for subdir in subdirs:
        sub_remote = f"{remote_dir.rstrip('/')}/{subdir}"
        process_directory(ftp_client, sub_remote, tracker, tally, failed_files, cfg, notifier)


def print_summary(tally, failed_files, log_file):
    gap()
    banner("RINGKASAN AKHIR")
    say_ok(f"Berhasil, ada data Bandung : {tally['berhasil_ada_data']}")
    say_skip(f"Berhasil, tanpa data       : {tally['berhasil_tanpa_data']}")
    say_skip(f"Dilewati (sudah diproses)  : {tally['dilewati']}")
    if tally["gagal"] > 0:
        say_fatal(f"Gagal total                : {tally['gagal']}")
        say_info(f"Detail lengkap ada di: {log_file}")
        for f in failed_files:
            print(f"    - {f}")
    else:
        say_ok("Gagal total                : 0")
    hr()


def main():
    cfg = load_config()
    setup_logging(cfg.LOG_FILE)

    os.makedirs(cfg.FINAL_BASE_DIR, exist_ok=True)
    os.makedirs(cfg.METADATA_BASE_DIR, exist_ok=True)

    banner("DOWNLOAD DATA NETCDF - RECURSIVE MODE")
    say_info(f"Base FTP dir    : {cfg.FTP_REMOTE_BASE}")
    say_info(f"Filter nama file: '{cfg.FILENAME_MUST_CONTAIN}'")
    say_info(f"Folder data     : {cfg.FINAL_BASE_DIR}")
    say_info(f"Folder metadata : {cfg.METADATA_BASE_DIR}")
    hr()

    ftp_client = FTPClient(cfg.FTP_HOST, cfg.FTP_USER, cfg.FTP_PASS, cfg.FTP_TIMEOUT)
    ftp_client.connect()

    tracker = FileTracker(cfg.METADATA_BASE_DIR)
    notifier = TelegramNotifier(cfg.TELEGRAM_BOT_TOKEN, cfg.TELEGRAM_CHAT_ID)

    tally = {"berhasil_ada_data": 0, "berhasil_tanpa_data": 0, "dilewati": 0, "gagal": 0}
    failed_files = []

    process_directory(ftp_client, cfg.FTP_REMOTE_BASE, tracker, tally, failed_files, cfg, notifier)

    ftp_client.quit()
    print_summary(tally, failed_files, cfg.LOG_FILE)


if __name__ == "__main__":
    main()