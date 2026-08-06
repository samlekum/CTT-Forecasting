# ./scripts/ui/terminal_display.py
# Menyediakan utilitas untuk menampilkan informasi, log, dan progress bar secara konsisten pada terminal.

import sys
from tqdm import tqdm

_USE_COLOR = sys.stdout.isatty()

def _c(code, text):
    """Bungkus teks dengan kode warna ANSI, kalau warna sedang aktif."""
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text

def hr(char="─", width=60):
    """Cetak garis horizontal sebagai pemisah section."""
    tqdm.write(_c("90", char * width))

def gap():
    """Baris kosong sebagai jeda antar sesi (misal antar download)."""
    tqdm.write("")

def banner(title):
    """Judul section dengan garis di atas & bawah."""
    hr()
    tqdm.write(_c("1;36", f" {title}"))
    hr()

def say(tag, msg, color="37"):
    """
    Cetak satu baris log dengan tag rata kiri, rapi dan konsisten.
    Pakai tqdm.write() (bukan print biasa) supaya tidak bentrok/tabrakan
    dengan progress bar tqdm yang sedang aktif di baris bawah terminal.
    """
    tqdm.write(f"{_c(color, f'[{tag:<8}]')} {msg}")

def say_info(msg):
    say("INFO", msg, "36")

def say_login(msg):
    say("LOGIN", msg, "34")

def say_download(msg):
    say("DOWNLOAD", msg, "33")

def say_ok(msg):
    say("OK", msg, "32")

def say_skip(msg):
    say("SKIP", msg, "90")

def say_error(msg):
    say("ERROR", msg, "31")

def say_fatal(msg):
    say("FATAL", msg, "1;31")

def make_download_progress_bar(total_size):
    """Buat progress bar tqdm khusus untuk proses unduh satu file."""
    return tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=_c("33", "  ↳ unduh"),
        bar_format="{desc} |{bar}| {percentage:3.0f}% {n_fmt}/{total_fmt} [{rate_fmt}]",
        leave=False,
        ncols=80,
    )

def make_total_progress_bar(files):
    """Buat progress bar tqdm untuk progres keseluruhan (semua file)."""
    return tqdm(
        files,
        desc="Total progres",
        bar_format="{desc} |{bar}| {n_fmt}/{total_fmt} file [{elapsed}<{remaining}]",
        ncols=80,
    )