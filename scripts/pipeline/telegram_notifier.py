# ./scripts/pipeline/telegram_notifier.py
# Menyediakan utilitas untuk mengirim notifikasi proses pipeline melalui Telegram, termasuk pelaporan keberhasilan dan kegagalan saat mendownload dataset netcdf.

import logging
import time
import requests


class TelegramNotifier:
    """Kirim notifikasi Telegram real-time (per file gagal & per direktori selesai); nonaktif otomatis kalau .env kosong, dan gagal kirim tidak menghentikan pipeline."""

    API_BASE = "https://api.telegram.org"
    TIMEOUT = 15
    MAX_SEND_RETRY = 3  

    def __init__(self, bot_token, chat_id):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = bool(bot_token) and bool(chat_id)

        if not self.enabled:
            logging.info(
                "Notifikasi Telegram TIDAK aktif (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID belum diset di .env)"
            )

    def _send(self, text):
        if not self.enabled:
            return
        url = f"{self.API_BASE}/bot{self.bot_token}/sendMessage"

        for attempt in range(1, self.MAX_SEND_RETRY + 1):
            try:
                resp = requests.post(
                    url,
                    data={
                        "chat_id": self.chat_id,
                        "text": text,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=self.TIMEOUT,
                )

                if resp.status_code == 200:
                    return

                if resp.status_code == 429:
                    retry_after = 3
                    try:
                        retry_after = resp.json().get("parameters", {}).get("retry_after", 3)
                    except Exception:
                        pass
                    logging.warning(
                        f"Kena rate limit Telegram (429), tunggu {retry_after}s lalu coba lagi "
                        f"(percobaan {attempt}/{self.MAX_SEND_RETRY})"
                    )
                    time.sleep(retry_after + 0.5)
                    continue

                logging.warning(
                    f"Gagal kirim notifikasi Telegram (HTTP {resp.status_code}): {resp.text}"
                )
                return

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < self.MAX_SEND_RETRY:
                    logging.warning(
                        f"Timeout/koneksi gagal saat kirim notifikasi Telegram, coba lagi "
                        f"(percobaan {attempt}/{self.MAX_SEND_RETRY}): {e}"
                    )
                    time.sleep(2)
                    continue
                logging.warning(
                    f"Gagal kirim notifikasi Telegram setelah {self.MAX_SEND_RETRY}x percobaan (timeout/koneksi): {e}"
                )
                return

            except Exception as e:
                logging.warning(f"Gagal kirim notifikasi Telegram: {e}")
                return

        logging.warning(
            f"Notifikasi Telegram gagal terkirim setelah {self.MAX_SEND_RETRY}x percobaan (rate limit terus-menerus)"
        )

    def notify_directory_done(self, remote_dir, dir_tally, dir_failed_files):
        """Kirim ringkasan setelah satu direktori selesai diproses."""
        total_ok = dir_tally["berhasil_ada_data"] + dir_tally["berhasil_tanpa_data"]
        total_file = total_ok + dir_tally["dilewati"] + dir_tally["gagal"]

        if total_file == 0:
            return

        if dir_tally["gagal"] == 0:
            text = (
                "✅ <b>Direktori selesai</b>\n"
                f"📁 <code>{remote_dir}</code>\n"
                f"• Ada data Bandung    : {dir_tally['berhasil_ada_data']}\n"
                f"• Tanpa data          : {dir_tally['berhasil_tanpa_data']}\n"
                f"• Dilewati            : {dir_tally['dilewati']}\n"
                f"• Total file          : {total_file}"
            )
        else:
            failed_list = "\n".join(f"  - {f}" for f in dir_failed_files) or "  (tidak ada detail)"
            text = (
                "⚠️ <b>Direktori selesai (dengan error)</b>\n"
                f"📁 <code>{remote_dir}</code>\n"
                f"• Berhasil    : {total_ok}\n"
                f"• Gagal       : {dir_tally['gagal']}\n"
                f"• Dilewati    : {dir_tally['dilewati']}\n"
                f"File gagal    :\n{failed_list}"
            )

        self._send(text)

    def notify_file_error(self, remote_dir, file, error_msg):
        """Kirim notifikasi segera (real-time) saat satu file gagal total."""
        text = (
            "❌ <b>Gagal download file</b>\n"
            f"📁 Folder : <code>{remote_dir}</code>\n"
            f"📄 File   : <code>{file}</code>\n"
            f"🔺 Error  : {error_msg}"
        )
        self._send(text)