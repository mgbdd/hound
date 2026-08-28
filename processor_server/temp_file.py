import logging
import os
import tempfile

import requests

logger = logging.getLogger(__name__)

_TELEGRAM_FILE_BASE = "https://api.telegram.org/file/bot"
_DOWNLOAD_TIMEOUT = 60  # секунд


def _to_url(file_ref: str) -> str | None:
    """file_ref — либо полный URL (легаси chat_data.json), либо относительный путь
    Telegram ('photos/file_123.jpg'). Токен в хранилище не лежит, подставляем здесь."""
    if file_ref.startswith("http"):
        return file_ref
    token = os.getenv("TG_BOT_TOKEN", "")
    if not token:
        logger.error("TG_BOT_TOKEN не задан — не могу собрать URL для %s", file_ref)
        return None
    return f"{_TELEGRAM_FILE_BASE}{token}/{file_ref.lstrip('/')}"


def load_temp_files(file_ref: str):
    url = _to_url(file_ref)
    if not url:
        return None

    # Уникальное имя (раньше basename(url) -> коллизии между чатами/сообщениями).
    suffix = os.path.splitext(file_ref.split("?")[0])[1]
    fd, temp_file_path = tempfile.mkstemp(prefix="hound_", suffix=suffix)
    os.close(fd)

    try:
        response = requests.get(url, timeout=_DOWNLOAD_TIMEOUT)
        response.raise_for_status()
        with open(temp_file_path, "wb") as file:
            file.write(response.content)
    except requests.exceptions.RequestException as e:
        logger.warning("Не удалось загрузить файл %s: %s", file_ref, e)
        try:
            os.remove(temp_file_path)
        except OSError:
            pass
        return None

    return temp_file_path


def delete_temp_files(paths: list[str] | str | None) -> None:
    if paths is None:
        return
    if isinstance(paths, str):
        paths = [paths]

    for p in paths:
        try:
            os.remove(p)
        except FileNotFoundError:
            pass
