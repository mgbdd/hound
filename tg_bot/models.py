from aiogram import types
import asyncio
import json
import os
import tempfile
from datetime import datetime
from typing import Dict

CHAT_DATA_FILE = "chat_data.json"
LAST_PROCESSED_FILE = "last_processed.json"


class JsonStorage:
    @staticmethod
    def load(filename: str) -> Dict:
        try:
            with open(filename, "r", encoding="UTF-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    @staticmethod
    def save(filename: str, data: Dict):
        """Атомарная запись: пишем во временный файл рядом и заменяем цель через os.replace,
        чтобы параллельный читатель никогда не увидел половину файла."""
        directory = os.path.dirname(os.path.abspath(filename)) or "."
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="UTF-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            os.replace(tmp, filename)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise


class StateStore:
    """Единая точка доступа к chat_data.json / last_processed.json.

    Все операции с файлами сериализованы одним локом. Медленные сетевые вызовы
    (отправка на processor_server) делаются ВНЕ лока — см. tg_bot/processing.py.
    """

    _lock = asyncio.Lock()

    @classmethod
    async def add_message(cls, chat_id: str, msg_id: str, record: dict) -> None:
        async with cls._lock:
            data = JsonStorage.load(CHAT_DATA_FILE)
            chat = data.setdefault(chat_id, {"chat_id": int(chat_id), "messages": {}})
            chat.setdefault("messages", {})[msg_id] = record
            JsonStorage.save(CHAT_DATA_FILE, data)

    @classmethod
    async def collect_new_messages(cls) -> dict[str, dict]:
        """{chat_id: {msg_id: record}} для сообщений с int(msg_id) > last_processed[chat_id].
        Ничего не удаляет — удаление только после подтверждённой обработки (mark_processed)."""
        async with cls._lock:
            data = JsonStorage.load(CHAT_DATA_FILE)
            last = JsonStorage.load(LAST_PROCESSED_FILE)
            out: dict[str, dict] = {}
            for chat_id, chat in data.items():
                last_id = int(last.get(chat_id, 0))
                new = {
                    mid: rec
                    for mid, rec in chat.get("messages", {}).items()
                    if int(mid) > last_id
                }
                if new:
                    out[chat_id] = new
            return out

    @classmethod
    async def mark_processed(cls, chat_id: str, processed_ids: list[str]) -> None:
        """Помечает processed_ids обработанными и удаляет их из chat_data.

        Состояние перечитывается под локом, поэтому сообщения, пришедшие во время
        обработки (у них message_id больше), не теряются и попадут в следующий проход.
        """
        if not processed_ids:
            return
        ids_int = [int(m) for m in processed_ids]
        async with cls._lock:
            data = JsonStorage.load(CHAT_DATA_FILE)
            last = JsonStorage.load(LAST_PROCESSED_FILE)

            chat = data.get(chat_id)
            if chat:
                messages = chat.get("messages", {})
                for mid in processed_ids:
                    messages.pop(mid, None)
                if not messages:
                    data.pop(chat_id, None)

            last[chat_id] = max(int(last.get(chat_id, 0)), max(ids_int))
            JsonStorage.save(CHAT_DATA_FILE, data)
            JsonStorage.save(LAST_PROCESSED_FILE, last)


class MessageData:
    def __init__(self, message: types.Message):
        self.message_id = message.message_id
        self.chat_id = message.chat.id
        self.date = message.date.isoformat() if message.date else datetime.now().isoformat()
        self.text = message.text or ""
        self.user_id = message.from_user.id if message.from_user else None
        self.user_name = message.from_user.full_name if message.from_user else "Unknown"

    def to_dict(self):
        return {
            "message_id": self.message_id,
            "chat_id": self.chat_id,
            "date": self.date,
            "text": self.text,
            "user_id": self.user_id,
            "user_name": self.user_name
        }
