"""Общий конвейер отправки новых сообщений на processor_server.

Используется и командой /process, и фоновым планировщиком, чтобы логика была одна.
"""

from tg_bot.models import StateStore
from tg_bot.requests import process_new_messages_request


async def run_processing() -> str:
    new_by_chat = await StateStore.collect_new_messages()
    if not new_by_chat:
        return "Нет новых сообщений в сохранённых чатах."

    total_ok = 0
    errors: list[str] = []
    for chat_id, new_msgs in new_by_chat.items():
        payload = {"chat_id": int(chat_id), "messages": new_msgs}
        ok, err = await process_new_messages_request(payload, chat_id)
        if ok:
            # last_processed двигаем и чистим chat_data только при подтверждённом успехе
            await StateStore.mark_processed(chat_id, list(new_msgs.keys()))
            total_ok += len(new_msgs)
        else:
            errors.append(f"Чат {chat_id}: {err}")

    text = f"Отправлено {total_ok} новых сообщений."
    if errors:
        text += "\nОшибки:\n" + "\n".join(errors)
    return text
