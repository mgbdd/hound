from aiogram.types import Message
from aiogram import Bot
from aiogram.exceptions import TelegramRetryAfter, TelegramBadRequest
import socket
import json
import asyncio
import hashlib
import logging
import requests
from typing import Any

from langgraph_sdk import get_client
from tg_bot.config import Config

log = logging.getLogger("hound.tg_bot")

_assistant_id_cache: str | None = None


def _extract_answer_payload(run_result: Any) -> dict[str, Any]:
    # Граф (finalize) отдаёт финальный стейт с message_ids/answer_text наверху.
    if isinstance(run_result, dict):
        return run_result
    return {}


async def _get_assistant_id(client) -> str:
    global _assistant_id_cache
    if _assistant_id_cache:
        return _assistant_id_cache

    assistants = await client.assistants.search(graph_id=Config.RAG_GRAPH_ID, limit=1)
    if assistants:
        _assistant_id_cache = assistants[0]["assistant_id"]
        return _assistant_id_cache

    assistant = await client.assistants.create(
        graph_id=Config.RAG_GRAPH_ID,
        name="tg-bot-search-assistant",
    )
    _assistant_id_cache = assistant["assistant_id"]
    return _assistant_id_cache


async def _wait_for_rag_server(base_url: str, attempts: int = 20, delay_seconds: float = 2.0) -> bool:
    """Подстраховка на случай, если agent перезапускается во время работы бота.
    Холодный старт гейтит compose (depends_on: condition: service_healthy)."""
    docs_url = f"{base_url.rstrip('/')}/docs"
    for attempt in range(1, attempts + 1):
        try:
            response = await asyncio.to_thread(requests.get, docs_url, timeout=5)
            if response.status_code < 500:
                return True
        except Exception:
            pass

        if attempt < attempts:
            await asyncio.sleep(delay_seconds)
    return False

async def process_new_messages_request(payload: dict, chat_id: str) -> tuple[bool, str | None]:
    """Отправляет батч сообщений на processor_server. Возвращает (ok, error).

    ok=True только если процессор явно подтвердил приём ("Upload successful"). Любой
    другой ответ / обрыв / исключение -> (False, текст ошибки), и вызывающий код
    НЕ должен двигать last_processed.
    """

    def send_socket_data() -> tuple[bool, str | None]:
        try:
            with socket.create_connection(('processor_server', 3030), timeout=1000) as sock:
                sock.sendall(json.dumps(payload).encode("utf-8"))
                sock.shutdown(socket.SHUT_WR)

                chunks = []
                while True:
                    data = sock.recv(4096)
                    if not data:
                        break
                    chunks.append(data)
                response = b"".join(chunks).decode("utf-8", errors="replace")
        except Exception as e:
            return False, str(e)

        if "Upload successful" in response:
            return True, None
        return False, f"неожиданный ответ процессора: {response!r}"

    ok, err = await asyncio.to_thread(send_socket_data)
    log.info(f"[processor] чат {chat_id}: ok={ok} err={err}")
    return ok, err

async def search_request(
    payload: dict, message: Message, bot: Bot, trace_id: str | None = None
) -> tuple[bool, list[str]]:
    """Возвращает (ok, shown_ids). shown_ids — message_id, показанные пользователю
    (форварднутые либо процитированные в текстовом ответе); нужны для исключения
    при повторном поиске."""
    try:
        trace = trace_id or "no-trace"
        # payload: {chat_id: {"request": q, "repeat": bool, "exclude_ids": [...], "attempt": int}}
        chat_id_key = next(iter(payload))
        sub = payload.get(chat_id_key) or {}
        input_state = {
            "user_query": sub.get("request") or sub.get("query") or "",
            "chat_id": str(chat_id_key),
            "excluded_message_ids": [str(x) for x in (sub.get("exclude_ids") or [])],
            "attempt_count": int(sub.get("attempt") or 0),
            "repeat_rag": bool(sub.get("repeat", False)),
        }
        payload_fingerprint = hashlib.sha1(
            json.dumps(input_state, ensure_ascii=False).encode("utf-8")
        ).hexdigest()[:10]
        log.info(f"[TRACE {trace}] search_request send chat_id={message.chat.id} payload_sha={payload_fingerprint}")
        rag_ready = await _wait_for_rag_server(Config.RAG_SERVER_URL)
        if not rag_ready:
            await message.answer("Сервер поиска пока запускается. Попробуйте снова через 20-30 секунд.")
            return False, []
        client = get_client(url=Config.RAG_SERVER_URL)
        assistant_id = await _get_assistant_id(client)
        thread = await client.threads.create(metadata={"chat_id": str(message.chat.id)})
        run_result = await client.runs.wait(
            thread_id=thread["thread_id"],
            assistant_id=assistant_id,
            input=input_state,
        )
        result = _extract_answer_payload(run_result)
        log.info(f"[TRACE {trace}] search_request recv payload_sha={payload_fingerprint} response={result!r}")
        if not result:
            await message.answer("Пустой ответ от LangGraph API.")
            return False, []

        cited_ids = [str(x).strip() for x in (result.get("message_ids") or []) if str(x).strip()]

        answer_text = result.get("answer_text")
        if isinstance(answer_text, str) and answer_text.strip():
            log.info(f"[TRACE {trace}] answer_text payload_sha={payload_fingerprint}")
            await message.answer(answer_text.strip())
            return True, cited_ids

        message_items = result.get('message_ids')
        if message_items is None:
            log.info(f"[TRACE {trace}] missing message_ids payload_sha={payload_fingerprint}")
            await message.answer("В ответе нет поля message_ids.")
            return False, []

        # Нормализуем в список
        if isinstance(message_items, (str, int)):
            message_items = [message_items]
        elif not isinstance(message_items, list):
            await message.answer(f"Поле message_ids имеет неподдерживаемый тип: {type(message_items).__name__}")
            return False, []

        source_chat_id = message.chat.id
        numeric_ids: list[int] = []
        for item in message_items:
            try:
                numeric_ids.append(int(item))
            except (ValueError, TypeError):
                # Ignore non-numeric placeholders like "(пусто)"
                continue

        # No numeric message_id -> treat as unsuccessful search.
        if not numeric_ids:
            log.info(f"[TRACE {trace}] no numeric message_ids payload_sha={payload_fingerprint}")
            await message.answer("Ничего не найдено.")
            return False, []

        log.info(f"[TRACE {trace}] forwarding {len(numeric_ids)} ids payload_sha={payload_fingerprint}")
        forwarded: list[str] = []
        for idx, msg_id in enumerate(numeric_ids):
            for attempt in range(2):
                try:
                    await bot.forward_message(
                        chat_id=message.chat.id,
                        from_chat_id=source_chat_id,
                        message_id=msg_id,
                    )
                    forwarded.append(str(msg_id))
                    break
                except TelegramRetryAfter as e:
                    log.info(f"[TRACE {trace}] flood control, sleep {e.retry_after}s (msg {msg_id})")
                    await asyncio.sleep(e.retry_after + 0.5)
                except TelegramBadRequest as e:
                    log.warning(f"[TRACE {trace}] forward {msg_id} failed: {e}")
                    break
            if idx < len(numeric_ids) - 1:
                await asyncio.sleep(0.1)  # мягкий троттлинг, чтобы не ловить flood control

        return True, forwarded

    except Exception as e:
        log.warning(f"[Ошибка сокет-соединения] Чат {message.chat.id}: {e}")
        await message.answer(f"Ошибка при поиске: {e}")
        return False, []
