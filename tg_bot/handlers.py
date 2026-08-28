from aiogram import Bot, Router
from aiogram.filters import Command, CommandStart
from aiogram.types import Message, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from datetime import datetime
from tg_bot.models import StateStore
from tg_bot.processing import run_processing
from tg_bot.requests import search_request
from tg_bot.config import Config
import asyncio
from aiogram.utils.keyboard import InlineKeyboardBuilder
from collections import defaultdict
from aiogram.types import CallbackQuery
from uuid import uuid4


router = Router()

MAX_SEARCH_RETRIES = 3
# chat_id -> {"query": str, "excluded": list[str], "attempts": int, "last_ids": list[str]}
last_search_state: dict[str, dict] = {}

media_group_buffers: dict[str, list[Message]] = defaultdict(list)
media_group_tasks: dict[str, asyncio.Task] = {}

menu_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Поиск")],
        [KeyboardButton(text="Обработать новые")],
        [KeyboardButton(text="Помощь")]
    ],
    resize_keyboard=True
)

@router.message(lambda m: m.text == "Поиск")
async def search_button(message: Message):
    await message.answer("Введите запрос в формате: /search <текст>")

@router.message(lambda m: m.text == "Обработать новые")
async def process_button(message: Message):
    await cmd_process(message)

@router.message(lambda m: m.text == "Помощь")
async def help_button(message: Message):
    await cmd_help(message)

@router.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer("Привет!\nЯ агент-сыщик для поиска сообщений.\nВызови команду /help для ознакомления с командами.", reply_markup=menu_keyboard)


@router.message(Command("help"))
async def cmd_help(message: Message):
    help_text = (
        "*Доступные команды:*\n\n"
        "/start — запустить бота\n"
        "/help — показать это сообщение\n"
        "/process — отправить новые сообщения на processing‑сервер\n"
        "/search <текст> — найти сообщение через RAG\n"
    )
    await message.answer(help_text, parse_mode="Markdown")


async def flush_media_group(group_key: str, bot: Bot):
    try:
        await asyncio.sleep(1.0)
        group = media_group_buffers.pop(group_key, [])
        group.sort(key=lambda m: m.message_id)
        for part in group:
            await save_single_message(part, bot)
    except Exception:
        print(f"Ошибка при обработке группы {group_key}")

    finally:
        media_group_tasks.pop(group_key, None)

@router.message(~Command("process", "start", "help", "search"))
async def save_message(message: Message, bot: Bot):
    if message.chat.type == "private":
        return
    if message.text and message.text.startswith("/"):
        return

    chat_id = str(message.chat.id)

    if message.media_group_id:
        group_key = f"{chat_id}_{message.media_group_id}"
        media_group_buffers[group_key].append(message)
        if group_key not in media_group_tasks:
            media_group_tasks[group_key] = asyncio.create_task(flush_media_group(group_key, bot))
    else:
        await save_single_message(message, bot)

async def save_single_message(message: Message, bot: Bot):
    chat_id = str(message.chat.id)
    msg_id = str(message.message_id)
    text = message.caption or message.text or ""
    msg_type = "unknown"
    file_urls = []
    print(f"Сохраняем сообщение {msg_id} в чат {chat_id}, тип: {message.content_type}")
    if message.photo:
        msg_type = "photo"
        photo = message.photo[-1]
        file = await bot.get_file(photo.file_id)
        file_url = f"https://api.telegram.org/file/bot{Config.TOKEN}/{file.file_path}"
        file_urls.append(file_url)


    elif message.document:
        msg_type = "document"
        file = await bot.get_file(message.document.file_id)
        file_url = f"https://api.telegram.org/file/bot{Config.TOKEN}/{file.file_path}"
        file_urls.append(file_url)

    elif message.video:
        msg_type = "video"
        file = await bot.get_file(message.video.file_id)
        file_url = f"https://api.telegram.org/file/bot{Config.TOKEN}/{file.file_path}"
        file_urls.append(file_url)

    elif message.voice:
        msg_type = "voice"
        file = await bot.get_file(message.voice.file_id)
        file_url = f"https://api.telegram.org/file/bot{Config.TOKEN}/{file.file_path}"
        file_urls.append(file_url)

    elif message.text:
        msg_type = "text"

    # elif message.video:
    #     msg_type = "video"
    #     file = await bot.get_file(video.file_id)
    #     file_url = f"https://api.telegram.org/file/bot{Config.TOKEN}/{file.file_path}"
    #     file_urls.append(file_url)
    print(f"Сохраняем сообщение {msg_id} типа {msg_type} с текстом: {text[:50]}... и файлами: {file_urls}")

    record = {
        "message_type": msg_type,
        "text": text,
        "file_urls": file_urls,
        "timestamp": (message.date or datetime.now()).isoformat(),
    }
    await StateStore.add_message(chat_id, msg_id, record)

@router.message(Command("process"))
async def cmd_process(message: Message):
    if message.chat.type == "private":
        await message.answer("Команда работает только в групповых чатах.")
        return
    print("Начинаем обработку новых сообщений...")
    summary = await run_processing()
    print(f"Обработка завершена: {summary}")
    await message.answer(summary)

@router.message(Command("search"))
async def cmd_search(message: Message, bot: Bot):
    if message.chat.type == "private":
        await message.answer("Поиск работает только в групповых чатах.")
        return
    parts = message.text.split(maxsplit=1)
    if len(parts) < 2:
        await message.answer("Укажите запрос: /search <текст>")
        return

    query = parts[1]
    chat_id = str(message.chat.id)
    # свежий /search сбрасывает историю повторов
    last_search_state[chat_id] = {"query": query, "excluded": [], "attempts": 0, "last_ids": []}
    payload = {chat_id: {"request": query, "repeat": False, "exclude_ids": [], "attempt": 0}}
    trace_id = uuid4().hex[:10]
    print(f"[TRACE {trace_id}] /search received chat_id={chat_id} query={query!r}")
    try:
        ok, shown_ids = await search_request(payload, message, bot, trace_id=trace_id)
        if ok:
            last_search_state[chat_id]["last_ids"] = [str(i) for i in shown_ids]
            await message.answer("Подошёл ли ответ?", reply_markup=feedback_keyboard())
    except Exception as e:
        await message.answer(f"Ошибка поиска: {e}")



def feedback_keyboard():
    builder = InlineKeyboardBuilder()
    builder.row(
        InlineKeyboardButton(text="Да", callback_data="search_feedback:yes"),
    )
    builder.row(
        InlineKeyboardButton(text="Нет, повторить запрос", callback_data="search_feedback:no")
    )
    return builder.as_markup()


@router.callback_query(lambda c: c.data.startswith("search_feedback:"))
async def process_feedback(callback: CallbackQuery, bot: Bot):
    action = callback.data.split(":")[1]
    chat_id = str(callback.message.chat.id)
    st = last_search_state.get(chat_id)

    if not st:
        await callback.answer("Не найден исходный запрос", show_alert=True)
        return

    if action == "yes":
        await callback.answer("Спасибо за подтверждение")
        await callback.message.delete()
        last_search_state.pop(chat_id, None)
        return

    # action == "no": показанные сообщения не подошли -> исключаем их и ищем заново
    await callback.message.delete()
    st["excluded"] = list(dict.fromkeys(st["excluded"] + st.get("last_ids", [])))
    st["attempts"] += 1

    if st["attempts"] > MAX_SEARCH_RETRIES:
        await callback.message.answer(
            "Не нашёл других подходящих сообщений. Попробуйте переформулировать запрос."
        )
        last_search_state.pop(chat_id, None)
        await callback.answer()
        return

    payload = {
        chat_id: {
            "request": st["query"],
            "repeat": True,
            "exclude_ids": st["excluded"],
            "attempt": st["attempts"],
        }
    }
    trace_id = uuid4().hex[:10]
    print(f"[TRACE {trace_id}] /search retry chat_id={chat_id} attempt={st['attempts']} "
          f"excluded={len(st['excluded'])}")
    try:
        ok, shown_ids = await search_request(payload, callback.message, bot, trace_id=trace_id)
        if ok:
            st["last_ids"] = [str(i) for i in shown_ids]
            await callback.message.answer(
                f"Повторный поиск (попытка {st['attempts']}). Подошёл ли ответ?",
                reply_markup=feedback_keyboard(),
            )
        else:
            await callback.message.answer("Повторный поиск не дал новых результатов.")
    except Exception as e:
        await callback.message.answer(f"Ошибка при повторном поиске: {e}")

    await callback.answer()
