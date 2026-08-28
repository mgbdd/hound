import asyncio
import logging
import os

from aiogram import Bot, Dispatcher
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from tg_bot.config import Config
from tg_bot.models import JsonStorage
from tg_bot.background import process_new_messages
from tg_bot.handlers import router

logging.basicConfig(level=logging.INFO)

bot = Bot(token=Config.TOKEN)
dp = Dispatcher()

async def on_startup():
    for fname in ("chat_data.json", "last_processed.json"):
        if not os.path.exists(fname):
            JsonStorage.save(fname, {})

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        process_new_messages,
        IntervalTrigger(minutes=5),
        args=[bot],
        max_instances=1
    )
    scheduler.start()
    logging.info("Бот запущен и планировщик стартовал")

async def main():
    dp.include_router(router)
    dp.startup.register(on_startup)
    await dp.start_polling(bot)

if __name__ == '__main__':
    asyncio.run(main())
