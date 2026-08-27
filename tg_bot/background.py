from datetime import datetime

from tg_bot.processing import run_processing


async def process_new_messages(bot):
    print(f"{datetime.now()}: Фоновая обработка стартовала")
    summary = await run_processing()
    print(f"{datetime.now()}: Фоновая обработка завершена — {summary}")
