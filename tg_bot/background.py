import logging

from tg_bot.processing import run_processing

log = logging.getLogger("hound.tg_bot")


async def process_new_messages(bot):
    log.info("Фоновая обработка стартовала")
    summary = await run_processing()
    log.info("Фоновая обработка завершена — %s", summary)
