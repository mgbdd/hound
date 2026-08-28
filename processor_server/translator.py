"""EN -> RU перевод (MarianMT). Модель грузится лениво при первом вызове."""

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

_MODEL_NAME = "Helsinki-NLP/opus-mt-en-ru"
_MAX_TOKENS = 512  # жёсткий предел MarianMT


@lru_cache(maxsize=1)
def _get_model():
    from transformers import MarianMTModel, MarianTokenizer

    logger.info("Загружаю модель перевода %s...", _MODEL_NAME)
    tokenizer = MarianTokenizer.from_pretrained(_MODEL_NAME)
    model = MarianMTModel.from_pretrained(_MODEL_NAME)
    return tokenizer, model


def translate_to_russian(text: str) -> str:
    if not text or not text.strip():
        return ""
    tokenizer, model = _get_model()
    inputs = tokenizer(
        text, return_tensors="pt", padding=True, truncation=True, max_length=_MAX_TOKENS
    )
    translated = model.generate(**inputs)
    return tokenizer.decode(translated[0], skip_special_tokens=True)
