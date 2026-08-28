"""Подпись к изображению (BLIP) + перевод на русский. Модель грузится лениво."""

import logging
from functools import lru_cache

from PIL import Image

from processor_server.translator import translate_to_russian

logger = logging.getLogger(__name__)

_MODEL_ID = "Salesforce/blip-image-captioning-base"


@lru_cache(maxsize=1)
def _get_model():
    from transformers import BlipProcessor, BlipForConditionalGeneration

    logger.info("Загружаю BLIP (%s)...", _MODEL_ID)
    proc = BlipProcessor.from_pretrained(_MODEL_ID, use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained(_MODEL_ID)
    return proc, model


def image_file_analyzer(file_path):
    try:
        processor, model = _get_model()
        image = Image.open(file_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption_en = processor.decode(out[0], skip_special_tokens=True)
        return translate_to_russian(caption_en)
    except Exception as e:
        logger.warning("Ошибка при анализе изображения %s: %s", file_path, e)
        return ""
