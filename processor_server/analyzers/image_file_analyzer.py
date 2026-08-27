from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from processor_server.translator import *

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")


def image_file_analyzer(file_path):
    try:
        image = Image.open(file_path).convert("RGB")
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption_en = processor.decode(out[0], skip_special_tokens=True)
        caption_ru = translate_to_russian(caption_en)
        # print("Изображение успешно обработано:", caption_ru)
        return caption_ru
    except Exception as e:
        print(f"Ошибка при анализе изображения {file_path}: {e}")
        return ""
