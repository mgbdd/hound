# 1) текстовое сообщение
# 2) голосовое сообщение
# 3) изображение
# 4) текстовый файл (txt)
# 5) документ (pdf, docs)
# 6) аудио
# 7) презентация

from processor_server.typifier import detect_extension
from processor_server.analyzers.txt_file_analyzer import txt_file_analyzer
from processor_server.analyzers.audio_file_analyzer import audio_file_analyzer
from processor_server.analyzers.image_file_analyzer import image_file_analyzer
from processor_server.analyzers.docx_file_analyzer import docx_file_analyzer
from processor_server.analyzers.pdf_file_analyzer import pdf_file_analyzer
from processor_server.analyzers.pptx_file_analyzer import pptx_file_analyzer
from processor_server.temp_file import load_temp_files, delete_temp_files


def file_analyzer(file_ref, raw_message_type):
    path = load_temp_files(file_ref)
    if not path:
        return "Файл не загружен", "unknown"

    match raw_message_type:
        case "document":
            file_type = detect_extension(path)

            match file_type:
                case ".txt" | ".md":
                    message_type = "текстовый файл"
                    description = txt_file_analyzer(path)

                case ".doc" | ".docx":
                    message_type = "документ"
                    description = docx_file_analyzer(path, img_limit=3)
                case ".pdf":
                    message_type = "документ"
                    description = pdf_file_analyzer(path, img_limit=3)
                case ".ppt" | ".pptx":
                    message_type = "презентация"
                    description = pptx_file_analyzer(path, img_limit=3, audio_limit=3)
                # case ".zip"| ".rar":
                #     message_type = "архив"
                #     description = ""
                # case ".csv":
                #     message_type = "табличный файл"
                #     description = ""

                case _:
                    message_type = "unknown"
                    description = ""

        case "voice" | "audio":
            if raw_message_type == "voice":
                message_type = "голосовое сообщение"
            else:
                message_type = "аудио"

            description = audio_file_analyzer(path)
            if description is None or description == "":
                message_type = "unknown"

        case "photo":
            message_type = "изображение"
            description = image_file_analyzer(path)
            if description is None or description == "":
                message_type = "unknown"

        # case "animation":
        #     TODO: добавить обработку анимаций (gif), сейчас обработка только первого кадра
        #     message_type = "изображение"
        #     description = image_file_analyzer(path)
        #     if description is None or description == "":
        #         message_type = "unknown"

        # case "video":
        # case "video_note":
        case _:
            message_type = "unknown"
            description = ""

    delete_temp_files(path)
    return [description, message_type]
