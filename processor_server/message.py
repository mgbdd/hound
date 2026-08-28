import logging

from processor_server.analyzers.file_analyzer import file_analyzer
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger("hound.processor")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=250,
    chunk_overlap=100,
    add_start_index=True,
)


def message_text_extractor(msg_id, raw_message):
    text = raw_message.get("text")
    if not text or not str(text).strip():
        # Нет текста/подписи -> не создаём пустой "текстовое сообщение" документ.
        return None

    return Document(
        page_content=str(text),
        metadata={
            "message_id": msg_id,
            "type": "текстовое сообщение",
        }
    )


def message_file_extractor(msg_id, raw_message):
    document_files = []

    if raw_message.get("message_type") != "text":
        for file_ref in raw_message.get("file_paths") or raw_message.get("file_urls") or []:
            description, message_type = file_analyzer(file_ref, raw_message["message_type"])

            if message_type == "unknown":
                continue

            document_files.append(
                Document(
                    page_content=description,
                    metadata={
                        "message_id": msg_id,
                        "type": message_type,
                    }
                )
            )

    return document_files


def message_transformer(msg_id, raw_message):
    document_text = message_text_extractor(msg_id, raw_message)

    document_files = message_file_extractor(msg_id, raw_message)

    docs = document_files if document_text is None else [document_text, *document_files]
    return docs


def messages_pipeline(raw_messages):
    documents = []
    for msg_id, raw_message in raw_messages.items():
        log.info("Сообщение id=%s тип=%s", msg_id, raw_message.get("message_type"))
        documents.extend(message_transformer(msg_id, raw_message))

    return text_splitter.split_documents(documents)
