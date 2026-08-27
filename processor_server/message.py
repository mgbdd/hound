from processor_server.analyzers.file_analyzer import *
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

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

    if raw_message["message_type"] != "text" and raw_message["file_urls"] is not None:
        for url in raw_message["file_urls"]:

            description, message_type = file_analyzer(url, raw_message["message_type"])

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
    doc_index = 0
    for msg_id, raw_message in raw_messages.items():
        print(f"Обработка сообщения {doc_index}, id:{msg_id} типа:{raw_message['message_type']}")
        message_union = message_transformer(msg_id, raw_message)

        documents.extend(message_union)

    chunks = text_splitter.split_documents(documents)
    return chunks
