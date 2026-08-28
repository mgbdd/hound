import logging

from processor_server.file_parsers.json_parser import Parser
from processor_server.qdrant import QdrantManager
from processor_server.message import messages_pipeline

log = logging.getLogger("hound.processor")
parser = Parser()


def upload_data(qm: QdrantManager, json_bytes):
    raw_data = parser.from_data_messages(json_bytes)
    chat_id = raw_data["chat_id"]
    raw_messages = raw_data["messages"]

    log.info("Обработка чата %s: %d сообщений", chat_id, len(raw_messages))
    qm.create_collection(chat_id)
    messages_chunks = messages_pipeline(raw_messages)
    log.info("Заливаю %d чанков в Qdrant (чат %s)", len(messages_chunks), chat_id)
    qm.add_documents(chat_id, messages_chunks)
