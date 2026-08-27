from processor_server.file_parsers.json_parser import Parser
from processor_server.qdrant import QdrantManager
from processor_server.message import *

parser = Parser()


def upload_data(qm: QdrantManager, json_bytes):
    print("1. Reading data from json...")
    raw_data = parser.from_data_messages(json_bytes)

    chat_id = raw_data["chat_id"]
    raw_messages = raw_data["messages"]

    print(f"2. Processing messages for chat_id: {chat_id}...")
    qm.create_collection(chat_id)

    print("3. Transforming messages...")
    messages_chunks = messages_pipeline(raw_messages)

    print("4. Adding documents to Qdrant...")
    qm.add_documents(chat_id, messages_chunks)
