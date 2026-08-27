import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    TOKEN = os.getenv("TG_BOT_TOKEN", "")
    PROCESSING_SERVER_URL = os.getenv("PROCESSING_SERVER_URL", "http://processor_server:3030/api")
    RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "http://agent:8001")
    RAG_GRAPH_ID = os.getenv("RAG_GRAPH_ID", "hound-search")


if not Config.TOKEN:
    raise RuntimeError("TG_BOT_TOKEN is not set — see .env.example")
