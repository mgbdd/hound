# keys
from dotenv import load_dotenv
import os

# embeddings
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings

# qdrant
from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from qdrant_client.http.models import PayloadSchemaType

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')

class QdrantManager:
    def __init__(self):
        # api
        load_dotenv(dotenv_path=DOTENV_PATH)
        self.url = os.getenv("QDRANT_URL")
        self.api_key = os.getenv("QDRANT_API_KEY")

        # model
        self.dense_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")

        # configs
        self.dense_size = 384
        self.dense_distance = Distance.COSINE

        self.client = None

    def connection_to_db(self):
        if self.client is None:
            self.client = QdrantClient(url=self.url, api_key=self.api_key)
        return self.client

    def create_collection(self, collection_name):
        client = self.connection_to_db()

        if not client.collection_exists(collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config={
                    "all-MiniLM-L6-v2": VectorParams(size=self.dense_size, distance=self.dense_distance)
                },
                sparse_vectors_config={
                    "bm25": SparseVectorParams(index=SparseIndexParams())
                }

            )
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.type",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
            self._ensure_message_id_integer_index(collection_name)
        else:
            self._ensure_payload_indexes(collection_name)

    def _ensure_message_id_integer_index(self, collection_name):
        """
        В payload message_id приходит как число — нужен INTEGER-индекс.
        Если ранее создан KEYWORD-индекс (старый деплой), удаляем и создаём INTEGER.
        """
        client = self.connection_to_db()
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.message_id",
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
            return
        except Exception as e:
            if "already exists" in str(e).lower():
                return
        try:
            client.delete_payload_index(collection_name=collection_name, field_name="metadata.message_id", wait=True)
        except Exception:
            pass
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.message_id",
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
        except Exception:
            pass

    def _ensure_payload_indexes(self, collection_name):
        """Индексы для существующих коллекций (без пересоздания) и повторных вызовов."""
        client = self.connection_to_db()
        try:
            client.create_payload_index(
                collection_name=collection_name,
                field_name="metadata.type",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception:
            pass
        self._ensure_message_id_integer_index(collection_name)

    def get_vector_store(self, collection_name):
        client = self.connection_to_db()

        self.create_collection(collection_name)

        return QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.dense_model,
            sparse_embedding=self.sparse_model,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="all-MiniLM-L6-v2",
            sparse_vector_name="bm25"
        )

    def delete_collection(self, collection_name):
        client = self.connection_to_db()
        if client.collection_exists(collection_name):
            client.delete_collection(collection_name)

    def add_documents(self, collection_name, chunks):
        store = self.get_vector_store(collection_name)
        store.add_documents(chunks)
