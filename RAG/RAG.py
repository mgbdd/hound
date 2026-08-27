from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import os
import json
import re
import time
import random
import itertools
from qdrant_client import QdrantClient, models
from qdrant_client.models import Filter, FieldCondition, MatchAny, MatchValue
from qdrant_client.http.models import PayloadSchemaType
from langchain_mistralai import ChatMistralAI
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_gigachat.chat_models import GigaChat
from langchain_community.chat_models import ChatYandexGPT
from langfuse import Langfuse
import inspect


def _resolve_mistral_openrouter_slug(ai_model: str) -> str:
    """Дублирует логику agent/openrouter_mistral_models.py — RAG не должен зависеть от пакета agent в Docker."""
    m = (ai_model or "").strip()
    if not m:
        raise ValueError("AI_MODEL пустой")
    if "/" in m:
        return m
    key = m.lower().replace("_", "-")
    aliases: dict[str, str] = {
        "mistral-small-latest": "mistralai/mistral-small-3.2-24b-instruct",
        "mistral-medium-latest": "mistralai/mistral-medium-3.1",
        "mistral-large-latest": "mistralai/mistral-large-2512",
    }
    return aliases.get(key, f"mistralai/{m}")


try:
    from config import (
        QDRANT_URL,
        QDRANT_API_KEY,
        AI_MODEL,
        AI_PROVIDER,
        MISTRAL_API_KEY,
        YANDEX_API_KEY,
        GEMINI_API_KEY,
        GEMINI_BASE_URL,
        OPENROUTER_API_KEY,
        OPENROUTER_BASE_URL,
        USE_OPENROUTER,
        GIGACHAT_API_KEY,
        GIGACHAT_SCOPE,
    )
except ModuleNotFoundError:
    # In docker containers `config.py` may be absent from PYTHONPATH.
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    AI_MODEL = os.getenv("AI_MODEL")
    AI_PROVIDER = os.getenv("AI_PROVIDER")
    MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")
    YANDEX_API_KEY = os.getenv("YANDEX_API_KEY")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL")
    GIGACHAT_API_KEY = os.getenv("GIGACHAT_API_KEY")
    GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE")
    _use_or = os.getenv("USE_OPENROUTER")
    if _use_or is None:
        USE_OPENROUTER = bool(OPENROUTER_API_KEY)
    else:
        USE_OPENROUTER = _use_or.strip().lower() in ("1", "true", "yes", "on")
    USE_OPENROUTER = USE_OPENROUTER and bool(OPENROUTER_API_KEY)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOTENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=DOTENV_PATH)

class RAG:
    def __init__(self):
        self.dense_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.sparse_model = FastEmbedSparse(model_name="Qdrant/bm25")
        self.qdrant_client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY
        )
        self.langfuse = Langfuse()
        if not self.langfuse.auth_check():
            print("Ошибка аутентификации Langfuse в RAG.")
        self.model = self._init_model()

    @staticmethod
    def _coerce_to_str(value) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            parts = [RAG._coerce_to_str(item) for item in value]
            return "\n".join([p for p in parts if p]).strip()
        if isinstance(value, dict):
            if isinstance(value.get("text"), str):
                return value["text"]
            for key in ("content", "message", "value"):
                if key in value:
                    nested = RAG._coerce_to_str(value.get(key))
                    if nested:
                        return nested
            return ""
        if hasattr(value, "content"):
            return RAG._coerce_to_str(getattr(value, "content"))
        return str(value)

    @staticmethod
    def _normalize_query_text(value: str) -> str:
        """
        Normalize model output item into a usable search query.
        Handles formats like:
        - '1. запрос'
        - '"запрос",'
        - bullet list items
        """
        text = (value or "").strip()
        if not text:
            return ""

        text = re.sub(r'^\s*(?:[-*]\s+|\d+[\).\s]+)', "", text).strip()

        text = text.strip(" \t\r\n\"'`,[]")
        return text.strip()

    def _parse_generated_queries(self, raw_output: str) -> list[str]:
        """
        Parse query expansion output from different LLM styles:
        - JSON list: ["q1", "q2"]
        - newline list
        - numbered lines
        """
        text = (raw_output or "").strip()
        if not text:
            return []

        queries: list[str] = []

        # 1) Try strict JSON list first.
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                queries = [self._normalize_query_text(str(item)) for item in parsed]
        except Exception:
            pass

        # 2) Fallback: line-based parsing.
        if not queries:
            queries = [self._normalize_query_text(line) for line in text.splitlines()]

        # Filter empty/noisy entries and deduplicate preserving order.
        cleaned: list[str] = []
        seen: set[str] = set()
        for q in queries:
            # Skip obvious noise tokens from malformed list rendering.
            if not q or q in {"[", "]", ",", "\"", "'"}:
                continue
            if q not in seen:
                seen.add(q)
                cleaned.append(q)
        return cleaned

    def _init_model(self):
        match AI_PROVIDER:
            case "mistral":
                if USE_OPENROUTER:
                    or_model = _resolve_mistral_openrouter_slug(AI_MODEL)
                    model = ChatOpenAI(
                        api_key=OPENROUTER_API_KEY,
                        base_url=OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1",
                        model=or_model,
                        temperature=0,
                        max_retries=2,
                        timeout=None,
                    )
                else:
                    model = ChatMistralAI(
                        model=AI_MODEL,
                        temperature=0,
                        max_retries=2,
                        api_key=MISTRAL_API_KEY,
                    )
            case "yandex":
                model = ChatYandexGPT(
                    api_key=YANDEX_API_KEY,
                    model_name=AI_MODEL, 
                    temperature=0, 
                    max_retries=2
                )
            case "gemini":
                if USE_OPENROUTER:
                    openrouter_model = AI_MODEL if "/" in AI_MODEL else f"google/{AI_MODEL}"
                    model = ChatOpenAI(
                        api_key=OPENROUTER_API_KEY,
                        base_url=OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1",
                        model=openrouter_model,
                        temperature=0,
                        max_retries=2,
                        timeout=None,
                    )
                else:
                    model = ChatGoogleGenerativeAI(
                        google_api_key=GEMINI_API_KEY,
                        base_url=GEMINI_BASE_URL or None,
                        model=AI_MODEL,
                        temperature=0,  # Gemini 3.0+ defaults to 1.0
                        max_tokens=None,
                        timeout=None,
                        max_retries=2,
                    )
            case "gigachat":
                model = GigaChat(
                    credentials=GIGACHAT_API_KEY,
                    scope=GIGACHAT_SCOPE,
                    verify_ssl_certs=False,
                    model=AI_MODEL
                )
            case _:
                raise ValueError(f"Модель {AI_MODEL} не поддерживается провайдером {AI_PROVIDER}")
        return model

    def _build_message_id_exclusion(self, collection, exclude_ids):
        """must_not-условие по metadata.message_id для исключения отклонённых сообщений."""
        ids = [str(x).strip() for x in (exclude_ids or []) if str(x).strip()]
        if not ids:
            return []
        try:
            if all(x.isdigit() for x in ids):
                self._ensure_message_id_integer_index(collection)
                return [FieldCondition(key="metadata.message_id",
                                       match=MatchAny(any=[int(x) for x in ids]))]
            self._ensure_message_id_keyword_index(collection)
            return [FieldCondition(key="metadata.message_id", match=MatchAny(any=ids))]
        except Exception as e:
            print(f"_build_message_id_exclusion: {e}")
            return []

    def retriever(self, request, chat_id, message_type, results, exclude_ids=None, attempt=0):
        collection = str(chat_id)
        try:
            if not self.qdrant_client.collection_exists(collection):
                print(f"retriever: коллекции {collection!r} нет — пропуск запроса")
                return results
        except Exception as e:
            print(f"retriever: проверка collection_exists({collection!r}) не удалась: {e}")
            return results

        exclude_ids = [str(x).strip() for x in (exclude_ids or []) if str(x).strip()]
        excluded_set = set(exclude_ids)
        attempt = int(attempt or 0)
        # с повторных попыток расширяем выдачу
        per_branch_limit = 6 if attempt >= 1 else 3
        final_limit = 10 if attempt >= 1 else 5

        dense_query_vector = self.dense_model.encode(request).tolist()
        sparse_query_vector = None
        try:
            sparse_embedding = self.sparse_model.embed_query(request)
            sparse_query_vector = models.SparseVector(
                indices=list(sparse_embedding.indices),
                values=list(sparse_embedding.values)
            )
        except Exception:
            pass

        must = []
        if message_type and message_type != "None":
            must.append(FieldCondition(key="metadata.type", match=MatchValue(value=message_type)))
        must_not = self._build_message_id_exclusion(collection, exclude_ids)
        flt = Filter(must=must or None, must_not=must_not or None) if (must or must_not) else None

        prefetch = [
            models.Prefetch(
                query=dense_query_vector,
                using="all-MiniLM-L6-v2",
                limit=per_branch_limit,
                filter=flt,
            ),
        ]
        if sparse_query_vector is not None:
            prefetch.append(
                models.Prefetch(
                    query=sparse_query_vector,
                    using="bm25",
                    limit=per_branch_limit,
                    filter=flt,
                )
            )

        answer = self.qdrant_client.query_points(
            collection_name=collection,
            prefetch=prefetch,
            query=models.FusionQuery(fusion=models.Fusion.RRF),
            limit=final_limit,
            with_payload=True
        )
        for point in answer.points:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            message_id = meta.get("message_id")
            if message_id is None:
                continue
            if str(message_id).strip() in excluded_set:  # страховка, если фильтр не сработал
                continue
            results.append({message_id: payload.get("page_content")})
        return results

    @staticmethod
    def _normalize_message_id_key(mid) -> str:
        if mid is None:
            return ""
        return str(mid).strip()

    @staticmethod
    def merge_text_chunks_by_start_index(pairs: list[tuple[int, str]]) -> str:
        if not pairs:
            return ""
        pairs = sorted(pairs, key=lambda x: x[0])
        start0, t0 = pairs[0]
        result = t0
        for i in range(1, len(pairs)):
            s, t = pairs[i]
            prev_s, prev_t = pairs[i - 1]
            prev_end = prev_s + len(prev_t)
            overlap = prev_end - s
            if overlap >= len(t):
                continue
            if overlap > 0:
                result += t[overlap:]
            else:
                result += t
        return result

    @staticmethod
    def _merge_two_boundary_overlap(a: str, b: str, max_overlap: int = 250) -> str:
        max_ol = min(len(a), len(b), max_overlap)
        for ol in range(max_ol, 0, -1):
            if a[-ol:] == b[:ol]:
                return a + b[ol:]
        return a + b

    @classmethod
    def merge_chunk_texts_best_effort(cls, parts: list[str]) -> str:
        """Склейка чанков без start_index: перебор порядка (малый n) или жадная цепочка."""
        cleaned = [p for p in parts if isinstance(p, str) and p.strip()]
        if not cleaned:
            return ""
        if len(cleaned) == 1:
            return cleaned[0]
        if len(cleaned) <= 8:
            best: str | None = None
            best_len = None
            for perm in itertools.permutations(cleaned):
                m = perm[0]
                for nxt in perm[1:]:
                    m = cls._merge_two_boundary_overlap(m, nxt)
                if best_len is None or len(m) < best_len:
                    best_len = len(m)
                    best = m
            return best or ""
        result = cleaned[0]
        remaining = cleaned[1:]
        while remaining:
            best_i, best_overlap = -1, 0
            for i, r in enumerate(remaining):
                max_ol = min(len(result), len(r), 250)
                for ol in range(max_ol, 0, -1):
                    if result[-ol:] == r[:ol] and ol > best_overlap:
                        best_overlap, best_i = ol, i
                        break
            if best_i >= 0 and best_overlap > 0:
                r = remaining.pop(best_i)
                result = result + r[best_overlap:]
            else:
                result += remaining.pop(0)
        return result

    def _ensure_message_id_integer_index(self, collection: str) -> None:
        """Для фильтра MatchAny по int нужен INTEGER-индекс."""
        client = self.qdrant_client
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name="metadata.message_id",
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
            return
        except Exception as e:
            if "already exists" in str(e).lower():
                return
        try:
            client.delete_payload_index(collection_name=collection, field_name="metadata.message_id", wait=True)
        except Exception:
            pass
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name="metadata.message_id",
                field_schema=PayloadSchemaType.INTEGER,
                wait=True,
            )
        except Exception as e2:
            print(f"_ensure_message_id_integer_index: {e2}")

    def _ensure_message_id_keyword_index(self, collection: str) -> None:
        """Для фильтра MatchAny по str нужен KEYWORD-индекс."""
        client = self.qdrant_client
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name="metadata.message_id",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
            return
        except Exception as e:
            if "already exists" in str(e).lower():
                return
        try:
            client.delete_payload_index(collection_name=collection, field_name="metadata.message_id", wait=True)
        except Exception:
            pass
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name="metadata.message_id",
                field_schema=PayloadSchemaType.KEYWORD,
                wait=True,
            )
        except Exception as e2:
            print(f"_ensure_message_id_keyword_index: {e2}")

    def fetch_merged_texts_for_message_ids(
        self,
        chat_id,
        message_ids: list,
    ) -> dict[str, str]:
        """
        Догружает из Qdrant все точки с данными message_id и склеивает чанки в полный текст.
        Ключи — нормализованные строковые id.
        """
        out: dict[str, str] = {}
        ids = [self._normalize_message_id_key(m) for m in (message_ids or [])]
        ids = [x for x in ids if x]
        if not ids:
            return out

        collection = str(int(chat_id)) if str(chat_id).strip().isdigit() else str(chat_id)
        try:
            if not self.qdrant_client.collection_exists(collection):
                return out
        except Exception:
            return out

        # Нельзя смешивать MatchAny(str) и MatchAny(int) в одном фильтре — Qdrant требует оба индекса
        # на одном поле и отвечает 400. Для числовых id только int + INTEGER; иначе str + KEYWORD.
        if all(x.isdigit() for x in ids):
            self._ensure_message_id_integer_index(collection)
            flt = Filter(
                must=[
                    FieldCondition(
                        key="metadata.message_id",
                        match=MatchAny(any=[int(x) for x in ids]),
                    )
                ]
            )
        else:
            self._ensure_message_id_keyword_index(collection)
            flt = Filter(
                must=[
                    FieldCondition(
                        key="metadata.message_id",
                        match=MatchAny(any=ids),
                    )
                ]
            )

        buckets: dict[str, list[tuple[int, str]]] = {}
        next_offset = None
        try:
            while True:
                records, next_offset = self.qdrant_client.scroll(
                    collection_name=collection,
                    scroll_filter=flt,
                    limit=256,
                    offset=next_offset,
                    with_payload=True,
                    with_vectors=False,
                )
                for rec in records:
                    payload = rec.payload or {}
                    meta = payload.get("metadata") or {}
                    mid = meta.get("message_id")
                    page = payload.get("page_content") or ""
                    if mid is None:
                        continue
                    key = self._normalize_message_id_key(mid)
                    if not key:
                        continue
                    si = meta.get("start_index")
                    idx = int(si) if isinstance(si, (int, float)) else -1
                    if key not in buckets:
                        buckets[key] = []
                    buckets[key].append((idx, page))
                if next_offset is None:
                    break
        except Exception as e:
            print(f"fetch_merged_texts_for_message_ids scroll error: {e}")
            return out

        want = set(ids)
        for k in want:
            pairs = buckets.get(k)
            if not pairs:
                continue
            if all(idx >= 0 for idx, _ in pairs):
                merged = self.merge_text_chunks_by_start_index(pairs)
            else:
                texts = [t for _, t in pairs]
                merged = self.merge_chunk_texts_best_effort(texts)
            out[k] = merged.strip()

        return out

    _RETRYABLE_MARKERS = ("524", "timeout", "timed out", "502", "503", "504", "429",
                          "resource_exhausted", "too many requests")

    def _invoke_model_with_retry(self, compiled_prompt, *, max_attempts: int = 3):
        for attempt in range(1, max_attempts + 1):
            try:
                return self.model.invoke(compiled_prompt)
            except Exception as e:
                retryable = any(m in str(e).lower() for m in self._RETRYABLE_MARKERS)
                print(f"Ошибка LLM (generate_rag_queries, попытка {attempt}/{max_attempts}): {e}")
                if (not retryable) or attempt == max_attempts:
                    raise
                time.sleep((2 ** (attempt - 1)) + random.uniform(0, 0.3))

    def generate_rag_queries(self, request):
        original_query = str(list(request.values())[0]).strip()
        prompt_name = inspect.currentframe().f_code.co_name.lstrip("_")
        prompt = self.langfuse.get_prompt(prompt_name)
        compiled_prompt = prompt.compile(query=original_query)
        try:
            queries = self._invoke_model_with_retry(compiled_prompt)
            queries_text = self._coerce_to_str(queries)
            paraphrases = self._parse_generated_queries(queries_text)
        except Exception as e:
            # Расширение запроса не критично — при сбое ищем по одному оригинальному запросу.
            print(f"generate_rag_queries: расширение запроса пропущено ({e})")
            paraphrases = []
        if original_query:
            return [original_query, *paraphrases]
        return paraphrases

    def search(self, request, message_type="текстовое сообщение", exclude_ids=None, attempt=0):
        attempt = int(attempt or 0)
        exclude_ids = [str(x) for x in (exclude_ids or [])]
        print(f"Type of message: {message_type}, attempt: {attempt}, exclude: {len(exclude_ids)}")
        queries = self.generate_rag_queries(request)
        raw_chat_id = list(request.keys())[0]
        try:
            chat_id = int(raw_chat_id)
        except (TypeError, ValueError):
            chat_id = raw_chat_id
        # со 2-й повторной попытки снимаем ограничение по типу сообщения
        effective_type = "None" if attempt >= 2 else message_type
        results = []
        for q in queries:
            results = self.retriever(
                q, chat_id, effective_type, results,
                exclude_ids=exclude_ids, attempt=attempt,
            )
        return results