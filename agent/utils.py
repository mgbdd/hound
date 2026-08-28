from typing import List, TypedDict, Optional
import logging
import os
from pathlib import Path

log = logging.getLogger("hound.agent")


def resolve_mistral_openrouter_slug(ai_model: str) -> str:
    """Имя модели Mistral -> slug OpenRouter.

    Native Mistral API принимает mistral-*-latest; OpenRouter требует конкретный id
    (mistralai/mistral-large-2512 и т.п.), иначе 400 invalid model ID. Если карточка
    на OpenRouter сменилась — обновите словарь или задайте AI_MODEL с полным «/».
    """
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


def coerce_to_str(value) -> str:
    """Ответ llm.invoke() -> строка.

    LangChain AIMessage.content бывает str ЛИБО list[{"type":"text","text":...}]
    (Gemini/Anthropic, часть провайдеров в LangChain 1.x). Приводим к строке любой
    из вариантов; None -> "".
    """
    if value is None:
        return ""
    content = getattr(value, "content", value)  # разворачиваем BaseMessage
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return "\n".join(p for p in parts if p).strip()
    if isinstance(content, dict):
        return str(content.get("text") or content.get("content") or "").strip()
    return str(content).strip()


def get_filepath(filename, current_dir=Path(__file__).parent.resolve()):
    for root, dirs, files in os.walk(current_dir):
        if filename in files:
            return str(Path(root) / filename)
    return None

def get_prompt(filename):
    try:
        filepath = get_filepath(filename)
        with open(filepath, "r", encoding="utf-8") as f:
            prompt = f.read()
            return prompt
    except FileNotFoundError:
        log.warning("Файл-промпт %r не найден", filename)
        return ""
    except Exception as e:
        log.warning("Ошибка при чтении файла %r: %s", filename, e)
        return ""



class AgentState(TypedDict):
    user_query: str
    chat_id: str
    message_type : str
    current_search_results : Optional[List[str]]
    raw_search_results: Optional[List[dict]]
    # "text" -> генерируем красивый ответ, "messages" -> отдаём список message_id.
    # Решается LLM (промпт route_output) один раз на входе графа.
    output_format: Optional[str]
    # message_id, отклонённые пользователем в предыдущих попытках поиска по этому запросу —
    # исключаются из RAG (Qdrant must_not) и из результатов rerank.
    excluded_message_ids: Optional[List[str]]
    # Итог графа (нода finalize) — то, что читает клиент вместо ручного разбора состояния.
    message_ids: Optional[List[str]]
    answer_text: Optional[str]
    # id сообщений, по которым собран текст для pretty_answer (для eval / трассировки; tg_bot при наличии answer_text их не форвардит)
    cited_message_ids: Optional[List[str]]
    repeat_rag: bool  # флаг для повторного RAG поиска (true - пользователь дал фидбек, что данный ему ответ не подходит)
    change_type: bool # флаг для потворного RAG поиска с новмы типом сообщений (можно повторить поиск с тем же типом, но вернуть другие сообщения)
    attempt_count : int