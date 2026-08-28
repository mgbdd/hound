from typing import List, Annotated, TypedDict,Optional
from langchain_core.messages import BaseMessage
import os
from pathlib import Path

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
        print(f"Файл '{filename}' не был найден.")
        return ""
    except Exception as e:
        print("Ошибка при чтении файла:", e)
        return ""



class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y] #сообщения о работе агента
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