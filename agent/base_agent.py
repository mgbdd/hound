import inspect
from typing import Literal
from langchain_core.output_parsers import JsonOutputParser
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from agent.utils import AgentState
from RAG.RAG import RAG
from langfuse import Langfuse
import re, json
import time
import random

from abc import ABC

class BaseAgent(ABC):
    def __init__(self, llm):
        self.llm = llm
        self.rag_search_tool = tool(self._rag_search)
        self.determine_message_type_tool = tool(self._determine_message_type)
        self.validation_tool = tool(self._rerank)
        self.pretty_answer_tool = tool(self._pretty_answer)
        self.rag = RAG()
        self.parser = JsonOutputParser()
        self.langfuse = Langfuse()
        if not self.langfuse.auth_check():
            print("Ошибка аутентификации Langfuse.")

        self.agent = self._init_agent()

    @staticmethod
    def _coerce_to_str(value) -> str:
        if value is None:
            return ""

        def extract_text(obj) -> str:
            if obj is None:
                return ""

            if isinstance(obj, str):
                return obj

            if isinstance(obj, list):
                parts = [extract_text(item) for item in obj]
                return "\n".join([p for p in parts if p]).strip()

            if isinstance(obj, dict):
                if isinstance(obj.get("text"), str):
                    return obj["text"]
                for key in ("content", "message", "value"):
                    if key in obj:
                        nested = extract_text(obj.get(key))
                        if nested:
                            return nested
                return ""
            if hasattr(obj, "content"):
                return extract_text(getattr(obj, "content"))

            return str(obj)

        return extract_text(value).strip()

    def clear_json_str(self, s : str) :
        match = re.search(r'```json\*(.*?)\s*```', s, re.DOTALL)
        if match:
            json_str = match.group(1).strip()
            try:
                json.loads(json_str)
            except json.JSONDecodeError:
                pass
            return json_str

        cleaned = re.sub(r'```json\s*', '', s)
        cleaned = re.sub(r'\s*```', '', cleaned)
        return cleaned.strip() 

    @staticmethod
    def _is_retryable_llm_error(error: Exception) -> bool:
        text = str(error).lower()
        retryable_markers = [
            "524",
            "timeout",
            "timed out",
            "502",
            "503",
            "504",
            "429",
            "resource_exhausted",
            "too many requests",
        ]
        return any(marker in text for marker in retryable_markers)

    def _invoke_llm_with_retry(self, compiled_prompt, *, max_attempts: int = 3, label: str = "llm"):
        """Вызов self.llm.invoke с экспоненциальным backoff на транзиентных ошибках (429/5xx/timeout).
        Пробрасывает исключение, если ретраи не помогли или ошибка не транзиентная."""
        for attempt in range(1, max_attempts + 1):
            try:
                return self.llm.invoke(compiled_prompt)
            except Exception as e:
                print(f"Ошибка LLM ({label}, попытка {attempt}/{max_attempts}): {e}")
                if (not self._is_retryable_llm_error(e)) or attempt == max_attempts:
                    raise
                delay = (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                print(f"{label}: повтор через {delay:.2f}s")
                time.sleep(delay)

    def _determine_message_type(self, query: str) -> str:
        """
        Этот инструмент принимает на вход запрос пользователя \n 
        и на основании него определяет наиболее релевантный тип данных для запрашиваемых сообщений.
        Возвращает тип сообщения, например: "текстовое сообщение", и ничего больше.
        """
        prompt_name = inspect.currentframe().f_code.co_name.lstrip("_")
        prompt = self.langfuse.get_prompt(prompt_name)

        compiled_prompt = prompt.compile(query=query)
        print(f"===     FULL PROMPT NAME:\n{prompt_name}")
        try:
            response_str = self._invoke_llm_with_retry(compiled_prompt, label="determine_message_type")
            print(f"===     RESPONSE CONTENT :\n{getattr(response_str, 'content', response_str)}")
            return self._coerce_to_str(response_str).strip() or "None"
        except Exception as e:
            print(f"Ошибка при определении типа сообщения: {e}. Возвращаю дефолт.")
            return "None"

    def _call_determine_message_type_node(self, state: AgentState) -> AgentState:
        # attempt_count задаётся в run() из payload и НЕ трогается нодами графа —
        # иначе ломается эскалация поиска по номеру попытки.
        try:
            query = state["user_query"]
            response = self.determine_message_type_tool.invoke({"query": query})
            print(f"_call_determine_message_type_node: {response}")
            return {"message_type": response}
        except Exception as e:
            print(f"Ошибка при вызове _determine_message_type: {e}")
            return {"message_type": "текстовое сообщение"}

    def _rag_search(self, query: str, chat_id: int, message_type: str,
                    exclude_ids: list | None = None, attempt: int = 0):
        """
        Этот инструмент осуществляет RAG поиск наибоилее релевантных \n
        сообщений, отвечающих на запрос пользователя.
        Принимает запрос, id чата, тип сообщения, список исключаемых message_id\n
        (отклонённых в прошлых попытках) и номер попытки.
        Возвращает список id наиболее релевантных сообщений, ничего больше.
        """

        print(f"_rag_search query: {query} (attempt={attempt}, exclude={len(exclude_ids or [])})")
        messages = self.rag.search(
            {str(chat_id): query}, message_type, exclude_ids=exclude_ids or [], attempt=attempt
        )
        def deduplicate_by_id(data):
            seen = set()
            result = []

            for item in data:
                for key, value in item.items():
                    if key not in seen:
                        seen.add(key)
                        result.append(item)
                    break 
            return result
        return {
            "deduplicated": deduplicate_by_id(messages),
            "all_chunks": messages,
        }
    
    def _call_rag_search_node(self, state: AgentState) -> AgentState:
        message_type = state["message_type"]
        exclude_ids = self._dedupe_ids_preserve_order(state.get("excluded_message_ids") or [])
        attempt = int(state.get("attempt_count") or 0)
        try:
            response = self.rag_search_tool.invoke({
                "query": state["user_query"],
                "chat_id": state["chat_id"],
                "message_type": message_type,
                "exclude_ids": exclude_ids,
                "attempt": attempt,
            })
            deduplicated = response.get("deduplicated", []) if isinstance(response, dict) else []
            all_chunks = response.get("all_chunks", deduplicated) if isinstance(response, dict) else deduplicated
            print(f"RAG SEARCH RESPONSE: {deduplicated}")
            return {
                "messages": [AIMessage(content=f"Выполнен RAG поиск типа '{message_type}'.")],
                "current_search_results": deduplicated,
                "raw_search_results": all_chunks,
            }
        except Exception as e:
            print(f"Ошибка при вызове rag_search: {e}")
            return {
                "messages": [AIMessage(content=f"Выполнен RAG поиск типа '{message_type}'.")],
                "current_search_results": [],
                "raw_search_results": [],
            }

    def _rerank(self, query, results):
        """
        Этот инструмент осуществляет реранк полученных результатов поиска в соотвествиии с запросом пользователяю\n
        Принимает на вход запрос и список релевантных сообщений. \n
        Возвращает список id наиболее релевантных сообщений, ничего больше. 
        """

        prompt_name = inspect.currentframe().f_code.co_name.lstrip("_")
        prompt = self.langfuse.get_prompt(prompt_name)
        compiled_prompt = prompt.compile(query=query, messages=results)
    
        print(f"===     FULL PROMPT NAME:\n{prompt_name}")
        max_attempts = 3
        last_error: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = self.llm.invoke(compiled_prompt)
                print(f"RAW RERANK RESPONSE: {raw_response}")
                response_text = self._coerce_to_str(raw_response)
                cleaned_response = self.clear_json_str(response_text)
                print(f"PARSED RERANK RESPONSE: {cleaned_response}")
                break
            except Exception as e:
                last_error = e
                is_retryable = self._is_retryable_llm_error(e)
                print(f"Ошибка при вызове rerank (attempt {attempt}/{max_attempts}): {e}")
                if (not is_retryable) or attempt == max_attempts:
                    return []

                delay = (2 ** (attempt - 1)) + random.uniform(0, 0.3)
                print(f"RERANK retry after {delay:.2f}s")
                time.sleep(delay)

        try:
            try:
                parsed_response = self.parser.parse(cleaned_response)
            except Exception:
                return []
            messages_id = list(parsed_response.keys())
            print(f"RERANK MESSAGE IDS: {messages_id}")
        except Exception as e:
            print(f"Ошибка при вызове rerank: {e}")
            if last_error:
                print(f"RERANK last retry error: {last_error}")
            return []      

        return messages_id

    @staticmethod
    def _dedupe_ids_preserve_order(ids) -> list:
        seen: set[str] = set()
        out: list = []
        for x in ids or []:
            s = str(x).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    def _merge_raw_chunks_for_message(self, raw_results: list, mid_key: str) -> str:
        chunks: list[str] = []
        for item in raw_results:
            if not isinstance(item, dict):
                continue
            for m, t in item.items():
                if str(m).strip() == mid_key and isinstance(t, str):
                    chunks.append(t)
        return self.rag.merge_chunk_texts_best_effort(chunks)

    def _call_rerank_node(self, state: AgentState) -> AgentState:
        try:
            query = state["user_query"]
            current_results = state["current_search_results"]
            excluded = set(self._dedupe_ids_preserve_order(state.get("excluded_message_ids") or []))
            rerank_results = self.validation_tool.invoke({"query": query, "results": current_results})
            rerank_results = [
                x for x in self._dedupe_ids_preserve_order(rerank_results)
                if str(x).strip() not in excluded
            ]
            print("RERUNK RESULTS: ", rerank_results)
            return {
                "messages": [AIMessage(content="Выполнен rerank")],
                "current_search_results": rerank_results,
                "raw_search_results": state.get("raw_search_results", current_results),
            }
        except Exception as e:
            print(f"Ошибка при вызове _call_validation_node: {e}")
            return {
                "messages": [AIMessage(content="Попытка выполнить rerank")],
                "current_search_results": [],
                "raw_search_results": state.get("raw_search_results", []),
            }

    def _pretty_answer(self, query, results):
        """
        Этот инструмент принимает запрос пользователя и результаты по этому запросу
        и формирует красивый текстовый ответ, 
        если в запросе не просят явно вернуть лишь список сообщений.
        """
        prompt_name = inspect.currentframe().f_code.co_name.lstrip("_")
        prompt = self.langfuse.get_prompt(prompt_name)
        compiled_prompt = prompt.compile(query = query, messages=results)
        print(f"===     FULL PROMPT NAME:\n{prompt_name}")

        pretty_answer = ""
        try:
            pretty_answer = self._invoke_llm_with_retry(compiled_prompt, label="pretty_answer")
            print(f"_pretty_answer: {pretty_answer}")
        except Exception as e:
            print(f"Ошибка при вызове _pretty_answer: {e}")
            return ""
        return self._coerce_to_str(pretty_answer)

    def _call_pretty_answer_node(self, state: AgentState) -> AgentState:
        try:
            current_results = state["current_search_results"]
            raw_results = state.get("raw_search_results", []) or []
            chat_id = state["chat_id"]
            ordered_ids = self._dedupe_ids_preserve_order(current_results)

            full_by_id = self.rag.fetch_merged_texts_for_message_ids(chat_id, ordered_ids)
            pretty_input: list = []
            for mid in ordered_ids:
                k = str(mid).strip()
                text = full_by_id.get(k)
                if not text and k.isdigit():
                    text = full_by_id.get(str(int(k)))
                if not text:
                    text = self._merge_raw_chunks_for_message(raw_results, k)
                if text:
                    pretty_input.append({k: text})

            if not pretty_input:
                selected_messages = []
                selected_ids = {str(x).strip() for x in ordered_ids}
                for item in raw_results:
                    if not isinstance(item, dict):
                        continue
                    for mid, t in item.items():
                        if str(mid).strip() in selected_ids:
                            selected_messages.append({str(mid).strip(): t})
                pretty_input = selected_messages if selected_messages else raw_results

            pretty_answer = self.pretty_answer_tool.invoke({"query": state["user_query"], "results": pretty_input})
            pretty_answer_str = (pretty_answer or "").strip()
            cited_ids = self._dedupe_ids_preserve_order(
                [str(x).strip() for x in ordered_ids if str(x).strip().isdigit()]
            )
            return {
                "current_search_results": [pretty_answer_str],
                "cited_message_ids": cited_ids,
                "messages": [AIMessage(content="Сформирован красивый ответ")],
            }
        except Exception as e:
            print(f"Ошибка при вызове _call_pretty_answer_node: {e}")
            return {
                "messages": [AIMessage(content="Попытка сформировать красивый ответ")],
                "current_search_results": []
            }

    def _classify_output_format(self, query: str) -> str:
        """LLM по запросу решает удобный формат ответа: 'text' или 'messages'.

        Промпт route_output зависит только от запроса (не от результатов поиска),
        поэтому классификация делается один раз на входе графа, а не на пути
        rerank -> ответ.
        """
        prompt = self.langfuse.get_prompt("route_output")
        compiled_prompt = prompt.compile(query=query)
        try:
            raw_response = self._invoke_llm_with_retry(compiled_prompt, label="route_output")
        except Exception as e:
            print(f"Ошибка при вызове route_output: {e}. Возвращаю 'messages'.")
            return "messages"

        response_text = self._coerce_to_str(raw_response).strip().lower()
        print(f"===     ROUTE OUTPUT RESPONSE:\n{response_text}")
        # Промпт обязан вернуть ровно "text" либо "messages"; всё остальное -> messages.
        if "text" in response_text or "текст" in response_text:
            return "text"
        return "messages"

    def _call_route_output_node(self, state: AgentState) -> AgentState:
        try:
            fmt = self._classify_output_format(state["user_query"])
        except Exception as e:
            print(f"Ошибка при вызове _call_route_output_node: {e}")
            fmt = "messages"
        print(f"_call_route_output_node: {fmt}")
        return {"output_format": fmt}

    def _output_edge(self, state: AgentState) -> Literal["text", "messages"]:
        # Если rerank ничего не вернул — красивый ответ строить не из чего.
        if not state.get("current_search_results"):
            return "messages"
        return "text" if state.get("output_format") == "text" else "messages"

    def _has_query_edge(self, state: AgentState):
        # Пустой запрос -> сразу к finalize, без прогона классификаторов и RAG.
        if (state.get("user_query") or "").strip():
            return ["determine_message_type", "route_output_node"]
        return "finalize"

    def _prepare_node(self, state: AgentState) -> AgentState:
        """Нормализует вход клиента (user_query / chat_id / exclude / attempt / repeat)
        в полное состояние графа. Раньше это делал BaseAgent.run()."""
        query = str(state.get("user_query", "") or "")
        return {
            "messages": [HumanMessage(content=query)],
            "user_query": query,
            "chat_id": str(state.get("chat_id", "") or ""),
            "message_type": "",
            "current_search_results": [],
            "raw_search_results": [],
            "cited_message_ids": [],
            "output_format": "messages",
            "excluded_message_ids": self._dedupe_ids_preserve_order(state.get("excluded_message_ids") or []),
            "repeat_rag": bool(state.get("repeat_rag", False)),
            "change_type": False,
            "attempt_count": int(state.get("attempt_count") or 0),
            "message_ids": [],
            "answer_text": "",
        }

    def _finalize_node(self, state: AgentState) -> AgentState:
        """Единый выход графа: {message_ids, answer_text}. Раньше — постобработка в run()."""
        results = state.get("current_search_results") or []
        cited = self._dedupe_ids_preserve_order(
            [str(x).strip() for x in (state.get("cited_message_ids") or []) if str(x).strip().isdigit()]
        )
        if len(results) == 1 and isinstance(results[0], str):
            text_result = results[0].strip()
            if text_result and not text_result.isdigit():
                return {"message_ids": cited, "answer_text": text_result}
            # pretty-answer не удался -> отдаём хотя бы id после rerank, а не пустоту
            if cited:
                return {"message_ids": cited, "answer_text": ""}
        digit_ids = self._dedupe_ids_preserve_order(
            [str(x).strip() for x in results if str(x).strip().isdigit()]
        )
        return {"message_ids": digit_ids, "answer_text": ""}

    def _init_agent(self):
        gb = StateGraph(AgentState)
        gb.add_node("prepare", self._prepare_node)
        gb.add_node("determine_message_type", self._call_determine_message_type_node)
        gb.add_node("route_output_node", self._call_route_output_node)
        gb.add_node("rag_search", self._call_rag_search_node)
        gb.add_node("rerank_node", self._call_rerank_node)
        gb.add_node("pretty_answer_node", self._call_pretty_answer_node)
        gb.add_node("finalize", self._finalize_node)

        gb.add_edge(START, "prepare")
        # determine_message_type и route_output зависят только от user_query — параллельно;
        # rag_search стартует, когда завершились обе ветки. Пустой запрос -> сразу finalize.
        gb.add_conditional_edges("prepare", self._has_query_edge,
                                 ["determine_message_type", "route_output_node", "finalize"])
        gb.add_edge("determine_message_type", "rag_search")
        gb.add_edge("route_output_node", "rag_search")
        gb.add_edge("rag_search", "rerank_node")
        gb.add_conditional_edges("rerank_node", self._output_edge, {
            "text": "pretty_answer_node",
            "messages": "finalize",
        })
        gb.add_edge("pretty_answer_node", "finalize")
        gb.add_edge("finalize", END)
        # Без checkpointer: под langgraph API сервер подставляет свой слой персистентности
        # (треды / чекпоинты / HIL); для eval и скриптов достаточно одноразового invoke.
        return gb.compile()

    def search(self, query: str, chat_id, exclude_ids: list | None = None,
               attempt: int = 0, repeat: bool = False) -> dict:
        """Одноразовый прогон графа без сервера langgraph (eval, скрипты, тесты)."""
        try:
            final = self.agent.invoke({
                "user_query": str(query),
                "chat_id": str(chat_id),
                "excluded_message_ids": exclude_ids or [],
                "attempt_count": int(attempt or 0),
                "repeat_rag": bool(repeat),
            })
            return {
                "message_ids": final.get("message_ids", []) or [],
                "answer_text": final.get("answer_text", "") or "",
            }
        except Exception as e:
            print(f"Ошибка при вызове graph.invoke: {e}")
            return {"message_ids": [], "answer_text": ""}


