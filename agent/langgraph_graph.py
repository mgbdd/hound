from typing import Any, TypedDict
import threading

from langgraph.graph import END, StateGraph

from agent.manager import DataManager

_manager: DataManager | None = None
_manager_lock = threading.Lock()


def _get_manager() -> DataManager:
    global _manager
    if _manager is None:
        with _manager_lock:
            if _manager is None:
                _manager = DataManager()
    return _manager


class SearchState(TypedDict, total=False):
    payload: dict[str, Any]
    result: dict[str, Any]


def run_search(state: SearchState) -> SearchState:
    payload = state.get("payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    manager = _get_manager()
    return {"result": manager.handle_search_payload(payload)}


graph_builder = StateGraph(SearchState)
graph_builder.add_node("run_search", run_search)
graph_builder.set_entry_point("run_search")
graph_builder.add_edge("run_search", END)
graph = graph_builder.compile()
