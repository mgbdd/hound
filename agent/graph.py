"""Точка входа для langgraph API (см. langgraph.json -> agent.graph:graph).

Сервируется НАСТОЯЩИЙ граф BaseAgent — без обёртки DataManager / Parser / run().
Провайдер выбирается по AI_PROVIDER при импорте модуля (на старте langgraph-сервера),
поэтому первый импорт тянет модели RAG, Langfuse и клиент LLM — сервер стартует не мгновенно.
"""

import logging
from functools import lru_cache
from importlib import import_module

from config import AI_PROVIDER

log = logging.getLogger("hound.agent")

_PROVIDERS = {
    "gemini": ("agent.agents.gemini_agent", "GeminiAIAgent"),
    "gigachat": ("agent.agents.gigachat_agent", "GigaChatAIAgent"),
    "mistral": ("agent.agents.mistral_agent", "MistralAIAgent"),
    "yandex": ("agent.agents.yandex_agent", "YandexAIAgent"),
}


@lru_cache(maxsize=1)
def build_agent():
    """BaseAgent выбранного провайдера (синглтон). Нужен eval/скриптам ради .search() и .rag."""
    key = AI_PROVIDER if AI_PROVIDER in _PROVIDERS else "yandex"
    if AI_PROVIDER not in _PROVIDERS:
        log.info(f"AI_PROVIDER={AI_PROVIDER!r} не поддерживается — использую yandex")
    mod_name, cls_name = _PROVIDERS[key]
    log.info(f"graph.py: строю агента ({key}) — это грузит модели RAG и клиент LLM...")
    agent = getattr(import_module(mod_name), cls_name)()
    log.info("graph.py: агент готов")
    return agent


agent = build_agent()
graph = agent.agent  # CompiledStateGraph — его сервирует langgraph API
