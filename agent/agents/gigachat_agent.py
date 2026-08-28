import logging

from agent.base_agent import BaseAgent
from langchain_gigachat.chat_models import GigaChat
from config import GIGACHAT_API_KEY, GIGACHAT_SCOPE, AI_MODEL, AI_PROVIDER

log = logging.getLogger("hound.agent")

AVAILABLE_MODELS = [
    "GigaChat-2",
    "GigaChat-2-Pro",
    "GigaChat-2-Max"
]

class GigaChatAIAgent(BaseAgent):
    def __init__(self):
        current_model = AI_MODEL
        if current_model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Модели {current_model} нет в списке доступных моделей провайдера {AI_PROVIDER}"
            )
        log.info("GigaChat: model=%s", current_model)
        llm = GigaChat(
                    credentials=GIGACHAT_API_KEY,
                    scope=GIGACHAT_SCOPE,
                    verify_ssl_certs=False,
                    model=current_model,
                    temperature=1,
                )
        super().__init__(llm)