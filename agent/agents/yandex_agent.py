from agent.base_agent import BaseAgent
from langchain_community.chat_models import ChatYandexGPT
from config import YANDEX_API_KEY, AI_MODEL, AI_PROVIDER

AVAILABLE_MODELS =[
    "aliceai-llm",
    "yandexgpt",
    "yandexgpt-lite",
    "qwen3-235b-a22b-fp8"
]

class YandexAIAgent(BaseAgent):
    def __init__(self):
        current_model = AI_MODEL
        if current_model not in AVAILABLE_MODELS:
            raise ValueError(
                f"Модели {current_model} нет в списке доступных моделей провайдера {AI_PROVIDER}"
            )

        llm = ChatYandexGPT(
            api_key=YANDEX_API_KEY,
            model_name=current_model, 
            temperature=0.5, 
            max_retries=2
        )
        super().__init__(llm)