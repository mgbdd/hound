from agent.base_agent import BaseAgent
from agent.utils import resolve_mistral_openrouter_slug
from langchain_mistralai import ChatMistralAI
from langchain_openai import ChatOpenAI
from config import (
    MISTRAL_API_KEY,
    AI_MODEL,
    AI_PROVIDER,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    USE_OPENROUTER,
)

AVAILABLE_MODELS = [
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
]


class MistralAIAgent(BaseAgent):
    def __init__(self):
        current_model = AI_MODEL
        if USE_OPENROUTER:
            openrouter_model = resolve_mistral_openrouter_slug(current_model)
            llm = ChatOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1",
                model=openrouter_model,
                temperature=0.5,
                max_retries=2,
                timeout=None,
            )
            print(f"Mistral через OpenRouter: model={openrouter_model!r}")
        else:
            if current_model not in AVAILABLE_MODELS:
                print(
                    f"Модели {current_model} нет в списке доступных моделей провайдера {AI_PROVIDER}"
                )
                raise ValueError(current_model)
            llm = ChatMistralAI(
                model=current_model,
                temperature=0.5,
                max_retries=2,
                api_key=MISTRAL_API_KEY,
            )
        super().__init__(llm)