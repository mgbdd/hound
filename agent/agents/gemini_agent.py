from agent.base_agent import BaseAgent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    USE_OPENROUTER,
    AI_MODEL,
    AI_PROVIDER,
)

AVAILABLE_MODELS =[
    "gemini-3.1-pro-preview",
    "gemini-3-flash-preview",
    "gemini-2.0-flash-lite",
]

class GeminiAIAgent(BaseAgent):
    def __init__(self):
        current_model = AI_MODEL
        if (not USE_OPENROUTER) and (current_model not in AVAILABLE_MODELS):
            raise ValueError(
                f"Модели {current_model} нет в списке доступных моделей провайдера {AI_PROVIDER}"
            )

        if USE_OPENROUTER:
            openrouter_model = current_model if "/" in current_model else f"google/{current_model}"
            llm = ChatOpenAI(
                api_key=OPENROUTER_API_KEY,
                base_url=OPENROUTER_BASE_URL or "https://openrouter.ai/api/v1",
                model=openrouter_model,
                temperature=1,
                max_retries=2,
                timeout=None,
            )
        else:
            llm = ChatGoogleGenerativeAI(
                google_api_key=GEMINI_API_KEY,
                base_url=GEMINI_BASE_URL or None,
                model=current_model,
                temperature=1,  
                max_tokens=None,
                timeout=None,
                max_retries=2,
            )

        super().__init__(llm)