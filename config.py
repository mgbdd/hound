import os
from dotenv import load_dotenv
load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# agent
AI_PROVIDER=os.getenv("AI_PROVIDER")
AI_MODEL=os.getenv("AI_MODEL")


# LangFuse
LANGFUSE_SECRET_KEY=os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY=os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_BASE_URL=os.getenv("LANGFUSE_BASE_URL")

#       GIGACHAT API keys
GIGACHAT_API_KEY=os.getenv("GIGACHAT_API_KEY")
GIGACHAT_CLIENT_ID=os.getenv("GIGACHAT_CLIENT_ID")
GIGACHAT_SCOPE=os.getenv("GIGACHAT_SCOPE")

#       YANDEX API keys
YC_FOLDER_ID=os.getenv("YC_FOLDER_ID")
YANDEX_API_KEY=os.getenv("YANDEX_API_KEY")

#       MISTRAL API keys
MISTRAL_API_KEY=os.getenv("MISTRAL_API_KEY")
MISTRAL_MODEL=os.getenv("MISTRAL_MODEL")

#       GEMINI API keys
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
GEMINI_BASE_URL=os.getenv("GEMINI_BASE_URL")
OPENROUTER_API_KEY=os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL=os.getenv("OPENROUTER_BASE_URL")

# Явный переключатель маршрутизации всех провайдеров через OpenRouter.
# По умолчанию включён, если задан OPENROUTER_API_KEY (обратная совместимость);
# принудительно выключить нативным SDK провайдера: USE_OPENROUTER=false
USE_OPENROUTER=_env_bool("USE_OPENROUTER", default=bool(OPENROUTER_API_KEY)) and bool(OPENROUTER_API_KEY)

#       EXTRA API KEYS
HF_TOKEN=os.getenv("HF_TOKEN")
TAVILY_API_KEY=os.getenv("TAVILY_API_KEY")


#       QDRANT API
QDRANT_URL=os.getenv("QDRANT_URL")
QDRANT_API_KEY=os.getenv("QDRANT_API_KEY")



