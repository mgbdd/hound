"""
Соответствие имён моделей Mistral Chat API и slug на OpenRouter.

На native Mistral API допустимы mistral-*-latest; OpenRouter ожидает конкретные id
(например mistralai/mistral-large-2512), иначе 400 invalid model ID.

Если карточина модели на OpenRouter сменилась — обновите словарь или задайте в AI_MODEL
полный slug с «/».
"""


def resolve_mistral_openrouter_slug(ai_model: str) -> str:
    m = (ai_model or "").strip()
    if not m:
        raise ValueError("AI_MODEL пустой")
    if "/" in m:
        return m

    key = m.lower().replace("_", "-")
    aliases: dict[str, str] = {
        # актуальные slug см. https://openrouter.ai/mistralai
        "mistral-small-latest": "mistralai/mistral-small-3.2-24b-instruct",
        "mistral-medium-latest": "mistralai/mistral-medium-3.1",
        "mistral-large-latest": "mistralai/mistral-large-2512",
    }
    return aliases.get(key, f"mistralai/{m}")
