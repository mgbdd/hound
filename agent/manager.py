from agent.parser import Parser
import json
import hashlib
from importlib import import_module
from config import AI_PROVIDER

class DataManager:
    def __init__(self):
        print("Запускаемся")
        self.parser = Parser()
        print(AI_PROVIDER)
        # Ленивый импорт одного агента — иначе при старте подтягиваются все провайдеры LangChain и растёт RAM.
        match AI_PROVIDER:
            case "gemini":
                mod = import_module("agent.agents.gemini_agent")
                self.agent = mod.GeminiAIAgent()
            case "gigachat":
                mod = import_module("agent.agents.gigachat_agent")
                self.agent = mod.GigaChatAIAgent()
            case "mistral":
                mod = import_module("agent.agents.mistral_agent")
                self.agent = mod.MistralAIAgent()
            case "yandex":
                mod = import_module("agent.agents.yandex_agent")
                self.agent = mod.YandexAIAgent()
            case _:
                print(f"Провайдер {AI_PROVIDER} не поддерживается. Будет использован yandex по дефолту")
                mod = import_module("agent.agents.yandex_agent")
                self.agent = mod.YandexAIAgent()

    def handle_search_payload(self, payload: dict) -> dict:
        try:
            data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        except Exception as e:
            print(f"Ошибка сериализации payload: {e}")
            return {"message_ids": [], "answer_text": ""}

        payload_sha = hashlib.sha1(data).hexdigest()[:10] if data else "empty"
        print(f"[TRACE payload_sha={payload_sha}] raw_bytes={len(data)}")
        print(data)
        try:
            request = self.parser.parse(data)
        except Exception as e:
            print(f"[TRACE payload_sha={payload_sha}] Ошибка парсинга запроса: {e}")
            request = None

        print(f"[TRACE payload_sha={payload_sha}] Запрос: {request}")
        answer_payload = {"message_ids": [], "answer_text": ""}
        try:
            if request:
                answer_payload = self.agent.run(request)
        except Exception as e:
            print(f"[TRACE payload_sha={payload_sha}] Ошибка при выполнении agent.run: {e}")
            answer_payload = {"message_ids": [], "answer_text": ""}

        print(f"[TRACE payload_sha={payload_sha}] Ответ: {answer_payload}")
        return answer_payload