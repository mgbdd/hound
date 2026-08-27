import json


class Parser:
    """Нормализует входной payload от tg_bot в {chat_id: {...}}.

    Поддерживает форматы:
      - {chat_id: {"request": "запрос"}}                       (обычный /search)
      - {chat_id: {"request": "...", "repeat": true,
                   "exclude_ids": [...], "attempt": 2}}         (повторный поиск)
      - {chat_id: {"<uuid>": "запрос"}}                         (легаси)
    Всегда возвращает {chat_id: {"query", "exclude_ids", "attempt", "repeat"}}.
    """

    def parse(self, data):
        if not data:
            return None

        try:
            payload = json.loads(data.decode("utf-8"))
        except Exception:
            # Malformed JSON or non-utf8 payload -> treat as empty request.
            return None

        for chat_id, sub in (payload or {}).items():
            if not isinstance(sub, dict):
                continue

            query = sub.get("request") or sub.get("query")
            if query is None:
                # легаси: {chat_id: {"<uuid>": "запрос"}}
                for value in sub.values():
                    if isinstance(value, str):
                        query = value
                        break
            if query is None:
                continue

            return {
                str(chat_id): {
                    "query": str(query),
                    "exclude_ids": [str(x) for x in (sub.get("exclude_ids") or [])],
                    "attempt": int(sub.get("attempt") or 0),
                    "repeat": bool(sub.get("repeat", False)),
                }
            }

        return None
