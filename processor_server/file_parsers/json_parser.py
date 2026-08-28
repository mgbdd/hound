# {
#   "chat_id": <ID чата>,
#   "messages": {
#     "<message_id_1>": {
#       "message_type": "<text | photo | document | ...>",
#       "text": "<текст сообщения или caption (если есть)>",
#       "file_paths": [ "<относительный путь Telegram>", ... ]
#     },
#     ...
#   }
# }

import json


class Parser:
    def from_data_messages(self, data):
        obj = json.loads(data.decode("utf-8"))
        chat_id = obj.get("chat_id")

        messages = {}
        for msg_id, msg in obj.get("messages", {}).items():
            # file_paths — новый ключ; file_urls — легаси (может остаться в старом chat_data.json)
            paths = msg.get("file_paths")
            if paths is None:
                paths = msg.get("file_urls") or []
            messages[msg_id] = {
                "message_type": msg.get("message_type"),
                "text": msg.get("text"),
                "file_paths": paths,
            }

        return {"chat_id": chat_id, "messages": messages}
