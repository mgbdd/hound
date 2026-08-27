# {
#   "chat_id": <ID чата>,
#   "messages": {
#     "<message_id_1>": {
#       "message_type": "<text | photo | document | ...>",
#       "text": "<текст сообщения или caption (если есть)>",
#       "file_urls": [ "<url_1>", "<url_2>", ... ]
#     },
#     "<message_id_2>": { ... },
#     ...
#   }
# }

import json
class Parser:
    def from_data_messages(self,data):
        obj = json.loads(data.decode("utf-8"))

        chat_id = obj.get("chat_id")

        messages= {}

        for msg_id, msg in obj.get("messages",{}).items():
            messages[msg_id]={
                "message_type": msg.get("message_type"),
                "text": msg.get("text"),
                "file_urls": msg.get("file_urls",[]),
            }

        return {"chat_id": chat_id,"messages": messages}


