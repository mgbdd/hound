FROM python:3.11-slim

WORKDIR /app

COPY requirements.docker.txt .
COPY langgraph.json .

RUN pip install --no-cache-dir -r requirements.docker.txt && pip install --no-cache-dir "langgraph-cli[inmem]"

COPY agent/ agent
COPY RAG RAG

CMD ["langgraph", "dev", "--host", "0.0.0.0", "--port", "8001", "--no-reload", "--no-browser"]