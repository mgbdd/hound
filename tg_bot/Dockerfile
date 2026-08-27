FROM python:3.11-slim

WORKDIR /app

COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

COPY tg_bot tg_bot

CMD ["python", "-m", "tg_bot.main"]
