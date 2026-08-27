FROM python:3.11-slim

WORKDIR /app

# 1️⃣ Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    libmagic1 \
    file \
    ffmpeg \
    libreoffice \
    tesseract-ocr \
    tesseract-ocr-rus \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

# 2️⃣ Python-зависимости
COPY requirements.docker.txt .
RUN pip install --no-cache-dir -r requirements.docker.txt

# 3️⃣ Код
COPY processor_server processor_server

# 4️⃣ Запуск
CMD ["python", "-m", "processor_server.main"]
