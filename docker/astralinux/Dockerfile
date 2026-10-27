# ---------- Этап 1: Сборка зависимостей ----------
FROM registry.astralinux.ru/library/astra/ubi18-python311:latest AS builder

# Устанавливаем только то, что нужно для сборки, без recommends
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    python3-venv \
    python3-pip \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Создаём venv и ставим Python-зависимости
COPY requirements.txt .
RUN python3 -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir --upgrade pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# ---------- Этап 2: Минимальный рантайм ----------
FROM registry.astralinux.ru/library/astra/ubi18-python311:latest

# Ставим только рантайм-зависимости, без recommends
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-rus \
    libgl1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Копируем готовое виртуальное окружение из билдера
COPY --from=builder /opt/venv /opt/venv

# Создаём пользователя
RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY --chown=appuser:appuser *.py .

# Делаем владельцем appuser
RUN chown -R appuser:appuser /app

USER appuser
ENV PATH="/opt/venv/bin:$PATH"
CMD ["python", "main.py"]