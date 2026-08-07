# ============================================================
# AINID — AI Network Intrusion Detector
# Dockerfile for the Streamlit UI (app.py)
# ============================================================
FROM python:3.12-slim

# Системные пакеты, нужные seaborn/matplotlib и pandas для сборки колёс
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Сначала только requirements.txt — чтобы Docker кэшировал слой с зависимостями
# и не переустанавливал их при каждом изменении кода
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Теперь весь остальной код проекта
COPY . .

# Директории, в которые пайплайн пишет во время работы
# (на случай, если их нет в образе — том всё равно их перекроет через volumes)
RUN mkdir -p data/raw data/processed models logs reports predictions

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "src/app.py", \
    "--server.address=0.0.0.0", \
    "--server.port=8501", \
    "--server.headless=true"]