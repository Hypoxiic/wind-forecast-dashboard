# Dockerfile  ←‑ this name is important; leave out any extension
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "src/pipeline.py"]
