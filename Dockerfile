FROM python:3.11-slim

# CatBoost needs OpenMP at runtime
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8050
EXPOSE 8050

# Match the Procfile / Render start command so this image can serve the
# dashboard on any container host (Fly, Cloud Run, etc.).
CMD ["sh", "-c", "gunicorn dashboard.app:server --bind 0.0.0.0:${PORT}"]
