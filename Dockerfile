FROM python:3.11-slim

WORKDIR /app

# Force cache invalidation - update this date to rebuild: 2026-01-14-21:22
ARG CACHEBUST=1

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app after pip install to avoid cache issues
COPY app.py .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
