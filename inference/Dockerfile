FROM python:3.11-slim

WORKDIR /app

COPY api /app/api
COPY bert-base-uncased /app/bert-base-uncased
COPY input /app/input

ENV PYTHONPATH=/app

RUN pip install --no-cache-dir fastapi uvicorn torch transformers tokenizers

EXPOSE 8000

CMD ["ls"]

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
