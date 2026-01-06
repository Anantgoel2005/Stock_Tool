FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download FinBERT weights at build time so first requests don't time out
RUN python - << "EOF" \
    && echo "Pre-downloading FinBERT model..." \
    && from stock_tool.config import FINBERT_MODEL_NAME \
    && from transformers import AutoTokenizer, AutoModelForSequenceClassification \
    && AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME) \
    && AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
EOF

ENV PORT=8000

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]


