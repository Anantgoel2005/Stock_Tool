FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download NLTK VADER lexicon (lightweight sentiment, always useful)
RUN python - << 'EOF'
import nltk
print("Pre-downloading VADER lexicon...")
nltk.download('vader_lexicon', quiet=True)
print("VADER lexicon ready.")
EOF

# Pre-download FinBERT weights at build time (only if not disabled)
# Set DISABLE_FINBERT=1 in Render env vars to skip this heavy step
RUN python - << 'EOF'
import os
if os.getenv("DISABLE_FINBERT", "0") != "1" and os.getenv("SENTIMENT_METHOD", "auto").lower() != "vader":
    from stock_tool.config import FINBERT_MODEL_NAME
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    print("Pre-downloading FinBERT model...", FINBERT_MODEL_NAME)
    AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
    AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
else:
    print("Skipping FinBERT download (disabled or VADER selected)")
EOF

ENV PORT=8000

CMD ["uvicorn", "webapp.main:app", "--host", "0.0.0.0", "--port", "8000"]


