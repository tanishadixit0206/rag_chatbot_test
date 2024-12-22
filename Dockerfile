FROM python:3.11-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from transformers import BartForConditionalGeneration, BartTokenizer; BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn'); BartTokenizer.from_pretrained('facebook/bart-large-cnn')" \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
ENV PYTHONWARNINGS="ignore"
ENTRYPOINT [ "python", "main.py" ]
