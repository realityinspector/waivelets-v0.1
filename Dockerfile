FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY web/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY fastprint.py .
COPY basis.npz .
COPY basis_clusters.json .
COPY MIDWAY_REPORT.md .
COPY web/ web/

# Predownload the model at build time so startup is fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 8080

CMD ["uvicorn", "web.server:app", "--host", "0.0.0.0", "--port", "8080"]
