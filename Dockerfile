FROM python:3.11-slim

WORKDIR /app

# Install system deps if needed (uncomment if scrapy/psycopg2 build fails)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     build-essential curl \
#     && rm -rf /var/lib/apt/lists/*

# Copy and install Python deps first (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8080
ENV STREAMLIT_SERVER_PORT=8080

# Cloud Run uses port 8080 by default
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.address 0.0.0.0 --server.port ${PORT:-8080}"]
