FROM python:3.11-slim

# Install nginx and system deps
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn[standard] \
    polars \
    pandas \
    numpy \
    xgboost \
    pydantic

# Copy application code and data
COPY Data/ ./Data/

# Copy nginx and entrypoint configs
COPY nginx.conf /etc/nginx/nginx.conf
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

EXPOSE 9091

ENTRYPOINT ["/docker-entrypoint.sh"]
