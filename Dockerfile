FROM python:3.11-slim

# Update pip
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libhdf5-dev \
    pkg-config \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app


COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download images
RUN python image-downloader.py

# Generate embeddings
RUN python src/embeddings.py

# Run unit-tests
RUN pytest src/test.py

# Run the API while running a container
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
