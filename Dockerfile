FROM python:3.11.15-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt', quiet=True); nltk.download('punkt_tab', quiet=True)"

# Copy project files
COPY . .

# Create directories
RUN mkdir -p data/processed data/uploads

# HuggingFace runs as user 1000
RUN useradd -m -u 1000 user && chown -R user:user /app
USER user

# Environment
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false
ENV LANGCHAIN_TRACING_V2=false

EXPOSE 7860 8000

CMD ["python", "app.py"]