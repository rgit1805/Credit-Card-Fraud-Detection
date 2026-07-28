FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies needed for libraries like SHAP / LightGBM
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy project source code
COPY . .

# Set environment configurations
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

# Expose API and Streamlit ports
EXPOSE 8000
EXPOSE 8501
