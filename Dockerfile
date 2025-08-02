# Use Python 3.11 slim image as base for lightweight container
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies that might be needed for Python packages
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first for better Docker layer caching
# This allows pip install to be cached when only source code changes
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project directory to the container
COPY . .

# Expose port 7860 for Gradio interface
EXPOSE 7860

# Keep container running for development purposes
# This allows developers to use docker exec to run commands inside the container
CMD ["tail", "-f", "/dev/null"]