# Use Python 3.10 as base image
FROM python:3.10-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code and models
COPY . .

# Make sure the models directory exists
RUN mkdir -p models

# Default environment variables
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
