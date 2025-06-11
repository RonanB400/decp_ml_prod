# Use Python 3.10.6 as base image
FROM python:3.10.6-buster

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
COPY api/ ./api/
COPY decp_amount/ ./decp_amount/
COPY decp_rag/ ./decp_rag/
COPY models/ ./models/
COPY scripts/ ./scripts/

COPY main.py .
COPY config.py .
COPY config.yaml .


# Default environment variables
ENV PORT=8000

# Expose the port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
