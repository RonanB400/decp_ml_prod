FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create a non-root user
RUN adduser --disabled-password --gecos '' api-user
USER api-user

# Define where the models should be mounted
ENV MODELS_DIR=/app/models

# Expose API port
EXPOSE 8000

# Start the API service
CMD ["uvicorn", "api.fast:app", "--host", "0.0.0.0", "--port", "8000"]
