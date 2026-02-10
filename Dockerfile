# Python 3.9 base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Create directory for Gensim data to persist (optional but good practice)
RUN mkdir -p /root/gensim-data

# Expose the port (Hugging Face Spaces uses 7860 by default)
EXPOSE 7860

# Command to run the application
# We use port 7860 for Hugging Face Spaces compatibility
CMD ["python", "-m", "uvicorn", "backend.server:app", "--host", "0.0.0.0", "--port", "7860"]
