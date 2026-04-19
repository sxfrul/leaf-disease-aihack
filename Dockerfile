# Use a lightweight Python base image
FROM python:3.10-slim

# Force Python output to log immediately
ENV PYTHONUNBUFFERED True

# Set working directory inside the container
WORKDIR /app

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files (app.py, model.keras, etc.) to the container
COPY . .

# Expose the standard Cloud Run port
EXPOSE 8080

# Run the API with 1 worker to save memory
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1"]