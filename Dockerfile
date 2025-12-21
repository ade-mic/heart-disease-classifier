# Use official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY heart_model.pkl .
COPY heart.csv .
COPY fastapi_app/ ./fastapi_app/

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "fastapi_app.app:app", "--host", "0.0.0.0", "--port", "8000"]
