# Use official lightweight Python image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# Set working directory
WORKDIR /app

# Install dependencies (We use the CPU version of PyTorch to keep the image small and deployment cheap)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy the core model and config
COPY multimodal_model.pth .
COPY multimodal_config.pkl .

# Copy the web application
COPY webapp/ /app/webapp/

# Set working directory to webapp where app.py lives
WORKDIR /app/webapp

# Expose port
EXPOSE 8080

# Run the application with Gunicorn for production
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", "--timeout", "0", "app:app"]
