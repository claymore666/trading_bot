FROM python:3.10-slim

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python code
COPY . /app/

# Make the scripts executable
RUN chmod +x /app/scalping_system_runner.py
RUN chmod +x /app/build_and_deploy.sh

# Set Python path
ENV PYTHONPATH=/app

# Create data and logs directories
RUN mkdir -p /app/data/historical /app/data/exports /app/logs

# Expose the API port
EXPOSE 8000

# Default command (can be overridden in docker-compose.yml)
CMD ["uvicorn", "scalping_engine.main:app", "--host", "0.0.0.0", "--port", "8000"]
