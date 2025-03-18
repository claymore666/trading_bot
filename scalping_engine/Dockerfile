FROM python:3.10-slim

# Install build essentials and Rust for PyO3 integration
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Set up working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Rust project
COPY rust-indicators /app/rust-indicators

# Build the Rust indicators library
RUN cd /app/rust-indicators && \
    cargo build --release && \
    cp target/release/librust_indicators.so /app/ || echo "Rust build skipped"

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
