FROM python:3.10-slim

# Set working directory
WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --upgrade pip

# Copy pre-downloaded PyTorch and dependencies (Linux-compatible wheels)
COPY wheels /wheels
RUN pip install /wheels/*.whl

# Copy and install project requirements
COPY requirements.txt .
RUN pip install --default-timeout=1000 --retries 10 --no-cache-dir -r requirements.txt

# Copy rest of the app code
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Start the app
CMD ["streamlit", "run", "explain_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
