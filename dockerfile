# Use a lightweight Python environment
FROM python:3.8-slim

# Install system dependencies (for compiling packages like shap)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory in container
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit default port
EXPOSE 8501

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
