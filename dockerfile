# Use a lightweight Python environment
FROM --platform=linux/amd64 python:3.8-slim

# Install system dependencies (SHAP, ONNX, AIF360 + Tkinter fix)
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    python3-dev \
    libffi-dev \
    libssl-dev \
    cmake \
    tk \
    libtk8.6 \
    libnss3 \
    libgconf-2-4 \
    libxss1 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files into container
COPY . /app

# Upgrade pip & install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit app
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
