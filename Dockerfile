FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYSPARK_PYTHON=python3
ENV PYSPARK_DRIVER_PYTHON=python3

# Install Java + utilities
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk-headless curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$JAVA_HOME/bin:$PATH

WORKDIR /app

COPY requirements.txt .

# Remove Windows-only dependency
RUN sed -i '/pywin32/d' requirements.txt

# Upgrade pip first
RUN pip install --upgrade pip

# Install dependencies with larger timeout
RUN pip install \
    --default-timeout=1000 \
    --no-cache-dir \
    -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]