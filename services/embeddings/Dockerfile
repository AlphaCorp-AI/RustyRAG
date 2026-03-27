FROM nvidia/cuda:12.8.0-runtime-ubuntu24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip python3-dev gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages -r requirements.txt

COPY main.py .

EXPOSE 80
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
