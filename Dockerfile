FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CUDA-enabled PyTorch first so the correct wheels are pulled.
COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt --break-system-packages

COPY . .

ENV MODEL_ID="openai/clip-vit-large-patch14"
ENV PYTHONUNBUFFERED=1

# Start the RunPod serverless handler directly (the `runpod` package does not
# expose a __main__ entrypoint).
CMD ["python3", "-u", "handler.py"]
