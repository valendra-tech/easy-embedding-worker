FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y --no-install-recommends python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install CUDA-enabled PyTorch first so the correct wheels are pulled.
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1

COPY requirements.txt ./
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .

ENV MODEL_ID="openai/clip-vit-large-patch14"
ENV PYTHONUNBUFFERED=1

CMD ["python3", "-m", "runpod"]
