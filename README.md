# Easy Embedding Worker

Easy Embedding Worker provides OpenAI-compatible embeddings for text and images, powered by HuggingFace models and designed to run on RunPod serverless infrastructure. It supports bulk embedding requests and optional image URL processing.

## Features
- OpenAI-compatible API for embeddings
- Supports both text and image inputs
- Bulk mode for efficient batch processing
- Powered by HuggingFace models
- GPU acceleration (NVIDIA, 8GB+ recommended)
- Configurable via environment variables
- Preset models for quick setup

## Quick Start

1. **Clone the repository:**
   ```bash
   git clone https://github.com/valendra/easy-embedding-worker.git
   cd easy-embedding-worker
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the worker:**
   ```bash
   python handler.py
   ```

## Configuration

Environment variables can be set to customize the worker:
- `MODEL_ID`: Default HuggingFace model ID (e.g., `openai/clip-vit-large-patch14`)
- `HF_TOKEN`: HuggingFace token for private models
- `TRANSFORMERS_CACHE`: Directory for model cache
- `CUDA_VISIBLE_DEVICES`: Restrict visible GPUs

See `.runpod/hub.json` for more details and presets.

## API

The worker exposes endpoints compatible with OpenAI's embedding API. You can send requests with text or image URLs and receive embeddings in response.

## Presets
- Default CLIP: `openai/clip-vit-large-patch14`
- BGE Small Text: `BAAI/bge-small-en-v1.5`
- Multilingual E5: `intfloat/multilingual-e5-small`

## Resources
- GPU: NVIDIA, 8GB+ memory recommended
- Volume: `/cache` (20GB)

## License
MIT

## Author
valendra

## Repository
[GitHub](https://github.com/valendra/easy-embedding-worker)
