import os
import torch

DEFAULT_MODEL_ID = os.getenv("MODEL_ID", "openai/clip-vit-large-patch14")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
