import io
from typing import Dict, List, Optional

import requests
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from app.config import DEVICE, DTYPE
from app.models.base import BaseEmbeddingModel


class ClipEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        self.model.eval()

    def _load_image(self, url: str) -> Image.Image:
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        return Image.open(io.BytesIO(response.content)).convert("RGB")

    def _encode_single(self, text: Optional[str], image_url: Optional[str]) -> List[float]:
        if not text and not image_url:
            raise ValueError("Provide 'input' text, 'image_url', or both.")

        image = None
        if image_url:
            image = self._load_image(image_url)

        if image is not None:
            inputs = self.processor(
                text=[text or ""],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=[text or ""],
                return_tensors="pt",
                padding=True,
            )

        prepared: Dict[str, torch.Tensor] = {}
        for key, value in inputs.items():
            if value.dtype in (torch.float16, torch.float32):
                prepared[key] = value.to(device=DEVICE, dtype=DTYPE)
            else:
                prepared[key] = value.to(device=DEVICE)

        with torch.no_grad():
            text_features = None
            image_features = None

            if text is not None:
                text_features = self.model.get_text_features(**prepared)
            if image is not None:
                image_features = self.model.get_image_features(**prepared)

            if text_features is not None and image_features is not None:
                combined = (text_features + image_features) / 2
            elif text_features is not None:
                combined = text_features
            else:
                combined = image_features

            combined = self._normalize(combined)
            return combined.cpu().tolist()[0]

    def encode(self, items: List[Dict[str, Optional[str]]]) -> List[List[float]]:
        return [self._encode_single(item.get("input"), item.get("image_url")) for item in items]
