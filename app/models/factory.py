from typing import Dict

from transformers import AutoConfig

from app.config import DEFAULT_MODEL_ID
from app.models.base import BaseEmbeddingModel
from app.models.clip_model import ClipEmbeddingModel
from app.models.text_model import TextEmbeddingModel


class ModelFactory:
    def __init__(self, preload_model_id: str | None = None):
        self.cache: Dict[str, BaseEmbeddingModel] = {}
        if preload_model_id:
            self.get(preload_model_id)

    def get(self, model_id: str) -> BaseEmbeddingModel:
        if model_id in self.cache:
            return self.cache[model_id]

        config = AutoConfig.from_pretrained(model_id)
        if getattr(config, "model_type", "") == "clip":
            model = ClipEmbeddingModel(model_id)
        else:
            model = TextEmbeddingModel(model_id)

        self.cache[model_id] = model
        return model


factory = ModelFactory(preload_model_id=DEFAULT_MODEL_ID)
