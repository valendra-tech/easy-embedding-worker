from typing import Dict, List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from app.config import DEVICE, DTYPE
from app.models.base import BaseEmbeddingModel


class TextEmbeddingModel(BaseEmbeddingModel):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=DTYPE).to(DEVICE)
        self.model.eval()

    @staticmethod
    def _mean_pool(model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
        return (token_embeddings * input_mask_expanded).sum(1) / input_mask_expanded.sum(1).clamp(min=1e-9)

    def _encode_single(self, text: Optional[str]) -> List[float]:
        if text is None:
            raise ValueError("'input' text is required for non-CLIP models.")

        encoded_input = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded_input = {k: v.to(DEVICE) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embedding = self._mean_pool(model_output, encoded_input["attention_mask"])
            sentence_embedding = self._normalize(sentence_embedding)
            return sentence_embedding.cpu().tolist()[0]

    def encode(self, items: List[Dict[str, Optional[str]]]) -> List[List[float]]:
        return [self._encode_single(item.get("input")) for item in items]
