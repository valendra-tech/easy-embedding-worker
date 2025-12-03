from typing import Dict, List, Optional

import torch


class BaseEmbeddingModel:
    def encode(self, items: List[Dict[str, Optional[str]]]) -> List[List[float]]:
        raise NotImplementedError

    @staticmethod
    def _normalize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(tensor, p=2, dim=-1)
