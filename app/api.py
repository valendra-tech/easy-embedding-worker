from typing import Dict, List, Optional

from app.config import DEFAULT_MODEL_ID
from app.models.factory import factory


def _build_items(payload: Dict) -> List[Dict[str, Optional[str]]]:
    if payload.get("bulk"):
        items = payload.get("items") or payload.get("inputs")
        if not isinstance(items, list):
            raise ValueError("When 'bulk' is true, provide a list under 'items' or 'inputs'.")
        return items

    return [
        {
            "input": payload.get("input"),
            "image_url": payload.get("image_url"),
        }
    ]


def _format_response(model_id: str, embeddings: List[List[float]]):
    return {
        "object": "list",
        "model": model_id,
        "data": [
            {
                "object": "embedding",
                "index": idx,
                "embedding": emb,
                "model": model_id,
            }
            for idx, emb in enumerate(embeddings)
        ],
    }


def handler(event):
    payload = event.get("input", {}) if isinstance(event, dict) else {}
    model_id = payload.get("model") or DEFAULT_MODEL_ID

    try:
        items = _build_items(payload)
        model = factory.get(model_id)
        embeddings = model.encode(items)
        return _format_response(model_id, embeddings)
    except Exception as exc:  # pragma: no cover - return friendly message
        return {
            "error": str(exc),
            "model": model_id,
        }
