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


def _build_items_openai(openai_input: Dict) -> List[Dict[str, Optional[str]]]:
    """
    Accept OpenAI-compatible payloads where `input` can be a string or list.
    """
    raw_input = openai_input.get("input")

    if isinstance(raw_input, list):
        items: List[Dict[str, Optional[str]]] = []
        for entry in raw_input:
            if isinstance(entry, str):
                items.append({"input": entry, "image_url": None})
            elif isinstance(entry, dict):
                items.append(
                    {
                        "input": entry.get("input") or entry.get("text") or entry.get("content"),
                        "image_url": entry.get("image_url"),
                    }
                )
            else:
                raise ValueError("Each item in 'input' list must be a string or dict.")
        return items

    if isinstance(raw_input, str):
        return [{"input": raw_input, "image_url": openai_input.get("image_url")}]

    raise ValueError("'input' must be a string or list for OpenAI-compatible route.")


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
        # Token accounting is model-specific; we return zeros to stay OpenAI-schema-compatible.
        "usage": {
            "prompt_tokens": 0,
            "total_tokens": 0,
        },
    }


def _list_models_response() -> Dict:
    # factory is preloaded with DEFAULT_MODEL_ID; include any cached models.
    ids = list(factory.cache.keys()) or [DEFAULT_MODEL_ID]
    # Deduplicate while preserving order
    seen = set()
    unique_ids = []
    for mid in ids:
        if mid not in seen:
            seen.add(mid)
            unique_ids.append(mid)

    return {
        "object": "list",
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "runpod",
                "permission": [],
            }
            for model_id in unique_ids
        ],
    }


def _handle_openai_route(route: str, openai_input: Dict) -> Dict:
    if route == "/v1/models":
        return _list_models_response()

    if route == "/v1/embeddings":
        model_id = openai_input.get("model") or DEFAULT_MODEL_ID
        items = _build_items_openai(openai_input)
        model = factory.get(model_id)
        embeddings = model.encode(items)
        return _format_response(model_id, embeddings)

    raise ValueError(f"Unsupported openai_route '{route}'.")


def handler(event):
    payload = event.get("input", {}) if isinstance(event, dict) else {}

    try:
        # RunPod OpenAI-compatible wrapper injects these keys when calling /openai/v1/*
        openai_route = payload.get("openai_route")
        openai_input = payload.get("openai_input") or {}
        if openai_route:
            return _handle_openai_route(openai_route, openai_input)

        model_id = payload.get("model") or DEFAULT_MODEL_ID
        items = _build_items(payload)
        model = factory.get(model_id)
        embeddings = model.encode(items)
        return _format_response(model_id, embeddings)
    except Exception as exc:  # pragma: no cover - return friendly message
        return {
            "error": str(exc),
            "model": model_id,
        }
