from __future__ import annotations

import dataclasses

from config import DEFAULT_CHAT_MODEL, DEFAULT_EMBED_MODEL


@dataclasses.dataclass
class ModelConfig:
    """Конфигурация моделей."""
    chat_model: str = DEFAULT_CHAT_MODEL
    embed_model: str = DEFAULT_EMBED_MODEL
    max_context_tokens: int = 8000  # rough conservative default
    temperature: float = 0.7
