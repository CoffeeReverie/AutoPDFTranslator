from __future__ import annotations

import abc
from typing import Any

from ..models import CommentItem
from .mock import MockProvider
from .openai_compatible import OpenAICompatibleProvider


class TranslationProvider(abc.ABC):
    @abc.abstractmethod
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        raise NotImplementedError

    def translate_text_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        return [self.translate_text(text, source_lang=source_lang, target_lang=target_lang) for text in texts]

    @abc.abstractmethod
    def vision_extract_and_translate(
        self,
        image: Any,
        page_index: int,
        source_lang: str,
        target_lang: str,
    ) -> list[CommentItem]:
        raise NotImplementedError


def build_provider(name: str) -> TranslationProvider:
    name = name.lower().strip()
    if name == "mock":
        return MockProvider()
    if name in {"openai", "openai-compatible", "compatible"}:
        return OpenAICompatibleProvider()
    raise ValueError(f"Unknown provider: {name}")


__all__ = [
    "TranslationProvider",
    "MockProvider",
    "OpenAICompatibleProvider",
    "build_provider",
]
