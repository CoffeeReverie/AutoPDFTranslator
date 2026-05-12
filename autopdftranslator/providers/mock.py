from __future__ import annotations

from typing import Any

from ..extractors import normalize_comment_text
from ..models import CommentItem


class MockProvider:
    """
    Safe default provider for local development.
    """

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        text = normalize_comment_text(text)
        if not text:
            return ""
        if target_lang.lower().startswith("zh"):
            return f"[待接入翻译] {text}"
        return text

    def translate_text_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        return [self.translate_text(text, source_lang=source_lang, target_lang=target_lang) for text in texts]

    def vision_extract_and_translate(
        self,
        image: Any,
        page_index: int,
        source_lang: str,
        target_lang: str,
    ) -> list[CommentItem]:
        return []
