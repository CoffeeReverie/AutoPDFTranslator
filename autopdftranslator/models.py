from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Optional


APP_NAME = "AutoPDFTranslator"
APP_VERSION = "0.3.0"


class TranslationCancelled(Exception):
    """Raised when the current translation run is cancelled by the user."""

DEFAULT_FONT = "STSong-Light"
DEFAULT_FONT_SIZE = 8.6
DEFAULT_MAX_BOX_WIDTH = 275
DEFAULT_LEFT = 18
DEFAULT_TOP = 18
DEFAULT_GAP_Y = 7
DEFAULT_PAD_X = 5
DEFAULT_PAD_Y = 4
DEFAULT_LINE_FACTOR = 1.24


@dataclasses.dataclass(slots=True)
class CommentItem:
    page_index: int
    text_original: str
    text_translated: str = ""
    source: str = "unknown"  # pdf_text | vision | merged | manual
    confidence: float = 0.0
    bbox: Optional[tuple[float, float, float, float]] = None
    metadata: dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclasses.dataclass(slots=True)
class PageExtractionResult:
    page_index: int
    comments: list[CommentItem]
    text_chars: int
    used_vision: bool
    notes: list[str] = dataclasses.field(default_factory=list)


@dataclasses.dataclass(slots=True)
class LayoutConfig:
    font_name: str = DEFAULT_FONT
    font_size: float = DEFAULT_FONT_SIZE
    max_box_width: float = DEFAULT_MAX_BOX_WIDTH
    left: float = DEFAULT_LEFT
    top: float = DEFAULT_TOP
    gap_y: float = DEFAULT_GAP_Y
    pad_x: float = DEFAULT_PAD_X
    pad_y: float = DEFAULT_PAD_Y
    line_factor: float = DEFAULT_LINE_FACTOR
    placement: str = "top-left-stacked"  # top-left-stacked | top-right-stacked


@dataclasses.dataclass(slots=True)
class PipelineConfig:
    source_lang: str = "en"
    target_lang: str = "zh-CN"
    vision_mode: str = "auto"  # never | auto | always
    typed_text_threshold: int = 120
    provider: str = "mock"
    dpi_for_vision: int = 220
    dump_json: Optional[Path] = None
    dry_run: bool = False
    max_pages: Optional[int] = None
    verbose: bool = False
    progress_callback: Any = None
