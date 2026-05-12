from __future__ import annotations

import dataclasses
import json
import logging
import math
import os
from pathlib import Path
from typing import Any, Optional

from .extractors import (
    PDFTextExtractor,
    PageRenderer,
    ensure_runtime_dependencies,
    extract_comments_with_opendataloader,
)
from .models import CommentItem, LayoutConfig, PageExtractionResult, PipelineConfig, TranslationCancelled
from .providers import TranslationProvider, build_provider
from .writer import AnnotationWriter


class HybridTranslatorPipeline:
    def __init__(self, config: PipelineConfig, provider: TranslationProvider) -> None:
        self.config = config
        self.provider = provider
        self.text_extractor = PDFTextExtractor()
        self.renderer = PageRenderer(dpi=config.dpi_for_vision)
        self.odl_fallback_enabled = os.getenv("AUTOPDFTRANSLATOR_ODL_FALLBACK", "").strip() == "1"
        self.odl_min_gain = _env_int("AUTOPDFTRANSLATOR_ODL_MIN_GAIN", 2, 1, 30)
        self._odl_page_comments: dict[int, list[CommentItem]] | None = None
        self._odl_unavailable = False

    def run(self, input_pdf: Path) -> list[PageExtractionResult]:
        import fitz

        doc = fitz.open(str(input_pdf))
        results: list[PageExtractionResult] = []
        text_items_pending_translation: list[CommentItem] = []
        try:
            page_count = doc.page_count
            if self.config.max_pages is not None:
                page_count = min(page_count, self.config.max_pages)
            self._emit_progress(stage="extracting", phase="start", pages_done=0, pages_total=page_count)

            for page_index in range(page_count):
                page = doc[page_index]
                mode = self.config.vision_mode.lower()

                if mode == "always":
                    image = self.renderer.render_page(page)
                    vision_items, vision_error = self._try_vision_extract_and_translate(image, page_index)
                    if vision_items:
                        text_items_pending_translation.extend(
                            item for item in vision_items if not (item.text_translated or "").strip()
                        )
                        result = PageExtractionResult(
                            page_index=page_index,
                            comments=vision_items,
                            text_chars=0,
                            used_vision=True,
                            notes=[
                                "vision_only_skip_pdf_text=1",
                                f"vision_items={len(vision_items)}",
                                *([f"vision_error={vision_error}"] if vision_error else []),
                            ],
                        )
                    else:
                        result = PageExtractionResult(
                            page_index=page_index,
                            comments=[],
                            text_chars=0,
                            used_vision=True,
                            notes=[
                                "vision_only_skip_pdf_text=1",
                                "vision_items=0",
                                "vision_only_no_text_fallback=1",
                                *([f"vision_error={vision_error}"] if vision_error else []),
                            ],
                        )
                else:
                    text_result = self.text_extractor.extract_page_comments(page, page_index)
                    text_result = self._maybe_apply_odl_fallback(input_pdf, page_index, text_result)
                    use_vision = self.should_use_vision(text_result, page=page)

                    if use_vision:
                        image = self.renderer.render_page(page)
                        vision_items, vision_error = self._try_vision_extract_and_translate(image, page_index)
                        if vision_items:
                            text_items_pending_translation.extend(
                                item for item in vision_items if not (item.text_translated or "").strip()
                            )
                            # Vision mode already performs semantic grouping from visual context.
                            # Avoid mixing in pdf_text items, which can cause duplicated comments.
                            result = PageExtractionResult(
                                page_index=page_index,
                                comments=vision_items,
                                text_chars=text_result.text_chars,
                                used_vision=True,
                                notes=[
                                    f"vision_items={len(vision_items)}",
                                    f"pdf_text_items_ignored={len(text_result.comments)}",
                                    *([f"vision_error={vision_error}"] if vision_error else []),
                                ],
                            )
                        else:
                            translated_items = self._prepare_text_items(text_result.comments)
                            text_items_pending_translation.extend(translated_items)
                            result = PageExtractionResult(
                                page_index=page_index,
                                comments=translated_items,
                                text_chars=text_result.text_chars,
                                used_vision=False,
                                notes=[
                                    "vision_items=0",
                                    *([f"vision_error={vision_error}"] if vision_error else []),
                                    f"pdf_text_items_fallback={len(translated_items)}",
                                ],
                            )
                    else:
                        translated_items = self._prepare_text_items(text_result.comments)
                        text_items_pending_translation.extend(translated_items)
                        result = PageExtractionResult(
                            page_index=page_index,
                            comments=translated_items,
                            text_chars=text_result.text_chars,
                            used_vision=False,
                            notes=[
                                *text_result.notes,
                                f"pdf_text_items={len(text_result.comments)}",
                            ],
                        )

                results.append(result)
                logging.info(
                    "Page %s: text_chars=%s used_vision=%s comments=%s",
                    page_index + 1,
                    result.text_chars,
                    result.used_vision,
                    len(result.comments),
                )
                self._emit_progress(
                    stage="extracting",
                    phase="page_done",
                    page_index=page_index,
                    pages_done=page_index + 1,
                    pages_total=page_count,
                    comments=sum(len(result.comments) for result in results),
                )
            self._translate_text_items_in_place(text_items_pending_translation)
        finally:
            doc.close()
        return results

    def _try_vision_extract_and_translate(self, image: Any, page_index: int) -> tuple[list[CommentItem], str]:
        try:
            return (
                self.provider.vision_extract_and_translate(
                    image=image,
                    page_index=page_index,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                ),
                "",
            )
        except Exception as exc:
            message = str(exc).replace("\n", " ").strip()
            logging.warning("Page %s vision extraction failed, continuing with fallback: %s", page_index + 1, message)
            return [], message[:500]

    def _emit_progress(self, **event: Any) -> None:
        callback = getattr(self.config, "progress_callback", None)
        if not callable(callback):
            return
        try:
            callback(event)
        except TranslationCancelled:
            raise
        except Exception:
            logging.debug("Progress callback failed", exc_info=True)

    def _maybe_apply_odl_fallback(
        self,
        input_pdf: Path,
        page_index: int,
        text_result: PageExtractionResult,
    ) -> PageExtractionResult:
        if not self.odl_fallback_enabled or self._odl_unavailable:
            return text_result
        if not self._looks_like_fragmented_page(text_result):
            return text_result

        if self._odl_page_comments is None:
            try:
                self._odl_page_comments = extract_comments_with_opendataloader(input_pdf)
                logging.info(
                    "ODL fallback loaded for %s pages",
                    len(self._odl_page_comments),
                )
            except Exception as exc:
                self._odl_unavailable = True
                logging.info("ODL fallback unavailable: %s", exc)
                return text_result

        odl_items = list((self._odl_page_comments or {}).get(page_index, []))
        if not odl_items:
            return text_result

        current_count = len(text_result.comments)
        should_replace = len(odl_items) >= current_count + self.odl_min_gain
        if not should_replace and self._has_oversized_text_item(text_result):
            should_replace = len(odl_items) >= max(2, current_count)
        if not should_replace:
            return text_result

        notes = list(text_result.notes)
        notes.append(f"odl_items={len(odl_items)}")
        notes.append(f"odl_replaced_pdf_text_items={current_count}")
        return dataclasses.replace(
            text_result,
            comments=odl_items,
            notes=notes,
        )

    def _looks_like_fragmented_page(self, text_result: PageExtractionResult) -> bool:
        fragment_count = _extract_note_int(text_result.notes, "fragments")
        comment_count = len(text_result.comments)
        if fragment_count >= 28 and comment_count <= 10:
            return True
        if fragment_count >= 48 and comment_count <= 16:
            return True
        if self._has_oversized_text_item(text_result):
            return True
        return False

    def _has_oversized_text_item(self, text_result: PageExtractionResult) -> bool:
        for item in text_result.comments:
            text = (item.text_original or "").strip()
            if len(text) >= 240:
                return True
        return False

    def _prepare_text_items(self, items: list[CommentItem]) -> list[CommentItem]:
        return [
            dataclasses.replace(
                item,
                source=item.source if item.source != "unknown" else "pdf_text",
            )
            for item in items
        ]

    def _translate_text_items_in_place(self, items: list[CommentItem]) -> None:
        if not items:
            return

        batch_size = _env_int("AUTOPDFTRANSLATOR_TEXT_BATCH_SIZE", 5000, 1, 20000)
        pages_total = len({item.page_index for item in items})
        translated_pages: set[int] = set()
        logging.info(
            "Translating %s extracted text comments in %s batch request(s)",
            len(items),
            math.ceil(len(items) / batch_size),
        )
        self._emit_progress(
            stage="translating",
            phase="start",
            pages_done=0,
            pages_total=pages_total,
            comments_total=len(items),
        )

        for offset in range(0, len(items), batch_size):
            chunk = items[offset : offset + batch_size]
            self._emit_progress(
                stage="translating",
                phase="batch_start",
                pages_done=len(translated_pages),
                pages_total=pages_total,
                comments_done=offset,
                comments_total=len(items),
            )
            src_texts = [item.text_original for item in chunk]
            translations: list[str] = []
            used_batch = False

            try:
                candidate = self.provider.translate_text_batch(
                    src_texts,
                    source_lang=self.config.source_lang,
                    target_lang=self.config.target_lang,
                )
                if isinstance(candidate, list) and len(candidate) == len(chunk):
                    translations = [str(t or "").strip() for t in candidate]
                    used_batch = True
            except Exception as exc:
                error_text = str(exc)[:300]
                for item in chunk:
                    item.metadata["translation_error"] = error_text
                logging.warning(
                    "Batch text translation failed on %s items; keeping original text instead of issuing per-item requests: %s",
                    len(chunk),
                    exc,
                )
                self._emit_progress(
                    stage="translating",
                    phase="batch_error",
                    pages_done=len(translated_pages),
                    pages_total=pages_total,
                    comments_done=offset,
                    comments_total=len(items),
                    error=error_text,
                )

            if not translations:
                translations = list(src_texts)

            if used_batch:
                logging.debug("Translated %s text comments in batch mode", len(chunk))
                for idx, (src, tr) in enumerate(zip(src_texts, translations)):
                    if src.strip() and not tr.strip():
                        translations[idx] = src

            for item, translated in zip(chunk, translations):
                item.text_translated = translated
                translated_pages.add(item.page_index)
            self._emit_progress(
                stage="translating",
                phase="batch_done",
                pages_done=len(translated_pages),
                pages_total=pages_total,
                comments_done=min(len(items), offset + len(chunk)),
                comments_total=len(items),
            )

    def should_use_vision(self, text_result: PageExtractionResult, page: Any | None = None) -> bool:
        mode = self.config.vision_mode.lower()
        if mode == "always":
            return True
        if mode == "never":
            return False
        if text_result.text_chars < self.config.typed_text_threshold and not text_result.comments:
            return True
        has_native_annotations = any(item.source == "pdf_annotation" for item in text_result.comments)
        has_images = False
        if page is not None:
            try:
                has_images = bool(page.get_images(full=True))
            except Exception:
                has_images = False
        if (
            has_images
            and not has_native_annotations
            and text_result.text_chars < self.config.typed_text_threshold
        ):
            return True
        return False


def dump_results_json(path: Path, results: list[PageExtractionResult]) -> None:
    payload = []
    for result in results:
        payload.append(
            {
                "page_index": result.page_index,
                "text_chars": result.text_chars,
                "used_vision": result.used_vision,
                "notes": result.notes,
                "comments": [
                    {
                        "original": item.text_original,
                        "translated": item.text_translated,
                        "source": item.source,
                        "confidence": item.confidence,
                        "bbox": item.bbox,
                        "metadata": item.metadata,
                    }
                    for item in result.comments
                ],
            }
        )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_translation(
    input_pdf: Path | str,
    output_pdf: Path | str | None,
    *,
    config: Optional[PipelineConfig] = None,
    layout: Optional[LayoutConfig] = None,
) -> list[PageExtractionResult]:
    ensure_runtime_dependencies()
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {input_path}")

    effective_config = config or PipelineConfig()
    provider = build_provider(effective_config.provider)
    pipeline = HybridTranslatorPipeline(effective_config, provider)
    results = pipeline.run(input_path)

    if effective_config.dump_json:
        dump_results_json(effective_config.dump_json, results)

    if not effective_config.dry_run:
        if output_pdf is None:
            raise ValueError("output_pdf is required when dry_run is False")
        output_path = Path(output_pdf)
        writer = AnnotationWriter(layout or LayoutConfig(), progress_callback=effective_config.progress_callback)
        writer.write(input_path, output_path, results)

    return results


def _resolve_vision_mode(mode: str) -> str:
    normalized = mode.strip().lower().replace("_", " ").replace("-", " ")
    aliases = {
        "never": "never",
        "text": "never",
        "text only": "never",
        "text extraction only": "never",
        "always": "always",
        "vision": "always",
        "vision only": "always",
        "auto": "auto",
        "hybrid": "auto",
    }
    if normalized not in aliases:
        raise ValueError(f"Unsupported mode: {mode!r}. Use text/hybrid/vision or never/auto/always.")
    return aliases[normalized]


def _estimate_text_tokens(text: str) -> int:
    return max(1, math.ceil(len(text) / 4))


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def _extract_note_int(notes: list[str], key: str) -> int:
    prefix = f"{key}="
    for note in notes:
        if not isinstance(note, str):
            continue
        if note.startswith(prefix):
            try:
                return int(note[len(prefix) :].strip())
            except Exception:
                continue
    return 0


def analyze_pdf(
    input_pdf: Path | str,
    *,
    mode: str = "hybrid",
    typed_text_threshold: int = 120,
    prompt_overhead_tokens: int = 85,
    vision_input_tokens_per_page: int = 1200,
    vision_output_tokens_per_page: int = 220,
    input_cost_per_1m: float = 0.40,
    output_cost_per_1m: float = 1.60,
) -> dict[str, Any]:
    ensure_runtime_dependencies()
    input_path = Path(input_pdf)
    if not input_path.exists():
        raise FileNotFoundError(f"Input PDF does not exist: {input_path}")

    vision_mode = _resolve_vision_mode(mode)
    text_extractor = PDFTextExtractor()

    import fitz

    doc = fitz.open(str(input_path))
    pages: list[dict[str, Any]] = []
    total_text_comments = 0
    text_comments_for_translation: list[str] = []
    predicted_vision_pages = 0
    input_tokens = 0
    output_tokens = 0

    try:
        for page_index in range(doc.page_count):
            page = doc[page_index]
            text_result = text_extractor.extract_page_comments(page, page_index)

            if vision_mode == "always":
                use_vision = True
            elif vision_mode == "never":
                use_vision = False
            else:
                has_native_annotations = any(item.source == "pdf_annotation" for item in text_result.comments)
                try:
                    has_images = bool(page.get_images(full=True))
                except Exception:
                    has_images = False
                use_vision = (
                    text_result.text_chars < typed_text_threshold
                    and (not text_result.comments or (has_images and not has_native_annotations))
                )

            text_comments = [item.text_original for item in text_result.comments]
            total_text_comments += len(text_comments)
            if use_vision:
                predicted_vision_pages += 1
                input_tokens += max(1, vision_input_tokens_per_page)
                output_tokens += max(1, vision_output_tokens_per_page)
            else:
                text_comments_for_translation.extend(text_comments)

            pages.append(
                {
                    "page_index": page_index + 1,
                    "text_chars": text_result.text_chars,
                    "text_comments": text_comments,
                    "will_use_vision": use_vision,
                }
            )
    finally:
        doc.close()

    text_batch_size = _env_int("AUTOPDFTRANSLATOR_TEXT_BATCH_SIZE", 5000, 1, 20000)
    for offset in range(0, len(text_comments_for_translation), text_batch_size):
        chunk = text_comments_for_translation[offset : offset + text_batch_size]
        input_tokens += prompt_overhead_tokens + _estimate_text_tokens(json.dumps(chunk, ensure_ascii=False))
        output_tokens += sum(max(1, int(_estimate_text_tokens(comment) * 1.1)) for comment in chunk)

    total_tokens = input_tokens + output_tokens
    estimated_cost_usd = (
        (input_tokens / 1_000_000.0) * input_cost_per_1m
        + (output_tokens / 1_000_000.0) * output_cost_per_1m
    )
    return {
        "filename": input_path.name,
        "mode": vision_mode,
        "pages": pages,
        "totals": {
            "pages": len(pages),
            "text_comments": total_text_comments,
            "predicted_vision_pages": predicted_vision_pages,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "estimated_cost_usd": estimated_cost_usd,
        },
    }


def translate_pdf(
    input_pdf: Path | str,
    output_pdf: Path | str | None,
    *,
    source_lang: str = "en",
    target_lang: str = "zh-CN",
    mode: str = "hybrid",
    provider: str = "mock",
    typed_text_threshold: int = 120,
    dpi_for_vision: int = 220,
    dump_json: Optional[Path] = None,
    dry_run: bool = False,
    max_pages: Optional[int] = None,
    verbose: bool = False,
    layout: Optional[LayoutConfig] = None,
    progress_callback: Any = None,
) -> list[PageExtractionResult]:
    config = PipelineConfig(
        source_lang=source_lang,
        target_lang=target_lang,
        vision_mode=_resolve_vision_mode(mode),
        typed_text_threshold=typed_text_threshold,
        provider=provider,
        dpi_for_vision=dpi_for_vision,
        dump_json=dump_json,
        dry_run=dry_run,
        max_pages=max_pages,
        verbose=verbose,
        progress_callback=progress_callback,
    )
    return run_translation(
        input_pdf=input_pdf,
        output_pdf=output_pdf,
        config=config,
        layout=layout,
    )
