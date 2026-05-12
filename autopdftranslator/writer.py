from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

from .extractors import wrap_mixed_text
from .models import APP_NAME, LayoutConfig, PageExtractionResult, TranslationCancelled

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]


class AnnotationWriter:
    def __init__(self, layout: LayoutConfig, progress_callback: Optional[Any] = None) -> None:
        self.layout = layout
        self.progress_callback = progress_callback

    def write(self, input_pdf: Path, output_pdf: Path, results: list[PageExtractionResult]) -> None:
        doc = fitz.open(str(input_pdf))
        try:
            page_map = {result.page_index: result for result in results}
            total_pages = len(page_map)
            written_pages = 0
            self._emit_progress(stage="writing", phase="start", pages_done=0, pages_total=total_pages)

            for page_index in range(doc.page_count):
                if page_index not in page_map:
                    continue
                self._write_page_annotations(doc[page_index], page_map[page_index])
                written_pages += 1
                self._emit_progress(
                    stage="writing",
                    phase="page_done",
                    page_index=page_index,
                    pages_done=written_pages,
                    pages_total=total_pages,
                )

            doc.save(str(output_pdf))
        finally:
            doc.close()

        self._clear_annotation_appearance_streams(output_pdf)

    def _emit_progress(self, **event: Any) -> None:
        if not callable(self.progress_callback):
            return
        try:
            self.progress_callback(event)
        except TranslationCancelled:
            raise
        except Exception:
            logging.debug("Progress callback failed", exc_info=True)

    def _write_page_annotations(self, page: Any, result: PageExtractionResult) -> None:
        annotation_specs = self._build_page_annotation_specs(page, result)
        logging.info(
            "Page %s: creating %s FreeText annotation boxes",
            result.page_index + 1,
            len(annotation_specs),
        )

        for place_rect, lines, item in annotation_specs:
            annot = page.add_freetext_annot(
                place_rect,
                "\n".join(lines),
                fontsize=self.layout.font_size,
                fontname=self.layout.font_name,
                text_color=(0, 0, 0),
                fill_color=(1, 1, 1),
                border_width=1,
                align=0,
                opacity=1,
                rotate=page.rotation,
            )
            info = annot.info
            info["subject"] = f"{APP_NAME}:{item.source}"
            info["title"] = APP_NAME
            info["content"] = "\n".join(lines)
            annot.set_info(info)

    def _build_page_annotation_specs(self, page: Any, result: PageExtractionResult) -> list[tuple[Any, list[str], Any]]:
        placement = self.layout.placement.strip().lower()
        if placement not in {"top-left-stacked", "top-right-stacked"}:
            placement = "top-left-stacked"

        x_disp = self.layout.left
        right_anchor = float(page.rect.width) - self.layout.left
        y_disp = self.layout.top
        display_w = float(page.rect.width)
        display_h = float(page.rect.height)
        specs: list[tuple[Any, list[str], Any]] = []

        for item in result.comments:
            text = item.text_translated or item.text_original
            lines, box_w, box_h = self._layout_text_box(text)

            if y_disp + box_h > display_h - 12:
                if placement == "top-right-stacked":
                    right_anchor -= self.layout.max_box_width + 10
                else:
                    x_disp += self.layout.max_box_width + 10
                y_disp = self.layout.top
                if placement == "top-right-stacked":
                    if right_anchor - box_w < 10:
                        right_anchor = display_w - self.layout.left
                else:
                    if x_disp + box_w > display_w - 10:
                        x_disp = self.layout.left

            if placement == "top-right-stacked":
                x_place = max(10.0, right_anchor - box_w)
            else:
                x_place = x_disp

            desired_rect = fitz.Rect(x_place, y_disp, x_place + box_w, y_disp + box_h)
            place_rect = desired_rect * page.derotation_matrix
            specs.append((place_rect, lines, item))
            y_disp += box_h + self.layout.gap_y
        return specs

    def _layout_text_box(self, text: str) -> tuple[list[str], float, float]:
        avg_char_w = self.layout.font_size * 0.62
        natural_w = min(
            self.layout.max_box_width,
            max(75.0, len(text) * avg_char_w + self.layout.pad_x * 2),
        )
        max_chars = max(6, int((natural_w - self.layout.pad_x * 2) / avg_char_w))
        lines = wrap_mixed_text(text, max_chars)
        longest = max(len(line) for line in lines) if lines else 1
        box_w = min(
            self.layout.max_box_width,
            max(75.0, longest * avg_char_w + self.layout.pad_x * 2 + 2),
        )
        box_h = len(lines) * (self.layout.font_size * self.layout.line_factor) + self.layout.pad_y * 2 + 1
        return lines, box_w, box_h

    def _clear_annotation_appearance_streams(self, pdf_path: Path) -> None:
        doc = fitz.open(str(pdf_path))
        fixed = 0
        try:
            for page in doc:
                annots = list(page.annots() or [])
                for annot in annots:
                    if annot.type[1] != "FreeText":
                        continue
                    content = annot.info.get("content") or ""
                    if any("\u4e00" <= ch <= "\u9fff" for ch in content):
                        try:
                            doc.xref_set_key(annot.xref, "AP", "null")
                            fixed += 1
                        except Exception:
                            logging.debug("Could not clear AP for annot xref=%s", annot.xref)
            doc.saveIncr()
        finally:
            doc.close()
        logging.info("Cleared appearance streams for %s FreeText annotations", fixed)
