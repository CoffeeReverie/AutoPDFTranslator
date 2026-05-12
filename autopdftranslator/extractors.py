from __future__ import annotations

import importlib
import inspect
import json
import logging
import re
import tempfile
import textwrap
from pathlib import Path
from typing import Any

from .models import APP_NAME, CommentItem, PageExtractionResult

try:
    import fitz  # PyMuPDF
except Exception:  # pragma: no cover
    fitz = None  # type: ignore[assignment]

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]


def ensure_runtime_dependencies() -> None:
    missing: list[str] = []
    if fitz is None:
        missing.append("pymupdf")
    if Image is None:
        missing.append("pillow")
    if missing:
        raise RuntimeError(
            "Missing required dependencies: "
            + ", ".join(missing)
            + ". Install with: pip install pymupdf pillow"
        )


class PDFTextExtractor:
    def extract_page_comments(self, page: Any, page_index: int) -> PageExtractionResult:
        annotation_comments = extract_annotation_comments(page, page_index)
        if annotation_comments:
            return PageExtractionResult(
                page_index=page_index,
                comments=annotation_comments,
                text_chars=sum(len(item.text_original or "") for item in annotation_comments),
                used_vision=False,
                notes=[f"annotation_items={len(annotation_comments)}"],
            )

        raw = page.get_text("text")
        chars = len(raw or "")
        fragments = extract_text_fragments(page)
        comments = split_text_into_comment_items(page_index, raw, fragments=fragments)
        return PageExtractionResult(
            page_index=page_index,
            comments=comments,
            text_chars=chars,
            used_vision=False,
            notes=[f"fragments={len(fragments)}"],
        )


def extract_annotation_comments(page: Any, page_index: int) -> list[CommentItem]:
    items: list[CommentItem] = []
    try:
        annotations = list(page.annots() or [])
    except Exception:
        annotations = []

    seen: set[tuple[str, float, float]] = set()
    for annot in annotations:
        try:
            type_name = str(annot.type[1])
        except Exception:
            type_name = ""
        if type_name != "FreeText":
            continue

        info = getattr(annot, "info", {}) or {}
        subject = str(info.get("subject", "") or "")
        title = str(info.get("title", "") or "")
        if APP_NAME.lower() in subject.lower() or APP_NAME.lower() in title.lower():
            continue

        text = normalize_comment_text(str(info.get("content", "") or ""))
        if not text or _should_ignore_line(text):
            continue

        rect = getattr(annot, "rect", None)
        if rect is not None:
            bbox = (float(rect.x0), float(rect.y0), float(rect.x1), float(rect.y1))
        else:
            bbox = None
        key = (
            text.lower(),
            round(((bbox[0] + bbox[2]) * 0.5) if bbox else 0.0, 1),
            round(((bbox[1] + bbox[3]) * 0.5) if bbox else 0.0, 1),
        )
        if key in seen:
            continue
        seen.add(key)
        items.append(
            CommentItem(
                page_index=page_index,
                text_original=text,
                source="pdf_annotation",
                confidence=1.0,
                bbox=bbox,
                metadata={"annotation_type": type_name, "subject": subject},
            )
        )

    return items


def split_text_into_comment_items(
    page_index: int,
    raw_text: str,
    *,
    fragments: list[dict[str, Any]] | None = None,
) -> list[CommentItem]:
    if not raw_text:
        return []

    raw_text = raw_text.replace("\r", "\n")
    if fragments:
        comments = _split_fragments_into_comments(fragments)
        # Fragment-based grouping already used geometry; avoid a second text-only
        # merge pass that can accidentally stitch neighboring sticky notes together.
        merged_entries = comments
    else:
        lines = [normalize_comment_text(line) for line in raw_text.split("\n")]
        filtered = [line for line in lines if line and not _should_ignore_line(line)]
        comments = _split_lines_into_comments(filtered)

        buffer = ""
        merged = []
        for line in comments:
            if not buffer:
                buffer = line
                continue
            if looks_like_continuation(buffer, line) and not _looks_like_new_comment_start(line):
                buffer = f"{buffer} {line}".strip()
            else:
                merged.append(buffer)
                buffer = line
        if buffer:
            merged.append(buffer)
        merged_entries = [{"text": line, "bbox": None} for line in merged]

    seen: set[tuple[Any, ...] | str] = set()
    items: list[CommentItem] = []
    for entry in merged_entries:
        normalized = normalize_comment_text(str(entry.get("text", "")))
        if not normalized:
            continue
        bbox = entry.get("bbox")
        if isinstance(bbox, tuple) and len(bbox) == 4:
            x0, y0, x1, y1 = [float(v) for v in bbox]
            key: tuple[Any, ...] | str = (
                normalized.lower(),
                round((x0 + x1) * 0.5, 1),
                round((y0 + y1) * 0.5, 1),
            )
            item_bbox: tuple[float, float, float, float] | None = (x0, y0, x1, y1)
        else:
            key = normalized.lower()
            item_bbox = None
        if key in seen:
            continue
        seen.add(key)
        items.append(
            CommentItem(
                page_index=page_index,
                text_original=normalized,
                source="pdf_text",
                confidence=0.98,
                bbox=item_bbox,
            )
        )
    return items


def looks_like_continuation(buffer: str, line: str) -> bool:
    if buffer.endswith((":", "-", "—", ",", "/", "(", "[")):
        return True
    if line and line[0].islower():
        return True
    if re.match(r"^(and|or|to|with|for|of|in|on|at|from|by)\b", line.lower()):
        return True
    if len(line) <= 14 and re.match(r"^[a-z0-9\(\[]", line) and not re.search(r"[.!?]$", buffer):
        # Short tail fragments are usually wrapped continuation lines.
        return True
    return False


def extract_text_fragments(page: Any) -> list[dict[str, Any]]:
    fragments: list[dict[str, Any]] = []
    try:
        text_dict = page.get_text("dict")
    except Exception:
        return fragments

    blocks = text_dict.get("blocks", []) if isinstance(text_dict, dict) else []
    for block_idx, block in enumerate(blocks):
        if not isinstance(block, dict) or int(block.get("type", -1)) != 0:
            continue
        for line_idx, line in enumerate(block.get("lines", []) or []):
            if not isinstance(line, dict):
                continue
            spans = line.get("spans", []) or []
            span_text: list[str] = []
            for span in spans:
                if not isinstance(span, dict):
                    continue
                piece = normalize_comment_text(str(span.get("text", "")))
                if piece:
                    span_text.append(piece)
            text = normalize_comment_text(" ".join(span_text))
            if not text or _should_ignore_line(text):
                continue
            bbox = line.get("bbox")
            if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            fragments.append(
                {
                    "text": text,
                    "bbox": (x0, y0, x1, y1),
                    "block_idx": block_idx,
                    "line_idx": line_idx,
                }
            )
    return fragments


def _split_fragments_into_comments(fragments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not fragments:
        return []

    groups = _cluster_fragments_by_sticky_note(fragments)
    comments: list[dict[str, Any]] = []
    for group in groups:
        ordered = sorted(group, key=lambda f: (f["bbox"][1], f["bbox"][0]))
        text = normalize_comment_text(" ".join(normalize_comment_text(str(it.get("text", ""))) for it in ordered))
        if not text:
            continue
        x0 = min(float(it["bbox"][0]) for it in ordered)
        y0 = min(float(it["bbox"][1]) for it in ordered)
        x1 = max(float(it["bbox"][2]) for it in ordered)
        y1 = max(float(it["bbox"][3]) for it in ordered)
        comments.append({"text": text, "bbox": (x0, y0, x1, y1)})
    return comments


def _split_lines_into_comments(lines: list[str]) -> list[str]:
    comments: list[str] = []
    buffer = ""
    for line in lines:
        if not buffer:
            buffer = line
            continue
        if looks_like_continuation(buffer, line) and not _looks_like_new_comment_start(line):
            buffer = f"{buffer} {line}".strip()
        else:
            comments.append(buffer)
            buffer = line
    if buffer:
        comments.append(buffer)
    return comments


def _should_ignore_line(line: str) -> bool:
    text = normalize_comment_text(line)
    if len(text) <= 2:
        return True
    if not re.search(r"[A-Za-z]", text):
        return True
    low = text.lower()
    ignore_patterns = [
        r"^\d+$",
        r"^page\s+\d+(\s+of\s+\d+)?$",
        r"^icon-a",
        r"^draft$",
        r"^land$",
        r"^/\d+$",
        r"^\d{2}\.\d{2}\.\d{4}",
        r"^(sheet|drawing|drawn|checked|approved|revision|rev\.?|scale|date|title|project)\s*[:#]?\s*[\w\-/\. ]*$",
        r"^(plan|section|elevation|detail|legend|notes?|north|south|east|west)\s*$",
        r"^[a-z]{1,3}[-_ ]?\d{1,4}[a-z]?$",
        r"^rendering coordination$",
        r"^kccc$",
        r"^may\s+\d{4}$",
        r"^view\s*_?\s*\d+\s*//",
        r"^image priorities:?$",
        r"^time of day:?$",
        r"^senate\s+deped$",
        r"^\d{6}_[a-z0-9_\\/-]+$",
        r"^[a-z][a-z\s]+\\[a-z][a-z\s]+$",
    ]
    if any(re.match(pattern, low) for pattern in ignore_patterns):
        return True
    words = re.findall(r"[A-Za-z][A-Za-z0-9'\-]*", text)
    if 1 <= len(words) <= 5:
        review_markers = {
            "add",
            "adjust",
            "align",
            "avoid",
            "brighter",
            "change",
            "dark",
            "darker",
            "decrease",
            "delete",
            "fix",
            "increase",
            "less",
            "make",
            "missing",
            "more",
            "move",
            "need",
            "needs",
            "reduce",
            "remove",
            "replace",
            "revise",
            "show",
            "too",
            "update",
            "use",
            "wrong",
        }
        word_lows = {word.lower() for word in words}
        if not (word_lows & review_markers) and text == text.upper():
            return True
    return False


def _looks_like_new_comment_start(line: str) -> bool:
    return bool(re.match(r"^(\d+[\.\)]|[-*•])\s+\S+", line))


def _cluster_fragments_by_sticky_note(fragments: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    clusters: list[dict[str, Any]] = []
    ordered = sorted(fragments, key=lambda f: (f["bbox"][1], f["bbox"][0]))

    for frag in ordered:
        bbox = frag.get("bbox")
        if not isinstance(bbox, tuple) or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = [float(v) for v in bbox]
        line_h = max(0.1, y1 - y0)
        center_y = (y0 + y1) * 0.5

        best_idx: int | None = None
        best_score = float("inf")
        for idx, cluster in enumerate(clusters):
            x_tolerance = max(2.5, float(cluster["avg_h"]) * 2.8)
            y_tolerance = max(8.0, float(cluster["avg_h"]) * 1.7)
            x_gap = abs(x0 - float(cluster["avg_x0"]))
            if x_gap > x_tolerance:
                continue

            min_cy = float(cluster["min_cy"])
            max_cy = float(cluster["max_cy"])
            if center_y < min_cy:
                y_gap = min_cy - center_y
            elif center_y > max_cy:
                y_gap = center_y - max_cy
            else:
                y_gap = 0.0
            if y_gap > y_tolerance:
                continue

            score = x_gap + y_gap
            if score < best_score:
                best_score = score
                best_idx = idx

        if best_idx is None:
            clusters.append(
                {
                    "items": [frag],
                    "avg_x0": x0,
                    "avg_h": line_h,
                    "min_cy": center_y,
                    "max_cy": center_y,
                }
            )
            continue

        cluster = clusters[best_idx]
        cluster["items"].append(frag)
        count = len(cluster["items"])
        cluster["avg_x0"] = (float(cluster["avg_x0"]) * (count - 1) + x0) / count
        cluster["avg_h"] = (float(cluster["avg_h"]) * (count - 1) + line_h) / count
        cluster["min_cy"] = min(float(cluster["min_cy"]), center_y)
        cluster["max_cy"] = max(float(cluster["max_cy"]), center_y)

    sorted_clusters = sorted(clusters, key=lambda c: (float(c["min_cy"]), float(c["avg_x0"])))
    return [list(c["items"]) for c in sorted_clusters]


def _x_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    a_x0, _, a_x1, _ = a
    b_x0, _, b_x1, _ = b
    overlap = max(0.0, min(a_x1, b_x1) - max(a_x0, b_x0))
    shorter = max(1.0, min(a_x1 - a_x0, b_x1 - b_x0))
    return overlap / shorter


def extract_comments_with_opendataloader(input_pdf: Path | str) -> dict[int, list[CommentItem]]:
    input_path = Path(input_pdf)
    entry = _resolve_opendataloader_entrypoint()
    if entry is None:
        raise RuntimeError("opendataloader-pdf is not installed or no compatible parser entrypoint was found")

    payload = _run_opendataloader(entry, input_path)
    if payload is None:
        return {}

    fragments = _collect_odl_fragments(payload)
    by_page: dict[int, list[dict[str, Any]]] = {}
    for frag in fragments:
        page_index = int(frag["page_index"])
        by_page.setdefault(page_index, []).append(frag)

    out: dict[int, list[CommentItem]] = {}
    for page_index, page_frags in by_page.items():
        grouped = _split_fragments_into_comments(page_frags)
        items: list[CommentItem] = []
        seen: set[tuple[str, float, float]] = set()
        for entry in grouped:
            text = normalize_comment_text(str(entry.get("text", "")))
            bbox = entry.get("bbox")
            if not text or not isinstance(bbox, tuple) or len(bbox) != 4:
                continue
            x0, y0, x1, y1 = [float(v) for v in bbox]
            key = (text.lower(), round((x0 + x1) * 0.5, 1), round((y0 + y1) * 0.5, 1))
            if key in seen:
                continue
            seen.add(key)
            items.append(
                CommentItem(
                    page_index=page_index,
                    text_original=text,
                    source="odl_text",
                    confidence=0.90,
                    bbox=(x0, y0, x1, y1),
                    metadata={"extractor": "opendataloader"},
                )
            )
        if items:
            out[page_index] = items
    return out


def _resolve_opendataloader_entrypoint() -> Any:
    candidates = [
        ("opendataloader.pdf", "parse_file"),
        ("opendataloader.pdf", "convert"),
        ("opendataloader_pdf", "parse_file"),
        ("opendataloader_pdf", "convert"),
    ]
    for module_name, attr_name in candidates:
        try:
            module = importlib.import_module(module_name)
            attr = getattr(module, attr_name, None)
            if callable(attr):
                return attr
        except Exception:
            continue
    return None


def _run_opendataloader(entrypoint: Any, input_pdf: Path) -> Any:
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        signature = None
        try:
            signature = inspect.signature(entrypoint)
        except Exception:
            signature = None

        kwargs: dict[str, Any] = {}
        if signature is not None:
            param_names = set(signature.parameters.keys())
            path_keys = ["file_path", "input_file", "pdf_path", "path", "source", "filename"]
            for key in path_keys:
                if key in param_names:
                    kwargs[key] = str(input_pdf)
                    break
            if "output_dir" in param_names:
                kwargs["output_dir"] = str(temp_path)
            if "format" in param_names:
                kwargs["format"] = "json"
            if "output_format" in param_names:
                kwargs["output_format"] = "json"
            if "include_images" in param_names:
                kwargs["include_images"] = False
            if "extract_images" in param_names:
                kwargs["extract_images"] = False

        attempts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        if kwargs:
            attempts.append((tuple(), kwargs))
        attempts.append(((str(input_pdf),), {}))

        errors: list[str] = []
        result: Any = None
        for args, kw in attempts:
            try:
                result = entrypoint(*args, **kw)
                break
            except Exception as exc:
                errors.append(str(exc))
                continue

        payload = _coerce_odl_payload(result)
        if payload is not None:
            return payload

        json_candidates = sorted(temp_path.rglob("*.json"))
        for json_file in json_candidates:
            try:
                return json.loads(json_file.read_text(encoding="utf-8"))
            except Exception:
                continue

        if errors:
            raise RuntimeError(" ; ".join(errors[:2]))
        return None


def _coerce_odl_payload(result: Any) -> Any:
    if result is None:
        return None
    if isinstance(result, (dict, list)):
        return result
    if isinstance(result, (str, Path)):
        path = Path(str(result))
        if path.exists() and path.is_file() and path.suffix.lower() == ".json":
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                return None
        text = str(result).strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except Exception:
                return None
        return None

    for method_name in ("model_dump", "to_dict", "dict"):
        method = getattr(result, method_name, None)
        if callable(method):
            try:
                payload = method()
                if isinstance(payload, (dict, list)):
                    return payload
            except Exception:
                continue

    method_json = getattr(result, "json", None)
    if callable(method_json):
        try:
            payload_text = method_json()
            if isinstance(payload_text, str):
                return json.loads(payload_text)
        except Exception:
            pass

    return None


def _collect_odl_fragments(payload: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    def walk(node: Any, page_hint: int | None = None) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item, page_hint)
            return

        if not isinstance(node, dict):
            return

        page_index = _extract_page_index(node, page_hint)
        text = _extract_text_value(node)
        bbox = _extract_bbox_value(node)
        node_type = str(
            node.get("type", node.get("element_type", node.get("kind", node.get("label", ""))))
        ).strip().lower()

        if page_index is not None and text and bbox and _is_candidate_text_node(text, node_type):
            out.append(
                {
                    "page_index": page_index,
                    "text": text,
                    "bbox": bbox,
                    "block_idx": page_index,
                    "line_idx": len(out),
                }
            )

        for key, value in node.items():
            if key in {"text", "content", "raw_text", "bbox", "bounding_box"}:
                continue
            walk(value, page_index)

    walk(payload, None)
    return out


def _extract_page_index(node: dict[str, Any], page_hint: int | None) -> int | None:
    for key in ("page_index", "page", "page_num", "page_number", "pageNumber"):
        value = node.get(key)
        if isinstance(value, int):
            return value - 1 if value > 0 else value
        if isinstance(value, str) and value.strip().isdigit():
            num = int(value.strip())
            return num - 1 if num > 0 else num
    return page_hint


def _extract_text_value(node: dict[str, Any]) -> str:
    candidates = [
        node.get("text"),
        node.get("content"),
        node.get("raw_text"),
        node.get("value"),
    ]
    for value in candidates:
        if isinstance(value, str):
            text = normalize_comment_text(value.replace("\n", " "))
            if text:
                return text
    return ""


def _extract_bbox_value(node: dict[str, Any]) -> tuple[float, float, float, float] | None:
    for key in ("bbox", "bounding_box", "box", "bounds"):
        value = node.get(key)
        bbox = _coerce_bbox(value)
        if bbox is not None:
            return bbox
    return None


def _coerce_bbox(value: Any) -> tuple[float, float, float, float] | None:
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x0, y0, x1, y1 = [float(v) for v in value[:4]]
        if x1 >= x0 and y1 >= y0:
            return (x0, y0, x1, y1)
        return None
    if not isinstance(value, dict):
        return None

    keys_variant = [
        ("x0", "y0", "x1", "y1"),
        ("left", "top", "right", "bottom"),
        ("x", "y", "w", "h"),
    ]
    for a, b, c, d in keys_variant:
        if all(k in value for k in (a, b, c, d)):
            x0 = float(value[a])
            y0 = float(value[b])
            x1 = float(value[c])
            y1 = float(value[d])
            if (a, b, c, d) == ("x", "y", "w", "h"):
                x1 = x0 + x1
                y1 = y0 + y1
            if x1 >= x0 and y1 >= y0:
                return (x0, y0, x1, y1)
    return None


def _is_candidate_text_node(text: str, node_type: str) -> bool:
    if not text or len(text) <= 2:
        return False
    if len(text) > 450:
        return False
    if _should_ignore_line(text):
        return False

    disallow = {"image", "table", "figure", "formula", "header", "footer", "caption", "page"}
    if node_type in disallow:
        return False
    return True


class PageRenderer:
    def __init__(self, dpi: int = 220) -> None:
        self.dpi = dpi

    def render_page(self, page: Any) -> Any:
        zoom = self.dpi / 72.0
        matrix = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=matrix, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def merge_comment_sets(typed_items: list[CommentItem], vision_items: list[CommentItem]) -> list[CommentItem]:
    merged: list[CommentItem] = []
    for item in vision_items:
        merged.append(item)

    for typed in typed_items:
        if any(is_near_duplicate(typed.text_original, vision.text_original) for vision in vision_items):
            continue
        merged.append(typed)
    return merged


def is_near_duplicate(a: str, b: str) -> bool:
    a_norm = re.sub(r"\s+", " ", a.strip().lower())
    b_norm = re.sub(r"\s+", " ", b.strip().lower())
    if a_norm == b_norm:
        return True
    if not a_norm or not b_norm:
        return False
    if a_norm in b_norm or b_norm in a_norm:
        shorter = min(len(a_norm), len(b_norm))
        longer = max(len(a_norm), len(b_norm))
        return shorter / max(longer, 1) > 0.75
    return False


def wrap_mixed_text(text: str, max_chars: int) -> list[str]:
    parts = text.split(" ")
    if len(parts) == 1:
        return textwrap.wrap(text, width=max_chars, break_long_words=True, break_on_hyphens=False) or [text]

    lines: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + " " + part).strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = part
    if current:
        lines.append(current)

    out: list[str] = []
    for line in lines:
        if len(line) > max_chars:
            out.extend(textwrap.wrap(line, width=max_chars, break_long_words=True, break_on_hyphens=False))
        else:
            out.append(line)
    return out or [text]


def normalize_comment_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r-–—")
    return text.strip()
