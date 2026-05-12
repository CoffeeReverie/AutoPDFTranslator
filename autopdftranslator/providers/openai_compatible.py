from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from ..extractors import normalize_comment_text
from ..models import CommentItem

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


class OpenAICompatibleProvider:
    """
    Minimal OpenAI-compatible provider over chat completions.
    """

    def __init__(self) -> None:
        if requests is None:
            raise RuntimeError("requests is required for OpenAICompatibleProvider")
        self.base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = _normalize_bearer_api_key(os.getenv("OPENAI_API_KEY", ""))
        allow_empty_key = os.getenv("AUTOPDFTRANSLATOR_ALLOW_EMPTY_KEY", "").strip() == "1"
        is_local = self.base_url.startswith("http://localhost") or self.base_url.startswith("http://127.0.0.1")
        if not self.api_key and not allow_empty_key and not is_local:
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.model = os.getenv("AUTOPDFTRANSLATOR_MODEL", "gpt-4.1-mini")
        self.prompt_hint = os.getenv("AUTOPDFTRANSLATOR_PROMPT_HINT", "").strip()
        self.translation_memory = _load_translation_memory_from_env()

    def _post_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.1,
        max_tokens: int | None = None,
        timeout_seconds: int = 120,
    ) -> str:
        url = self._chat_completions_url()
        payload = {
            "model": self.model,
            "messages": messages,
        }
        if self._is_kimi():
            payload["thinking"] = {"type": "disabled"}
        else:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if self._is_minimax():
            payload["reasoning_split"] = True
        response = requests.post(
            url,
            headers=self._build_headers(),
            json=payload,
            timeout=timeout_seconds,
        )
        if response.status_code >= 400:
            detail = _extract_http_error_detail(response)
            raise RuntimeError(
                f"HTTP {response.status_code} calling {url}. "
                f"model={self.model}. details={detail}"
            )
        data = response.json()
        text = _extract_chat_output_text(data)
        if text:
            return text

        finish_reason = ""
        try:
            choices = data.get("choices", [])
            if choices and isinstance(choices[0], dict):
                finish_reason = str(choices[0].get("finish_reason", ""))
        except Exception:
            finish_reason = ""
        raise RuntimeError(
            "Chat completion returned empty textual content. "
            f"model={self.model}, finish_reason={finish_reason or '<unknown>'}, payload={str(data)[:1200]}"
        )

    def _resolve_temperature(self, requested: float) -> float:
        """
        Some OpenAI-compatible providers/models enforce fixed temperature values.
        Moonshot Kimi k2.5 currently requires temperature=1.
        """
        base = self.base_url.lower()
        model = self.model.lower()
        if model.startswith("kimi-k2.5") or ("moonshot.cn" in base and "kimi-k2.5" in model):
            return 1.0
        return requested

    def _is_kimi_k2_5(self) -> bool:
        model = self.model.lower()
        base = self.base_url.lower()
        return model.startswith("kimi-k2.5") or ("moonshot.cn" in base and "kimi-k2.5" in model)

    def _is_kimi(self) -> bool:
        model = self.model.lower()
        base = self.base_url.lower()
        return model.startswith("kimi") or (("moonshot.cn" in base or "moonshot.ai" in base) and "kimi" in model)

    def _supports_responses_endpoint(self) -> bool:
        base = self.base_url.lower()
        if "moonshot.cn" in base:
            return False
        return True

    def _is_minimax(self) -> bool:
        base = self.base_url.lower()
        return "minimax.io" in base or "minimaxi.com" in base

    def _is_ollama(self) -> bool:
        base = self.base_url.lower().rstrip("/")
        return "localhost:11434" in base or "127.0.0.1:11434" in base

    def _is_local_endpoint(self) -> bool:
        base = self.base_url.lower()
        return "localhost" in base or "127.0.0.1" in base

    def _chat_completions_url(self) -> str:
        base = self.base_url.rstrip("/")
        if self._is_ollama() and not base.lower().endswith("/v1"):
            base = f"{base}/v1"
        return f"{base}/chat/completions"

    def _is_gemini(self) -> bool:
        model = self.model.lower()
        base = self.base_url.lower()
        return "generativelanguage.googleapis.com" in base or model.startswith("gemini")

    def _is_translategemma(self) -> bool:
        return "translategemma" in self.model.lower()

    def _post_responses_vision(self, prompt: str, data_url: str) -> str:
        url = self.base_url.rstrip("/") + "/responses"
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }
            ],
            "max_output_tokens": 1200,
        }
        response = requests.post(
            url,
            headers=self._build_headers(),
            json=payload,
            timeout=120,
        )
        if response.status_code >= 400:
            detail = _extract_http_error_detail(response)
            raise RuntimeError(
                f"HTTP {response.status_code} calling {url}. "
                f"model={self.model}. details={detail}"
            )
        data = response.json()
        text = _extract_responses_output_text(data)
        if not text:
            raise RuntimeError(f"Responses API returned no text content. payload={str(data)[:1200]}")
        return text

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        text = normalize_comment_text(text)
        if not text:
            return ""
        if self._is_translategemma():
            memory_hint = self._build_memory_hint([text], include_fallback_terms=False)
            prompt = _build_translategemma_prompt(
                text,
                source_lang,
                target_lang,
                prompt_hint=self.prompt_hint,
                memory_hint=memory_hint,
            )
            translated = self._post_chat(
                [{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=_env_int("AUTOPDFTRANSLATOR_TRANSLATEGEMMA_MAX_TOKENS", 512, 128, 4096),
            ).strip()
            return _polish_archviz_translation(text, translated, target_lang)
        prompt = (
            f"Translate the following review comment from {source_lang} to {target_lang}. "
            "Keep it concise, accurate, and suitable for architecture / visualization review markup. "
            f"Return only the translation.\n\n{text}"
        )
        system_prompt = self._build_system_prompt([text])
        translated = self._post_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        ).strip()
        return _polish_archviz_translation(text, translated, target_lang)

    def translate_text_batch(self, texts: list[str], source_lang: str, target_lang: str) -> list[str]:
        cleaned = [normalize_comment_text(text) for text in texts]
        if not cleaned:
            return []

        non_empty_indices = [idx for idx, text in enumerate(cleaned) if text]
        if not non_empty_indices:
            return [""] * len(cleaned)

        payload_items = [cleaned[idx] for idx in non_empty_indices]
        if self._is_translategemma():
            out = [""] * len(cleaned)
            translated_payload = self._translate_text_batch_translategemma(payload_items, source_lang, target_lang)
            if len(translated_payload) != len(payload_items):
                translated_payload = [
                    self.translate_text(text, source_lang=source_lang, target_lang=target_lang)
                    for text in payload_items
                ]
            for local_idx, translated in enumerate(translated_payload):
                source_idx = non_empty_indices[local_idx]
                out[source_idx] = normalize_comment_text(
                    _polish_archviz_translation(cleaned[source_idx], translated, target_lang)
                )
            return out
        if self._is_local_endpoint():
            local_batch_size = _env_int("AUTOPDFTRANSLATOR_LOCAL_TEXT_BATCH_SIZE", 8, 1, 200)
            if len(payload_items) > local_batch_size:
                translated_payload: list[str] = []
                for offset in range(0, len(payload_items), local_batch_size):
                    translated_payload.extend(
                        self.translate_text_batch(
                            payload_items[offset : offset + local_batch_size],
                            source_lang=source_lang,
                            target_lang=target_lang,
                        )
                    )
                out = [""] * len(cleaned)
                for local_idx, translated in enumerate(translated_payload):
                    source_idx = non_empty_indices[local_idx]
                    out[source_idx] = normalize_comment_text(translated)
                return out

        logging.info("Sending one text translation request for %s comments", len(payload_items))
        prompt = (
            f"Translate each review comment from {source_lang} to {target_lang}. "
            "Return a strict JSON string array only. "
            "Keep the same item count and index order. "
            "Do not merge, split, remove, explain, or add items. "
            "Keep translations accurate but as short as possible for PDF markup. "
            "Preserve proper nouns, project/site/building names, brand names, and signage text unless a glossary says otherwise. "
            "Use concise architecture / visualization review wording. "
            "Return only the translated array, no keys or prose.\n\n"
            f"Input JSON array:\n{json.dumps(payload_items, ensure_ascii=False)}"
        )
        system_prompt = self._build_system_prompt(payload_items)
        max_tokens = _env_int(
            "AUTOPDFTRANSLATOR_TEXT_MAX_TOKENS",
            max(800, min(16000, 80 * len(payload_items))),
            256,
            32768,
        )

        raw = self._post_chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_seconds=180,
        )

        parsed = _parse_translation_batch_json(raw)
        if len(parsed) > len(payload_items):
            parsed = parsed[: len(payload_items)]
        if not parsed or len(parsed) != len(payload_items):
            raise RuntimeError(
                f"Batch translation size mismatch: expected={len(payload_items)}, got={len(parsed)}"
            )

        out = [""] * len(cleaned)
        for local_idx, translated in enumerate(parsed):
            source_idx = non_empty_indices[local_idx]
            out[source_idx] = normalize_comment_text(
                _polish_archviz_translation(cleaned[source_idx], translated, target_lang)
            )
        return out

    def _translate_text_batch_translategemma(
        self,
        texts: list[str],
        source_lang: str,
        target_lang: str,
    ) -> list[str]:
        batch_size = _env_int("AUTOPDFTRANSLATOR_TRANSLATEGEMMA_BATCH_SIZE", 40, 1, 200)
        if len(texts) > batch_size:
            out: list[str] = []
            for offset in range(0, len(texts), batch_size):
                out.extend(
                    self._translate_text_batch_translategemma(
                        texts[offset : offset + batch_size],
                        source_lang=source_lang,
                        target_lang=target_lang,
                    )
                )
            return out

        memory_hint = self._build_memory_hint(texts, include_fallback_terms=False)
        prompt = _build_translategemma_batch_prompt(
            texts,
            source_lang,
            target_lang,
            prompt_hint=self.prompt_hint,
            memory_hint=memory_hint,
        )
        max_tokens = _env_int(
            "AUTOPDFTRANSLATOR_TRANSLATEGEMMA_BATCH_MAX_TOKENS",
            max(512, min(8192, 80 * len(texts))),
            256,
            16384,
        )
        raw = self._post_chat(
            [{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=max_tokens,
            timeout_seconds=180,
        )
        parsed = _parse_numbered_translations(raw, len(texts))
        if len(parsed) != len(texts):
            return []
        return [
            _polish_archviz_translation(source, translated, target_lang)
            for source, translated in zip(texts, parsed)
        ]

    def vision_extract_and_translate(
        self,
        image: Any,
        page_index: int,
        source_lang: str,
        target_lang: str,
    ) -> list[CommentItem]:
        prompt = (
            "Read this review / markup page. Extract design review comments and markup notes in their original language. Do not translate. "
            "You must inspect both printed text and handwritten markups, including faint pencil/pen notes, small handwriting, arrows, callouts, and sketch annotations. "
            "For handwriting, include short labels, floor-count notes, material notes, removal notes, arrows, and fragmentary sketch annotations when they communicate design intent. "
            "Ignore title blocks, page numbers, logos, decorative labels, signatures, author/company names, dates, revision/file codes, and lower-right stamp text unless they are actual comments. "
            "Do not translate sign-off text such as names, dates, project codes, drawing numbers, or revision labels. "
            "Output format is strict JSON array only. Each item must be an object with exactly one key: original. "
            "Example: [{\"original\":\"...\"},{\"original\":\"...\"}]. "
            "Do not include translated, confidence, type, uncertain, or any other keys. "
            "Do not output analysis, reasoning, markdown, numbering, or prose. Return final JSON directly. "
            "Keep each extracted line concise and practical for PDF annotations. "
            f"Source language: {source_lang}. Target language: {target_lang}."
        )
        if self.prompt_hint:
            prompt = (
                f"{prompt}\nAdditional instruction: {self.prompt_hint}\n"
                "If any instruction conflicts with output format, keep the strict output format above."
            )
        memory_hint = self._build_memory_hint([], include_fallback_terms=True)
        if memory_hint:
            prompt = (
                f"{prompt}\nTranslation memory:\n{memory_hint}\n"
                "Use this memory only when relevant. If any memory conflicts with visible page content, trust the page."
            )
        errors: list[str] = []
        raw = ""
        is_kimi = self._is_kimi()
        is_gemini = self._is_gemini()
        is_local = self._is_local_endpoint()
        default_timeout = 60 if is_local else (300 if is_kimi else 180)
        default_tokens = 1024 if is_local else (8192 if is_kimi else (4096 if is_gemini else 1200))
        default_max_side = 1800 if is_local else (2200 if is_kimi else (3400 if is_gemini else 2600))
        default_max_bytes = 1_800_000 if is_local else (3_500_000 if is_kimi else 5_000_000)
        timeout_seconds = _env_int("AUTOPDFTRANSLATOR_VISION_TIMEOUT_SECONDS", default_timeout, 30, 600)
        vision_max_tokens = _env_int("AUTOPDFTRANSLATOR_VISION_MAX_TOKENS", default_tokens, 256, 16384)
        if is_kimi:
            vision_max_tokens = max(8192, vision_max_tokens)
        retry_attempts = _env_int("AUTOPDFTRANSLATOR_VISION_RETRY_ATTEMPTS", 2 if is_kimi else 1, 1, 4)
        vision_max_side = _env_int("AUTOPDFTRANSLATOR_VISION_MAX_SIDE", default_max_side, 1000, 4200)
        vision_max_bytes = _env_int("AUTOPDFTRANSLATOR_VISION_MAX_BYTES", default_max_bytes, 1_500_000, 10_000_000)
        zoom_enabled = os.getenv("AUTOPDFTRANSLATOR_VISION_ZOOM", "1").strip() != "0"
        debug_dir = _vision_debug_dir()
        if debug_dir is not None:
            _save_debug_image(debug_dir, page_index, "full_page", image)

        if is_gemini:
            try:
                full_page_items = self._vision_full_page_extract(
                    image=image,
                    page_index=page_index,
                    prompt=prompt,
                    is_kimi=is_kimi,
                    timeout_seconds=timeout_seconds,
                    vision_max_tokens=vision_max_tokens,
                    vision_max_side=vision_max_side,
                    vision_max_bytes=vision_max_bytes,
                    debug_dir=debug_dir,
                )
                if full_page_items:
                    return full_page_items
            except Exception as exc:
                errors.append(str(exc))
                logging.info("Gemini full-page vision pass failed, falling back to zoom/full-page retry: %s", exc)

        if zoom_enabled:
            try:
                roi_items = self._vision_roi_panel_extract(
                    image=image,
                    page_index=page_index,
                    is_kimi=is_kimi,
                    timeout_seconds=timeout_seconds,
                    vision_max_tokens=vision_max_tokens,
                    vision_max_bytes=vision_max_bytes,
                    debug_dir=debug_dir,
                )
                if roi_items:
                    return roi_items
            except Exception as exc:
                if debug_dir is not None:
                    _save_debug_text(debug_dir, page_index, "roi_panel_request_error", str(exc))
                logging.info("Vision ROI panel pass failed, falling back to locate/crop zoom: %s", exc)
            if is_local:
                return []
            try:
                zoom_items = self._vision_zoom_extract_and_translate(
                    image=image,
                    page_index=page_index,
                    base_prompt=prompt,
                    is_kimi=is_kimi,
                    timeout_seconds=timeout_seconds,
                    vision_max_tokens=vision_max_tokens,
                    vision_max_bytes=vision_max_bytes,
                    debug_dir=debug_dir,
                )
                if zoom_items:
                    return zoom_items
            except Exception as exc:
                if debug_dir is not None:
                    _save_debug_text(debug_dir, page_index, "zoom_request_error", str(exc))
                logging.info("Vision zoom pass failed, falling back to full-page vision: %s", exc)

        # For Kimi k2.5, favor fewer format retries and smaller fallback image on retry.
        attempt_plans: list[tuple[int, int, int]] = []
        current_side = vision_max_side
        for _ in range(retry_attempts):
            attempt_plans.append((current_side, vision_max_tokens, timeout_seconds))
            current_side = max(1000, int(current_side * 0.86))
        for max_side, vision_max_tokens, timeout_seconds in attempt_plans:
            image_data, mime = _encode_image_for_vision(image, max_side=max_side, max_bytes=vision_max_bytes)
            b64 = base64.b64encode(image_data).decode("ascii")
            data_url = f"data:{mime};base64,{b64}"

            content_variants = [
                [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ]
            ]
            if not is_kimi:
                content_variants.append(
                    [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": data_url},
                    ]
                )

            for content in content_variants:
                try:
                    raw = self._post_chat(
                        [
                            {"role": "system", "content": "You are a precise multimodal markup reader."},
                            {"role": "user", "content": content},
                        ],
                        temperature=0.0,
                        max_tokens=vision_max_tokens,
                        timeout_seconds=timeout_seconds,
                    )
                    break
                except Exception as exc:
                    errors.append(str(exc))
            if raw:
                break

        if not raw:
            if self._supports_responses_endpoint():
                try:
                    raw = self._post_responses_vision(prompt, data_url)
                except Exception as exc:
                    errors.append(str(exc))
                    joined = " || ".join(errors[:3])
                    raise RuntimeError(
                        "Vision request failed after trying chat.completions and responses formats. "
                        f"{joined}"
                    ) from exc
            else:
                joined = " || ".join(errors[:3]) or "No usable response from chat.completions."
                raise RuntimeError(
                    "Vision request failed on chat.completions. "
                    f"{joined}"
                )

        try:
            parsed = json.loads(_extract_json_block(raw))
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(f"Vision response was not valid JSON: {raw[:500]!r}") from exc
        if debug_dir is not None:
            _save_debug_text(debug_dir, page_index, "full_page_raw", raw)
            _save_debug_text(debug_dir, page_index, "full_page_parsed", json.dumps(parsed, ensure_ascii=False, indent=2))

        return _parse_vision_items(parsed, page_index)

    def _vision_full_page_extract(
        self,
        *,
        image: Any,
        page_index: int,
        prompt: str,
        is_kimi: bool,
        timeout_seconds: int,
        vision_max_tokens: int,
        vision_max_side: int,
        vision_max_bytes: int,
        debug_dir: Path | None,
    ) -> list[CommentItem]:
        full_page_prompt = (
            f"{prompt}\n\n"
            "Read the full page at multiple visual scales before answering. "
            "First understand the overall architectural image, then inspect small handwritten notes, arrows, floor-count labels, and callouts. "
            "Use architecture context to read common terms such as PVs, DIAGRID, storeys, recess, podium, driveway, street lamps, openings, remove building, people, trees, and creekside park. "
            "Extract only visible review markup content that should be translated. "
            "Ignore title-block and signature/date content such as company names, author names, issue dates, drawing/file numbers, revision codes, and lower-right stamps. "
            "Return one JSON item per distinct note or label. Do not categorize, explain, or include alternatives."
        )
        raw = self._post_vision_chat(
            prompt=full_page_prompt,
            image=image,
            is_kimi=is_kimi,
            max_side=vision_max_side,
            max_bytes=vision_max_bytes,
            max_tokens=vision_max_tokens,
            timeout_seconds=timeout_seconds,
        )
        if debug_dir is not None:
            _save_debug_text(debug_dir, page_index, "gemini_full_page_raw", raw)
        parsed = json.loads(_extract_json_block(raw))
        if debug_dir is not None:
            _save_debug_text(
                debug_dir,
                page_index,
                "gemini_full_page_parsed",
                json.dumps(parsed, ensure_ascii=False, indent=2),
        )
        return _parse_vision_items(parsed, page_index)

    def _vision_roi_panel_extract(
        self,
        *,
        image: Any,
        page_index: int,
        is_kimi: bool,
        timeout_seconds: int,
        vision_max_tokens: int,
        vision_max_bytes: int,
        debug_dir: Path | None,
    ) -> list[CommentItem]:
        regions = _detect_dark_markup_regions(image)
        if not regions:
            return []
        is_local = self._is_local_endpoint()
        local_default_regions = 18 if is_local else 10
        max_regions = _env_int("AUTOPDFTRANSLATOR_VISION_ROI_MAX_REGIONS", 18 if is_kimi else local_default_regions, 1, 24)
        selected_regions = regions[:max_regions]
        crops = _crop_regions_with_padding(image, selected_regions)
        if not crops:
            return []
        prompt = (
            "You are reading zoomed regions from an architectural review page. "
            "Each tile is labeled R1, R2, etc. The tiles were locally cropped from likely handwritten/markup regions and enhanced for handwriting. "
            "Extract visible English review comments, handwritten notes, callouts, floor-count labels, and markup instructions from all tiles. "
            "Use architecture context for terms such as storeys, F/floor, PVs, DIAGRID, recess, podium, driveway, street lamps, openings, remove building, people, trees, creekside park, green roof, and warm light. "
            "For floor labels, read F as floor, not P. For example, 33/F means 33 floors. "
            "Ignore title blocks, signatures, dates, company/author names, file codes, drawing numbers, logos, and lower-right stamp text unless they are actual review comments. "
            "Do not translate. Do not guess. Do not include alternatives with 'or', 'maybe', brackets, or question marks. "
            "Return strict JSON array only. Each item must be {\"region\":\"R1\",\"original\":\"text\"}. "
            "If a tile has no readable review comment, omit it. Return [] if no review comments are readable."
        )

        chunk_size = _env_int("AUTOPDFTRANSLATOR_LOCAL_VISION_ROI_CHUNK_SIZE", 3, 1, 12) if is_local else len(crops)
        items: list[CommentItem] = []
        if debug_dir is not None:
            _save_debug_text(
                debug_dir,
                page_index,
                "roi_panel_regions",
                json.dumps(selected_regions, ensure_ascii=False, indent=2),
            )
        for chunk_index, offset in enumerate(range(0, len(crops), chunk_size), start=1):
            chunk = crops[offset : offset + chunk_size]
            panel = _make_roi_contact_sheet(chunk)
            debug_suffix = "" if chunk_index == 1 else f"_{chunk_index:02d}"
            if debug_dir is not None:
                _save_debug_image(debug_dir, page_index, f"roi_panel{debug_suffix}", panel)

            raw = self._post_vision_chat(
                prompt=prompt,
                image=panel,
                is_kimi=is_kimi,
                max_side=_env_int(
                    "AUTOPDFTRANSLATOR_VISION_ROI_MAX_SIDE",
                    1800 if is_local else 3600,
                    1000,
                    5200,
                ),
                max_bytes=vision_max_bytes,
                max_tokens=_env_int(
                    "AUTOPDFTRANSLATOR_VISION_ROI_MAX_TOKENS",
                    4096 if is_kimi else min(vision_max_tokens, 4096),
                    512,
                    8192,
                ),
                timeout_seconds=timeout_seconds,
            )
            if debug_dir is not None:
                _save_debug_text(debug_dir, page_index, f"roi_panel_raw{debug_suffix}", raw)
            parsed = json.loads(_extract_json_block(raw))
            if debug_dir is not None:
                _save_debug_text(
                    debug_dir,
                    page_index,
                    f"roi_panel_parsed{debug_suffix}",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                )
            chunk_items = _parse_vision_items(parsed, page_index)
            for item in chunk_items:
                item.metadata["vision_region"] = f"roi_panel_{chunk_index}"
            items.extend(chunk_items)
            if is_local:
                refined_items = self._vision_refine_sparse_roi_chunk(
                    crops=chunk,
                    chunk_items=chunk_items,
                    page_index=page_index,
                    is_kimi=is_kimi,
                    timeout_seconds=timeout_seconds,
                    vision_max_bytes=vision_max_bytes,
                    debug_dir=debug_dir,
                    chunk_index=chunk_index,
                )
                items.extend(refined_items)

        if debug_dir is not None:
            _save_debug_text(
                debug_dir,
                page_index,
                "roi_panel_all_items",
                json.dumps(
                    [{"original": item.text_original, "metadata": item.metadata} for item in items],
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        if is_kimi and len(items) < _env_int("AUTOPDFTRANSLATOR_VISION_EXPECTED_MIN_ITEMS", 18, 1, 60):
            try:
                supplemental = self._vision_full_page_missing_extract(
                    image=image,
                    page_index=page_index,
                    existing_items=items,
                    is_kimi=is_kimi,
                    timeout_seconds=timeout_seconds,
                    vision_max_bytes=vision_max_bytes,
                    debug_dir=debug_dir,
                )
                items.extend(supplemental)
            except Exception as exc:
                if debug_dir is not None:
                    _save_debug_text(debug_dir, page_index, "full_page_missing_request_error", str(exc))
                logging.info("Vision full-page missing pass failed: %s", exc)
        return _dedupe_comment_items(items)

    def _vision_refine_sparse_roi_chunk(
        self,
        *,
        crops: list[Any],
        chunk_items: list[CommentItem],
        page_index: int,
        is_kimi: bool,
        timeout_seconds: int,
        vision_max_bytes: int,
        debug_dir: Path | None,
        chunk_index: int,
    ) -> list[CommentItem]:
        if not crops:
            return []
        has_suspicious_item = any(
            _looks_like_local_vision_misread(
                str(item.metadata.get("vision_raw_original", "")) or item.text_original
            )
            for item in chunk_items
        )
        if len(chunk_items) >= max(2, len(crops) // 2) and not has_suspicious_item:
            return []
        max_refine = _env_int("AUTOPDFTRANSLATOR_LOCAL_VISION_REFINE_MAX_ROIS", 6, 0, 12)
        if max_refine <= 0:
            return []
        out: list[CommentItem] = []
        for local_idx, crop in enumerate(crops[:max_refine], start=1):
            panel = _make_handwriting_enhanced_panel(crop)
            if debug_dir is not None:
                _save_debug_image(debug_dir, page_index, f"roi_refine_{chunk_index:02d}_{local_idx:02d}", panel)
            prompt = (
                "Read this single zoomed region from an architectural review image. "
                "Extract every visible English handwritten review comment or floor/material/markup label. "
                "Pay special attention to small floor labels such as 20 storeys, 24 storeys, 25 storeys, 27 storeys, 33/F, Total 31 storeys; "
                "instructions such as add PVs, add openings, remove building, recess, same DIAGRID, add people & activity, people & trees, see model. "
                "For floor labels, read F as floor, not P. Do not translate. Do not guess. "
                "Return strict JSON array only: [{\"original\":\"text\"}]. Return [] if no readable review note is visible."
            )
            try:
                raw = self._post_vision_chat(
                    prompt=prompt,
                    image=panel,
                    is_kimi=is_kimi,
                    max_side=_env_int("AUTOPDFTRANSLATOR_LOCAL_VISION_REFINE_MAX_SIDE", 2200, 1000, 3600),
                    max_bytes=vision_max_bytes,
                    max_tokens=_env_int("AUTOPDFTRANSLATOR_LOCAL_VISION_REFINE_MAX_TOKENS", 1024, 256, 4096),
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                if debug_dir is not None:
                    _save_debug_text(debug_dir, page_index, f"roi_refine_{chunk_index:02d}_{local_idx:02d}_error", str(exc))
                continue
            if debug_dir is not None:
                _save_debug_text(debug_dir, page_index, f"roi_refine_{chunk_index:02d}_{local_idx:02d}_raw", raw)
            try:
                parsed = json.loads(_extract_json_block(raw))
            except Exception:
                continue
            refined = _parse_vision_items(parsed, page_index)
            for item in refined:
                item.metadata["vision_region"] = f"roi_panel_{chunk_index}"
                item.metadata["vision_tile_region"] = f"R{local_idx}"
                item.metadata["vision_pass"] = "roi_refine"
            out.extend(refined)
        return out

    def _vision_full_page_missing_extract(
        self,
        *,
        image: Any,
        page_index: int,
        existing_items: list[CommentItem],
        is_kimi: bool,
        timeout_seconds: int,
        vision_max_bytes: int,
        debug_dir: Path | None,
    ) -> list[CommentItem]:
        existing_texts = [item.text_original for item in existing_items if item.text_original]
        prompt = (
            "Read the entire architectural review page and find visible handwritten English review comments or markup labels that are missing from the existing list. "
            "This is a supplement pass after OCR from cropped regions, so focus on omissions: handwritten notes, floor-count labels, arrows/callouts, roof notes, removal/addition/recess instructions, landscape/public-realm notes, and model/photo references. "
            "Use architecture context for terms such as storeys, F/floor, PVs, DIAGRID, recess, podium, driveway, street lamps, openings, remove building, people, trees, creekside park, green roof, privacy, extension, and warm light. "
            "Do not repeat items already in the existing list unless the visible text is clearly a separate note at another location. "
            "Ignore lower-right title block, signatures, dates, company/author names, file codes, drawing numbers, logos, and page numbers. "
            "Do not translate. Do not include guesses, alternatives, brackets, or question marks. "
            "Return strict JSON array only. Each item must be {\"original\":\"text\"}. Return [] if no additional review comments are readable.\n\n"
            f"Existing list JSON:\n{json.dumps(existing_texts, ensure_ascii=False)}"
        )
        raw = self._post_vision_chat(
            prompt=prompt,
            image=image,
            is_kimi=is_kimi,
            max_side=_env_int("AUTOPDFTRANSLATOR_VISION_MISSING_MAX_SIDE", 3600, 1600, 5200),
            max_bytes=vision_max_bytes,
            max_tokens=_env_int("AUTOPDFTRANSLATOR_VISION_MISSING_MAX_TOKENS", 4096, 512, 8192),
            timeout_seconds=timeout_seconds,
        )
        if debug_dir is not None:
            _save_debug_text(debug_dir, page_index, "full_page_missing_raw", raw)
        parsed = json.loads(_extract_json_block(raw))
        if debug_dir is not None:
            _save_debug_text(
                debug_dir,
                page_index,
                "full_page_missing_parsed",
                json.dumps(parsed, ensure_ascii=False, indent=2),
            )
        items = _parse_vision_items(parsed, page_index)
        for item in items:
            item.metadata["vision_region"] = "full_page_missing"
        return items

    def _vision_zoom_extract_and_translate(
        self,
        *,
        image: Any,
        page_index: int,
        base_prompt: str,
        is_kimi: bool,
        timeout_seconds: int,
        vision_max_tokens: int,
        vision_max_bytes: int,
        debug_dir: Path | None,
    ) -> list[CommentItem]:
        locate_prompt = (
            "Inspect the full page and locate regions that contain actionable review comments or handwritten markup. "
            "Act like a human reviewer deciding where to zoom in. Include faint handwriting, small handwritten notes, arrows, callouts, and printed review comments. "
            "Ignore empty image areas, title blocks, page numbers, logos, decorative labels, signatures, author/company names, dates, revision/file codes, and lower-right stamp text. "
            "Return strict JSON array only. Each item must be {\"bbox_pct\":[x0,y0,x1,y1]}. "
            "Coordinates are percentages from 0 to 100 relative to the full page image. "
            "Use one box per nearby cluster of comments; make boxes generous enough to include the whole note and arrow context. "
            "Return [] if there are no review comments."
        )
        locate_raw = self._post_vision_chat(
            prompt=locate_prompt,
            image=image,
            is_kimi=is_kimi,
            max_side=_env_int("AUTOPDFTRANSLATOR_VISION_LOCATE_MAX_SIDE", 1800, 1000, 3200),
            max_bytes=vision_max_bytes,
            max_tokens=1200,
            timeout_seconds=timeout_seconds,
        )
        if debug_dir is not None:
            _save_debug_text(debug_dir, page_index, "zoom_locate_raw", locate_raw)
        ai_regions = _parse_vision_regions(locate_raw)
        heuristic_regions = _detect_dark_markup_regions(image)
        regions = _dedupe_regions([*ai_regions, *heuristic_regions])
        if debug_dir is not None:
            _save_debug_text(
                debug_dir,
                page_index,
                "zoom_regions",
                json.dumps(
                    {
                        "ai_regions": ai_regions,
                        "heuristic_regions": heuristic_regions,
                        "merged_regions": regions,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
            )
        logging.info(
            "Vision zoom candidate regions: ai=%s heuristic=%s merged=%s",
            len(ai_regions),
            len(heuristic_regions),
            len(regions),
        )
        if not regions:
            return []

        max_regions = _env_int("AUTOPDFTRANSLATOR_VISION_ZOOM_MAX_REGIONS", 2, 1, 20)
        crops = _crop_regions_with_padding(image, regions[:max_regions])
        items: list[CommentItem] = []
        for idx, crop in enumerate(crops, start=1):
            crop_prompt = (
                "Extract readable handwritten or printed review notes from this zoomed crop. "
                "The image may show original and high-contrast copies of the same crop; use both. "
                "Return strict JSON only: [{\"original\":\"text\"}]. "
                "Do not translate. Do not explain. Do not include guesses with 'or', 'maybe', 'something', or brackets. "
                "Ignore signatures, dates, title blocks, file codes, page numbers, and lower-right stamps."
            )
            crop_for_model = _make_handwriting_enhanced_panel(crop)
            if debug_dir is not None:
                _save_debug_image(debug_dir, page_index, f"zoom_crop_{idx:02d}_source", crop)
                _save_debug_image(debug_dir, page_index, f"zoom_crop_{idx:02d}_panel", crop_for_model)
            try:
                raw = self._post_vision_chat(
                    prompt=crop_prompt,
                    image=crop_for_model,
                    is_kimi=is_kimi,
                max_side=_env_int("AUTOPDFTRANSLATOR_VISION_ZOOM_MAX_SIDE", 2600, 1200, 4200),
                max_bytes=vision_max_bytes,
                max_tokens=_env_int("AUTOPDFTRANSLATOR_VISION_ZOOM_MAX_TOKENS", 1600, 512, 4096),
                timeout_seconds=timeout_seconds,
            )
            except Exception as exc:
                if debug_dir is not None:
                    _save_debug_text(debug_dir, page_index, f"zoom_crop_{idx:02d}_request_error", str(exc))
                logging.info("Vision zoom crop %s request failed: %s", idx, exc)
                continue
            if debug_dir is not None:
                _save_debug_text(debug_dir, page_index, f"zoom_crop_{idx:02d}_raw", raw)
            try:
                parsed = json.loads(_extract_json_block(raw))
            except Exception:
                if debug_dir is not None:
                    _save_debug_text(debug_dir, page_index, f"zoom_crop_{idx:02d}_parse_error", raw)
                continue
            if debug_dir is not None:
                _save_debug_text(
                    debug_dir,
                    page_index,
                    f"zoom_crop_{idx:02d}_parsed",
                    json.dumps(parsed, ensure_ascii=False, indent=2),
                )
            crop_items = _parse_vision_items(parsed, page_index)
            for item in crop_items:
                item.metadata["vision_region"] = idx
            items.extend(crop_items)
        return _dedupe_comment_items(items)

    def _post_vision_chat(
        self,
        *,
        prompt: str,
        image: Any,
        is_kimi: bool,
        max_side: int,
        max_bytes: int,
        max_tokens: int,
        timeout_seconds: int,
    ) -> str:
        image_data, mime = _encode_image_for_vision(image, max_side=max_side, max_bytes=max_bytes)
        b64 = base64.b64encode(image_data).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        content_variants = [
            [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        ]
        if not is_kimi:
            content_variants.append(
                [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": data_url},
                ]
            )

        errors: list[str] = []
        for content in content_variants:
            try:
                return self._post_chat(
                    [
                        {"role": "system", "content": "You are a precise multimodal markup reader."},
                        {"role": "user", "content": content},
                    ],
                    temperature=0.0,
                    max_tokens=max_tokens,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as exc:
                errors.append(str(exc))
        raise RuntimeError("Vision chat request failed. " + " || ".join(errors[:2]))

    def _build_system_prompt(self, source_texts: list[str]) -> str:
        system_prompt = "You are a precise architectural review translator."
        additions: list[str] = []
        if self.prompt_hint:
            additions.append(self.prompt_hint)
        memory_hint = self._build_memory_hint(source_texts, include_fallback_terms=False)
        if memory_hint:
            additions.append(f"Translation memory:\n{memory_hint}")
        if additions:
            system_prompt = f"{system_prompt} Additional instruction:\n" + "\n\n".join(additions)
        return system_prompt

    def _build_memory_hint(self, source_texts: list[str], *, include_fallback_terms: bool) -> str:
        memory = self.translation_memory if isinstance(self.translation_memory, dict) else {}
        terms = memory.get("terms", [])
        style_rules = memory.get("style_rules", [])
        examples = memory.get("examples", [])
        if not isinstance(terms, list):
            terms = []
        if not isinstance(style_rules, list):
            style_rules = []
        if not isinstance(examples, list):
            examples = []

        haystack = "\n".join(source_texts).lower()
        selected_terms: list[dict[str, str]] = []
        for item in terms:
            if not isinstance(item, dict):
                continue
            source = str(item.get("source", "")).strip()
            target = str(item.get("target", "")).strip()
            if not source or not target:
                continue
            source_lower = source.lower()
            source_words = [
                word
                for word in re.findall(r"[a-zA-Z][a-zA-Z0-9\-]{2,}", source_lower)
                if len(word) >= 3
            ]
            is_relevant = bool(haystack) and (
                source_lower in haystack or any(word in haystack for word in source_words)
            )
            if is_relevant or (include_fallback_terms and len(selected_terms) < 30):
                selected_terms.append({"source": source, "target": target})
            if len(selected_terms) >= 40:
                break

        blocks: list[str] = []
        rules = [str(rule).strip() for rule in style_rules if str(rule).strip()][:10]
        if rules:
            blocks.append("Style rules:\n" + "\n".join(f"- {rule}" for rule in rules))
        if selected_terms:
            blocks.append(
                "Relevant terms:\n"
                + "\n".join(f"- {item['source']} => {item['target']}" for item in selected_terms)
            )
        compact_examples = [str(example).strip() for example in examples if str(example).strip()][:3]
        if compact_examples:
            blocks.append("Style examples:\n" + "\n".join(f"- {example[:240]}" for example in compact_examples))
        return "\n".join(blocks)

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers


def _extract_json_block(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _load_translation_memory_from_env() -> dict[str, Any]:
    raw = os.getenv("AUTOPDFTRANSLATOR_MEMORY_JSON", "").strip()
    if not raw:
        path_raw = os.getenv("AUTOPDFTRANSLATOR_MEMORY_JSON_PATH", "").strip()
        if path_raw:
            try:
                raw = Path(path_raw).read_text(encoding="utf-8")
            except Exception:
                raw = ""
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _normalize_bearer_api_key(value: str) -> str:
    key = str(value or "").strip()
    if key.lower().startswith("bearer "):
        key = key[7:].strip()
    return key


def _extract_http_error_detail(response: Any) -> str:
    text = ""
    try:
        payload = response.json()
        if isinstance(payload, dict):
            err = payload.get("error")
            if isinstance(err, dict):
                parts = [
                    str(err.get("message", "")).strip(),
                    str(err.get("type", "")).strip(),
                    str(err.get("code", "")).strip(),
                ]
                text = " | ".join(part for part in parts if part)
            else:
                text = json.dumps(payload, ensure_ascii=False)
    except Exception:
        text = ""
    if not text:
        body = str(getattr(response, "text", "") or "").strip()
        text = body[:1200] if body else "<empty response body>"
    return text


def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except Exception:
        return default
    return max(minimum, min(maximum, value))


def _extract_chat_output_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    first = choices[0]
    if not isinstance(first, dict):
        return ""
    message = first.get("message")
    if not isinstance(message, dict):
        return ""

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return _strip_reasoning_tags(content)
    if isinstance(content, dict):
        text_val = content.get("text")
        if isinstance(text_val, str):
            return _strip_reasoning_tags(text_val)
    if isinstance(content, list):
        chunks: list[str] = []
        for part in content:
            if isinstance(part, str):
                part_text = part.strip()
                if part_text:
                    chunks.append(part_text)
                continue
            if not isinstance(part, dict):
                continue
            text_val = part.get("text")
            if isinstance(text_val, str) and text_val.strip():
                chunks.append(text_val.strip())
                continue
            value_val = part.get("value")
            if isinstance(value_val, str) and value_val.strip():
                chunks.append(value_val.strip())
                continue
        return _strip_reasoning_tags("\n".join(chunks))

    # Some reasoning models put verbose thought in reasoning_content and can still
    # contain a valid JSON block there when content is empty.
    reasoning_content = message.get("reasoning_content")
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        candidate = _extract_first_json_array_block(reasoning_content)
        if candidate:
            return candidate
        translations = _extract_translation_lines_from_reasoning(reasoning_content)
        if translations:
            return json.dumps([{"translated": item} for item in translations], ensure_ascii=False)
    return ""


def _strip_reasoning_tags(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.IGNORECASE | re.DOTALL)
    return text.strip()


def _extract_first_json_array_block(text: str) -> str:
    s = text.strip()
    start = s.find("[")
    while start != -1:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    block = s[start : i + 1]
                    try:
                        obj = json.loads(block)
                    except Exception:
                        break
                    if isinstance(obj, list):
                        return block
                    break
        start = s.find("[", start + 1)
    return ""


def _extract_translation_lines_from_reasoning(text: str) -> list[str]:
    out: list[str] = []
    pattern = re.compile(
        r"(?:^|\n)\s*(?:[-*\d\.\)\s]*(?:Translation|Translated)\s*[:：]\s*)(.+?)(?=\n\s*(?:\d+[\.\)]|[-*])|\n\s*(?:Top|Bottom|Middle|Left|Right|Center|Centre)\b|\Z)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    for match in pattern.finditer(text):
        value = normalize_comment_text(match.group(1))
        value = re.sub(r"^(?:Yes|No)\s*[,.，。:：-]*\s*", "", value, flags=re.IGNORECASE).strip()
        if value and not re.search(r"[A-Za-z]{4,}", value):
            out.append(value)
    return out


def _extract_responses_output_text(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    # Some implementations provide direct output_text.
    direct = payload.get("output_text")
    if isinstance(direct, str) and direct.strip():
        return direct.strip()

    output = payload.get("output")
    if not isinstance(output, list):
        return ""

    chunks: list[str] = []
    for item in output:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict):
                continue
            text_value = part.get("text")
            if isinstance(text_value, str) and text_value.strip():
                chunks.append(text_value.strip())
    return "\n".join(chunks).strip()


def _encode_image_for_vision(image: Any, *, max_side: int = 1800, max_bytes: int = 4_000_000) -> tuple[bytes, str]:
    """
    Prepare image payload for broad OpenAI-compatible vision endpoints:
    - Downscale large pages to reduce payload / token pressure.
    - Use JPEG for much smaller body than PNG screenshots.
    """
    img = image.convert("RGB")
    width, height = img.size
    longest = max(width, height)
    if longest > max_side:
        scale = max_side / float(longest)
        new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        img = img.resize(new_size)

    quality_levels = [82, 72, 62]
    for quality in quality_levels:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality, optimize=True)
        data = buf.getvalue()
        if len(data) <= max_bytes:
            return data, "image/jpeg"

    # Final fallback: lowest quality output
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=55, optimize=True)
    return buf.getvalue(), "image/jpeg"


def _parse_vision_items(parsed: Any, page_index: int) -> list[CommentItem]:
    if isinstance(parsed, dict):
        parsed = [parsed]
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []

    items: list[CommentItem] = []
    for obj in parsed:
        item_metadata: dict[str, Any] = {}
        if isinstance(obj, str):
            translated = ""
            raw_original = obj
            original = normalize_comment_text(raw_original)
            confidence = 1.0
        elif isinstance(obj, dict):
            translated = normalize_comment_text(str(obj.get("translated", "")))
            raw_original = str(obj.get("original", ""))
            original = normalize_comment_text(raw_original)
            confidence = float(obj.get("confidence", 1.0))
            region_label = normalize_comment_text(str(obj.get("region", "")))
            if region_label:
                item_metadata["vision_tile_region"] = region_label
        else:
            continue

        if raw_original:
            item_metadata["vision_raw_original"] = raw_original
        original = _normalize_vision_original_text(original)
        text_value = translated or original
        if not text_value or _is_placeholder_translation(text_value):
            continue
        if _looks_like_uncertain_vision_guess(text_value):
            continue
        items.append(
            CommentItem(
                page_index=page_index,
                text_original=original or translated,
                text_translated=translated,
                source="vision",
                confidence=confidence,
                metadata=item_metadata,
            )
        )
    return items


def _looks_like_uncertain_vision_guess(text: str) -> bool:
    low = text.lower()
    uncertain_markers = [
        "something",
        "???",
        "[",
        "]",
        " or ",
        "maybe",
        "looks like",
        "seems",
        "not sure",
    ]
    return any(marker in low for marker in uncertain_markers)


def _looks_like_local_vision_misread(text: str) -> bool:
    normalized = normalize_comment_text(text).lower()
    return bool(
        re.search(r"\b\d{1,2}\s*/\s*p\b", normalized)
        or re.search(r"\b7[.,]0\s+stor(?:ey|e)y?s?\b", normalized)
    )


def _normalize_vision_original_text(text: str) -> str:
    if not text:
        return text
    normalized = text
    normalized = re.sub(r"\b(\d{1,2})\s*/\s*[Pp]\b", r"\1/F", normalized)
    normalized = re.sub(r"\b7[.,]0\s+stor(?:ey|e)y?s?\b", "20 storeys", normalized, flags=re.IGNORECASE)
    return normalize_comment_text(normalized)


def _parse_vision_regions(raw: str) -> list[tuple[float, float, float, float]]:
    try:
        parsed = json.loads(_extract_json_block(raw))
    except Exception:
        return []
    if (
        isinstance(parsed, list)
        and len(parsed) >= 4
        and all(isinstance(value, (int, float, str)) for value in parsed[:4])
    ):
        parsed = [parsed[:4]]
    if isinstance(parsed, dict):
        for key in ("regions", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break
    if not isinstance(parsed, list):
        return []

    regions: list[tuple[float, float, float, float]] = []
    for item in parsed:
        bbox = None
        if isinstance(item, dict):
            bbox = item.get("bbox_pct") or item.get("bbox") or item.get("box")
        elif isinstance(item, (list, tuple)):
            bbox = item
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        try:
            x0, y0, x1, y1 = [float(v) for v in bbox[:4]]
        except Exception:
            continue
        if max(x0, y0, x1, y1) <= 1.0:
            x0, y0, x1, y1 = x0 * 100.0, y0 * 100.0, x1 * 100.0, y1 * 100.0
        x0, x1 = sorted((max(0.0, min(100.0, x0)), max(0.0, min(100.0, x1))))
        y0, y1 = sorted((max(0.0, min(100.0, y0)), max(0.0, min(100.0, y1))))
        if (x1 - x0) < 2.0 or (y1 - y0) < 2.0:
            continue
        regions.append((x0, y0, x1, y1))
    return _merge_regions(regions)


def _merge_regions(regions: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    merged: list[tuple[float, float, float, float]] = []
    for region in sorted(regions, key=lambda r: (r[1], r[0])):
        x0, y0, x1, y1 = region
        merged_into_existing = False
        for idx, existing in enumerate(merged):
            ex0, ey0, ex1, ey1 = existing
            expanded = (ex0 - 3.0, ey0 - 3.0, ex1 + 3.0, ey1 + 3.0)
            if not (x1 < expanded[0] or x0 > expanded[2] or y1 < expanded[1] or y0 > expanded[3]):
                merged[idx] = (min(ex0, x0), min(ey0, y0), max(ex1, x1), max(ey1, y1))
                merged_into_existing = True
                break
        if not merged_into_existing:
            merged.append(region)
    return merged


def _cluster_nearby_regions(regions: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    clustered: list[tuple[float, float, float, float]] = []
    for region in sorted(regions, key=lambda r: (r[1], r[0])):
        x0, y0, x1, y1 = region
        merged_idx: int | None = None
        for idx, existing in enumerate(clustered):
            ex0, ey0, ex1, ey1 = existing
            ew = max(0.1, ex1 - ex0)
            eh = max(0.1, ey1 - ey0)
            rw = max(0.1, x1 - x0)
            rh = max(0.1, y1 - y0)
            expanded = (
                ex0 - max(1.1, ew * 0.45),
                ey0 - max(1.1, eh * 0.65),
                ex1 + max(1.1, ew * 0.55),
                ey1 + max(1.1, eh * 0.75),
            )
            close_enough = not (x1 < expanded[0] or x0 > expanded[2] or y1 < expanded[1] or y0 > expanded[3])
            combined_area = (max(ex1, x1) - min(ex0, x0)) * (max(ey1, y1) - min(ey0, y0))
            if close_enough and combined_area <= max(18.0, (ew * eh + rw * rh) * 8.0):
                merged_idx = idx
                break
        if merged_idx is None:
            clustered.append(region)
        else:
            ex0, ey0, ex1, ey1 = clustered[merged_idx]
            clustered[merged_idx] = (min(ex0, x0), min(ey0, y0), max(ex1, x1), max(ey1, y1))

    # Merge a few rounds because handwriting strokes may connect through intermediate words.
    for _ in range(2):
        before = len(clustered)
        clustered = _merge_regions(clustered)
        if len(clustered) == before:
            break
    return _dedupe_regions(clustered)


def _dedupe_regions(regions: list[tuple[float, float, float, float]]) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for region in sorted(regions, key=lambda r: ((r[2] - r[0]) * (r[3] - r[1])), reverse=True):
        if any(_regions_are_near_duplicates(region, existing) for existing in out):
            continue
        out.append(region)
    return out


def _regions_are_near_duplicates(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> bool:
    overlap = _region_overlap_ratio(a, b)
    if overlap <= 0.82:
        return False
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    area_a = max(0.01, (ax1 - ax0) * (ay1 - ay0))
    area_b = max(0.01, (bx1 - bx0) * (by1 - by0))
    ratio = area_a / area_b
    return 0.45 <= ratio <= 2.2


def _region_overlap_ratio(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    overlap_w = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    overlap_h = max(0.0, min(ay1, by1) - max(ay0, by0))
    overlap = overlap_w * overlap_h
    smaller = min(max(0.01, (ax1 - ax0) * (ay1 - ay0)), max(0.01, (bx1 - bx0) * (by1 - by0)))
    return overlap / smaller


def _crop_regions_with_padding(image: Any, regions: list[tuple[float, float, float, float]]) -> list[Any]:
    width, height = image.size
    crops: list[Any] = []
    for x0p, y0p, x1p, y1p in regions:
        pad_x = max(4.0, (x1p - x0p) * 0.28)
        pad_y = max(4.0, (y1p - y0p) * 0.35)
        x0 = int(max(0, (x0p - pad_x) / 100.0 * width))
        y0 = int(max(0, (y0p - pad_y) / 100.0 * height))
        x1 = int(min(width, (x1p + pad_x) / 100.0 * width))
        y1 = int(min(height, (y1p + pad_y) / 100.0 * height))
        if x1 - x0 < 24 or y1 - y0 < 24:
            continue
        crops.append(image.crop((x0, y0, x1, y1)))
    return crops


def _make_handwriting_enhanced_panel(image: Any) -> Any:
    try:
        from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    except Exception:
        return image

    original = image.convert("RGB")
    grayscale = ImageOps.grayscale(original)
    enhanced = ImageOps.autocontrast(grayscale, cutoff=1)
    enhanced = ImageEnhance.Contrast(enhanced).enhance(2.8)
    enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.0)
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.4, percent=180, threshold=3))
    enhanced_rgb = ImageOps.colorize(enhanced, black="#111111", white="#ffffff")

    max_panel_w = 3000
    scale = min(1.0, max_panel_w / max(original.width * 2, 1))
    if scale < 1.0:
        new_size = (max(1, int(original.width * scale)), max(1, int(original.height * scale)))
        original = original.resize(new_size)
        enhanced_rgb = enhanced_rgb.resize(new_size)

    gap = 16
    panel = Image.new("RGB", (original.width * 2 + gap, original.height), "white")
    panel.paste(original, (0, 0))
    panel.paste(enhanced_rgb, (original.width + gap, 0))
    return panel


def _make_roi_contact_sheet(crops: list[Any]) -> Any:
    try:
        from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
    except Exception:
        return crops[0] if crops else None

    tile_w = 760
    label_h = 34
    gap = 18
    tiles: list[Any] = []
    for idx, crop in enumerate(crops, start=1):
        original = crop.convert("RGB")
        scale = min(1.0, tile_w / max(original.width, 1))
        if scale < 1.0:
            original = original.resize((max(1, int(original.width * scale)), max(1, int(original.height * scale))))

        grayscale = ImageOps.grayscale(original)
        enhanced = ImageOps.autocontrast(grayscale, cutoff=1)
        enhanced = ImageEnhance.Contrast(enhanced).enhance(3.2)
        enhanced = ImageEnhance.Sharpness(enhanced).enhance(2.2)
        enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1.2, percent=190, threshold=2))
        enhanced_rgb = ImageOps.colorize(enhanced, black="#111111", white="#ffffff")

        tile_h = label_h + original.height + gap + enhanced_rgb.height
        tile = Image.new("RGB", (tile_w, tile_h), "white")
        draw = ImageDraw.Draw(tile)
        label = f"R{idx}"
        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except Exception:
            font = ImageFont.load_default()
        draw.rectangle((0, 0, tile_w, label_h), fill="#0f172a")
        draw.text((12, 6), label, fill="white", font=font)
        tile.paste(original, ((tile_w - original.width) // 2, label_h))
        tile.paste(enhanced_rgb, ((tile_w - enhanced_rgb.width) // 2, label_h + original.height + gap))
        tiles.append(tile)

    columns = 3 if len(tiles) > 10 else (2 if len(tiles) > 1 else 1)
    rows = (len(tiles) + columns - 1) // columns
    row_heights = [0] * rows
    for idx, tile in enumerate(tiles):
        row_heights[idx // columns] = max(row_heights[idx // columns], tile.height)
    sheet_w = columns * tile_w + (columns + 1) * gap
    sheet_h = sum(row_heights) + (rows + 1) * gap
    sheet = Image.new("RGB", (sheet_w, sheet_h), "#f8fafc")

    y = gap
    tile_idx = 0
    for row in range(rows):
        x = gap
        for _ in range(columns):
            if tile_idx >= len(tiles):
                break
            tile = tiles[tile_idx]
            sheet.paste(tile, (x, y))
            x += tile_w + gap
            tile_idx += 1
        y += row_heights[row] + gap
    return sheet


def _detect_dark_markup_regions(image: Any) -> list[tuple[float, float, float, float]]:
    try:
        from PIL import ImageFilter, ImageOps
    except Exception:
        return []

    grayscale = ImageOps.grayscale(image.convert("RGB"))
    grayscale.thumbnail((900, 900))
    grayscale = ImageOps.autocontrast(grayscale, cutoff=1)
    binary = grayscale.point(lambda p: 255 if p < 120 else 0)
    binary = binary.filter(ImageFilter.MaxFilter(3))
    width, height = binary.size
    pixels = binary.load()
    seen: set[tuple[int, int]] = set()
    regions: list[tuple[float, float, float, float]] = []

    for y in range(height):
        for x in range(width):
            if pixels[x, y] == 0 or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            xs: list[int] = []
            ys: list[int] = []
            while stack:
                cx, cy = stack.pop()
                xs.append(cx)
                ys.append(cy)
                for nx in (cx - 1, cx, cx + 1):
                    for ny in (cy - 1, cy, cy + 1):
                        if nx < 0 or ny < 0 or nx >= width or ny >= height:
                            continue
                        if (nx, ny) in seen or pixels[nx, ny] == 0:
                            continue
                        seen.add((nx, ny))
                        stack.append((nx, ny))

            area = len(xs)
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            box_w = x1 - x0
            box_h = y1 - y0
            if area < 20 or box_w < 4 or box_h < 3:
                continue
            x0p = x0 / width * 100.0
            y0p = y0 / height * 100.0
            x1p = x1 / width * 100.0
            y1p = y1 / height * 100.0
            box_area_ratio = ((x1 - x0) * (y1 - y0)) / max(1, width * height)
            density = area / max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
            if box_area_ratio > 0.08 or density > 0.92:
                continue
            # Drop likely page numbers / title-block fragments in the lower-right corner.
            if x0p > 78 and y0p > 78 and box_w < width * 0.15 and box_h < height * 0.16:
                continue
            regions.append((x0p, y0p, x1p, y1p))

    regions = _cluster_nearby_regions(regions)
    regions = sorted(regions, key=lambda r: (r[1], r[0]))
    return regions[:18]


def _dedupe_comment_items(items: list[CommentItem]) -> list[CommentItem]:
    out: list[CommentItem] = []
    seen: set[str] = set()
    seen_locations: dict[str, set[str]] = {}
    for item in items:
        key = normalize_comment_text(item.text_translated or item.text_original).lower()
        key = re.sub(r"\s+", " ", key)
        if not key:
            continue
        location = _item_location_key(item)
        if key in seen:
            known_locations = seen_locations.get(key, set())
            if location and location not in known_locations and (
                _is_repeatable_markup_label(key) or _is_short_location_specific_markup_label(key)
            ):
                pass
            elif not location and _is_repeatable_markup_label(key):
                pass
            else:
                continue
        if any(
            key in prev or prev in key
            for prev in seen
            if min(len(key), len(prev)) >= 24 and _looks_like_same_comment_family(key, prev)
        ):
            continue
        seen.add(key)
        if location:
            seen_locations.setdefault(key, set()).add(location)
        out.append(item)
    return out


def _looks_like_same_comment_family(a: str, b: str) -> bool:
    if a == b:
        return True
    protected_markers = ("same", "total", "behind", "add", "remove", "recess", "recessed")
    if any(marker in a or marker in b for marker in protected_markers):
        return False
    return True


def _is_repeatable_markup_label(text: str) -> bool:
    if len(text) > 36:
        return False
    patterns = [
        r"\b\d+\s*/?\s*f\b",
        r"\b\d+\s*storey(?:s)?\b",
        r"\bremove\s+building\b",
        r"\badd\s+openings?\b",
        r"\brecess(?:ed)?\b",
        r"\badd\s+pvs?\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _is_short_location_specific_markup_label(text: str) -> bool:
    if len(text) > 42:
        return False
    patterns = [
        r"\bsee\s+model\b",
        r"\bpeople\s*&\s*trees\b",
        r"\badd\s+people\s*&\s*activity\b",
        r"\bstreet\s+lamps?\b",
        r"\bdriveway\b",
        r"\bsame\s+diagrid\b",
    ]
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns)


def _item_location_key(item: CommentItem) -> str:
    metadata = item.metadata or {}
    parts = [
        str(metadata.get("vision_region", "")),
        str(metadata.get("vision_tile_region", "")),
    ]
    return ":".join(part for part in parts if part)


def _recover_original_items_from_error(error_text: str, page_index: int) -> list[CommentItem]:
    reasoning_chunks = re.findall(r"reasoning_content['\"]?\s*:\s*['\"](.+?)(?:['\"],\s*['\"](?:audio|function_call|tool_calls|refusal|annotations|finish_reason)|['\"]}\])", error_text, flags=re.DOTALL)
    haystack = "\n".join(reasoning_chunks) if reasoning_chunks else error_text
    candidates: list[str] = []
    patterns = [
        r'"([^"]{3,90})"',
        r"'([^']{3,90})'",
        r"(?:it (?:says|reads|looks like|seems to say)\s+)([^.\n;]{3,90})",
        r"(?:text(?: saying)?[:：]\s*)([^.\n;]{3,90})",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, haystack, flags=re.IGNORECASE):
            value = normalize_comment_text(match.group(1))
            if _looks_like_reasoning_comment_candidate(value):
                candidates.append(value)

    out: list[CommentItem] = []
    seen: set[str] = set()
    for value in candidates:
        key = value.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            CommentItem(
                page_index=page_index,
                text_original=value,
                text_translated="",
                source="vision",
                confidence=0.45,
                metadata={"recovery": "reasoning_content"},
            )
        )
    return out


def _looks_like_reasoning_comment_candidate(value: str) -> bool:
    if not value or len(value) < 3 or len(value) > 90:
        return False
    low = value.lower()
    reject_patterns = [
        r"^translated$",
        r"^original$",
        r"^content$",
        r"^senate",
        r"hawkins\\brown",
        r"^\d+$",
        r"^\d{6}_",
        r"chat completion",
        r"strict json",
        r"chatcmpl",
        r"^object$",
        r"^created$",
        r"^model$",
        r"^choices$",
        r"^index$",
        r"^message$",
        r"^role$",
        r"^assistant$",
        r"^kimi",
        r"^finish_reason",
        r"^payload",
    ]
    if any(re.search(pattern, low) for pattern in reject_patterns):
        return False
    if "reasoning_content" in low or "chat.completion" in low:
        return False
    if " actually " in low or " or similar" in low or " let me " in low:
        return False
    return bool(re.search(r"[A-Za-z]", value))


def _vision_debug_dir() -> Path | None:
    raw = os.getenv("AUTOPDFTRANSLATOR_VISION_DEBUG_DIR", "").strip()
    if not raw:
        return None
    path = Path(raw)
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception:
        return None
    return path


def _save_debug_image(debug_dir: Path, page_index: int, name: str, image: Any) -> None:
    try:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "image"
        path = debug_dir / f"page_{page_index + 1:03d}_{safe_name}.jpg"
        image.convert("RGB").save(path, format="JPEG", quality=92)
    except Exception:
        logging.debug("Could not save vision debug image", exc_info=True)


def _save_debug_text(debug_dir: Path, page_index: int, name: str, text: str) -> None:
    try:
        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_") or "text"
        path = debug_dir / f"page_{page_index + 1:03d}_{safe_name}.txt"
        path.write_text(str(text or ""), encoding="utf-8")
    except Exception:
        logging.debug("Could not save vision debug text", exc_info=True)


def _is_placeholder_translation(text: str) -> bool:
    normalized = normalize_comment_text(text)
    if not normalized:
        return True
    return bool(re.fullmatch(r"[\.\。…\s]+", normalized))


def _parse_translation_batch_json(raw: str) -> list[str]:
    block = _extract_json_block(raw)
    try:
        parsed = json.loads(block)
    except Exception:
        try:
            parsed = json.loads(_normalize_jsonish_quotes(block))
        except Exception:
            return []

    if isinstance(parsed, dict):
        for key in ("translations", "translated", "items", "data"):
            value = parsed.get(key)
            if isinstance(value, list):
                parsed = value
                break

    if not isinstance(parsed, list):
        return []

    out: list[str] = []
    for item in parsed:
        if isinstance(item, str):
            out.append(item)
            continue
        if isinstance(item, dict):
            text = item.get("translated")
            if isinstance(text, str):
                out.append(text)
                continue
            text = item.get("text")
            if isinstance(text, str):
                out.append(text)
                continue
        out.append("")
    return out


def _normalize_jsonish_quotes(text: str) -> str:
    replacements = {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u2033": '"',
        "\u301d": '"',
        "\u301e": '"',
        "\u301f": '"',
        "\u2018": "'",
        "\u2019": "'",
    }
    return "".join(replacements.get(ch, ch) for ch in text)


def _parse_numbered_translations(raw: str, expected_count: int) -> list[str]:
    text = _strip_reasoning_tags(raw)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    indexed: dict[int, str] = {}
    for line in lines:
        match = re.match(r"^\s*(?:\[|\()?(\d{1,4})(?:\]|\)|\.|、|：|:)\s*(.+?)\s*$", line)
        if not match:
            continue
        idx = int(match.group(1))
        value = normalize_comment_text(match.group(2))
        if 1 <= idx <= expected_count and value:
            indexed[idx] = value
    if len(indexed) == expected_count:
        return [indexed[i] for i in range(1, expected_count + 1)]

    # Some local models keep everything on one line. Split before numbered markers.
    compact = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"\s*(?=(?:\[|\()?(\d{1,4})(?:\]|\)|\.|、|：|:)\s*)", compact)
    indexed.clear()
    i = 1
    while i + 1 < len(parts):
        try:
            idx = int(parts[i])
        except Exception:
            i += 2
            continue
        value = normalize_comment_text(re.sub(r"^(?:\]|\)|\.|、|：|:)\s*", "", parts[i + 1]))
        value = re.sub(r"\s*(?=(?:\[|\()?\d{1,4}(?:\]|\)|\.|、|：|:)\s*)$", "", value).strip()
        if 1 <= idx <= expected_count and value:
            indexed[idx] = value
        i += 2
    if len(indexed) == expected_count:
        return [indexed[i] for i in range(1, expected_count + 1)]
    return []


def _build_translategemma_prompt(
    text: str,
    source_lang: str,
    target_lang: str,
    *,
    prompt_hint: str = "",
    memory_hint: str = "",
) -> str:
    source_name, source_code = _language_name_and_code(source_lang, fallback_name="English")
    target_name, target_code = _language_name_and_code(target_lang, fallback_name="Chinese (Simplified)")
    domain_instruction = _build_translategemma_domain_instruction(prompt_hint=prompt_hint, memory_hint=memory_hint)
    return (
        f"You are a professional {source_name} ({source_code}) to {target_name} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"{domain_instruction}\n"
        f"Produce only the {target_name} translation, without any additional explanations or commentary. "
        f"Please translate the following {source_name} text into {target_name}:\n\n\n"
        f"{text}"
    )


def _build_translategemma_batch_prompt(
    texts: list[str],
    source_lang: str,
    target_lang: str,
    *,
    prompt_hint: str = "",
    memory_hint: str = "",
) -> str:
    source_name, source_code = _language_name_and_code(source_lang, fallback_name="English")
    target_name, target_code = _language_name_and_code(target_lang, fallback_name="Chinese (Simplified)")
    domain_instruction = _build_translategemma_domain_instruction(prompt_hint=prompt_hint, memory_hint=memory_hint)
    numbered = "\n".join(f"[{idx}] {text}" for idx, text in enumerate(texts, start=1))
    return (
        f"You are a professional {source_name} ({source_code}) to {target_name} ({target_code}) translator. "
        f"Your goal is to accurately convey the meaning and nuances of the original {source_name} text while adhering to {target_name} grammar, vocabulary, and cultural sensitivities.\n"
        f"{domain_instruction}\n"
        f"Produce only the {target_name} translation, without any additional explanations or commentary. "
        "Translate each numbered source item separately. Keep exactly the same numbering and item count. "
        "Do not merge, split, omit, reorder, explain, or add content. "
        f"Please translate the following {source_name} text into {target_name}:\n\n\n"
        f"{numbered}"
    )


def _build_translategemma_domain_instruction(*, prompt_hint: str = "", memory_hint: str = "") -> str:
    sections = [
        "You are translating concise architectural visualization review comments and PDF markup notes. "
        "Use natural Chinese wording used by architects and visualization reviewers: compact, directive, and specific. "
        "Prefer '需/应/可/不宜' review-note wording over verbose machine-translation prose. "
        "Do not expand, soften, summarize, explain, or invent missing details. Preserve negation and criticism exactly. "
        "Preserve proper nouns, project/site/building names, brand names, and signage text unless they are common architectural terms.",
        "Domain terminology: storeys/F = 层; PVs = 光伏板; DIAGRID = 斜肋结构; recess/recessed = 退台/退界/凹进; podium = 裙楼; decking/deck = 甲板/平台铺装; boardwalk = 木栈道; bollard lighting = 矮柱灯; up-light = 上照灯; pool edge = 泳池边缘; water feature = 水景; signage = 标识; wakes = 船尾浪/尾流, not wind; character = 特色/辨识度, not text; MP = 总图/总平面; dusk view may be misspelled as dust view in markups and means 黄昏视图/暮色视图 unless the sentence clearly describes dust particles.",
        "Style examples: 'People look fake. To be more realistic' -> '人物看起来假，需更真实'; 'Pool bar seating and counter top to be shown' -> '需显示泳池吧座位和台面'; 'No lighting reflection nor diffused light emission ... looks unnatural' -> '缺少灯光反射或漫射发光，显得不自然'.",
    ]
    if memory_hint:
        sections.append(f"Project glossary:\n{memory_hint}")
    if prompt_hint:
        sections.append(f"Additional project instruction:\n{prompt_hint}")
    return "\n".join(sections)


def _polish_archviz_translation(source: str, translated: str, target_lang: str) -> str:
    out = normalize_comment_text(translated)
    if not out or not (target_lang or "").lower().startswith("zh"):
        return out
    low = (source or "").lower()
    if "wakes" in low:
        if "too strong" in low and "speed" in low:
            return "船尾浪相对于速度过于强烈"
        out = re.sub(r"风力|风浪|尾波", "船尾浪", out)
        out = out.replace("与速度不匹配", "相对于速度过于强烈")
    if "does not have character" in low:
        out = re.sub(r"这张(?:图片|图像|图)中?没有文字。?", "这张图缺少特色。", out)
        out = out.replace("没有字符", "缺少特色").replace("没有特点", "缺少特色")
    if "dust view" in low or "dusk view" in low:
        out = re.sub(r"灰尘(?:环境|视图|角度|场景)|尘土(?:环境|视图|场景)|尘埃(?:环境|视图|场景)", "黄昏视图", out)
        out = out.replace("从灰尘的角度来看", "作为黄昏视图")
        out = out.replace("从尘土的角度来看", "作为黄昏视图")
        out = out.replace("从黄昏视图来看", "作为黄昏视图")
    if "no lighting reflection" in low and "looks unnatural" in low:
        if "没有出现不自然" in out or "没有不自然" in out:
            out = "明亮窗户周围的景观中缺少灯光反射或漫射发光，显得不自然"
    if "pool bar seating" in low and "counter" in low and "show" in low:
        out = "需显示泳池吧座位和台面"
    if "people look fake" in low:
        out = re.sub(r"效果不真实。?为了更逼真。?", "人物看起来假，需更真实", out)
        out = out.replace("人看起来不真实。为了更逼真。", "人物看起来假，需更真实")
    if "bollard lighting" in low:
        out = out.replace("路灯", "矮柱灯")
    if "up-light" in low or "uplight" in low:
        out = out.replace("向上照明", "上照灯").replace("向上灯光", "上照灯")
    if "to be high lighted by the interior light" in low or "to be highlighted by the interior light" in low:
        out = "需由室内灯光突出照亮"
    return normalize_comment_text(out)


def _language_name_and_code(code: str, *, fallback_name: str) -> tuple[str, str]:
    normalized = (code or "").strip()
    mapping = {
        "auto": ("English", "en"),
        "en": ("English", "en"),
        "zh-CN": ("Chinese (Simplified)", "zh-CN"),
        "zh-TW": ("Chinese (Traditional)", "zh-TW"),
        "ja": ("Japanese", "ja"),
        "ko": ("Korean", "ko"),
        "fr": ("French", "fr"),
        "de": ("German", "de"),
        "es": ("Spanish", "es"),
    }
    if normalized in mapping:
        return mapping[normalized]
    if normalized.lower() in mapping:
        return mapping[normalized.lower()]
    return fallback_name, normalized or fallback_name
