from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

from .models import LayoutConfig
from .pipeline import translate_pdf


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autopdftranslator", description="Hybrid PDF review-comment translator")
    parser.add_argument("input_pdf", type=Path)
    parser.add_argument("output_pdf", type=Path)
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--target-lang", default="zh-CN")
    parser.add_argument("--vision", dest="vision_mode", choices=["never", "auto", "always"], default="auto")
    parser.add_argument("--provider", default="mock", help="mock | openai-compatible")
    parser.add_argument("--typed-text-threshold", type=int, default=120)
    parser.add_argument("--dpi-for-vision", type=int, default=220)
    parser.add_argument("--dump-json", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-pages", type=int)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    if not args.input_pdf.exists():
        logging.error("Input PDF does not exist: %s", args.input_pdf)
        return 2

    try:
        results = translate_pdf(
            input_pdf=args.input_pdf,
            output_pdf=args.output_pdf,
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            mode=args.vision_mode,
            typed_text_threshold=args.typed_text_threshold,
            provider=args.provider,
            dpi_for_vision=args.dpi_for_vision,
            dump_json=args.dump_json,
            dry_run=args.dry_run,
            max_pages=args.max_pages,
            verbose=args.verbose,
            layout=LayoutConfig(),
        )
    except Exception as exc:
        logging.error("Pipeline failed: %s", exc)
        return 2

    if args.dump_json:
        logging.info("Wrote extraction / translation JSON: %s", args.dump_json)
    if args.dry_run:
        logging.info("Dry run complete. No PDF written.")
    else:
        logging.info("Wrote translated PDF: %s", args.output_pdf)
    logging.info("Total extracted comments: %s", sum(len(page.comments) for page in results))
    return 0
