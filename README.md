# AutoPDFTranslator 0.3.0

AutoPDFTranslator is a Streamlit tool for translating English PDF review comments into Chinese while preserving the original PDF pages. It extracts review comments, translates them with an API provider or local model, and writes editable FreeText annotations back into a new PDF.

The current version is optimized for architectural visualization review PDFs, including typed PDF comments, image-based review sheets, and handwritten markups.

## Features

- Batch upload and translate multiple PDFs.
- PDF type detection after upload, with a recommended translation mode.
- Three translation modes:
  - `text extraction only`: fastest mode; uses extractable PDF text / FreeText comments.
  - `hybrid`: uses text extraction first, then vision fallback when needed.
  - `vision only`: uses visual recognition on every page. API Providers are recommended for this mode.
- API Providers and Local Models configuration.
- OpenAI-compatible API support, including OpenAI, Moonshot/Kimi, Gemini OpenAI-compatible endpoints, and Ollama-compatible local endpoints.
- Local TranslateGemma prompt preset for architectural visualization comment translation.
- Architecture/visualization terminology polishing after translation to reduce common mistranslations.
- Staged progress UI:
  - comment extraction
  - AI translation
  - PDF writing
- Stop button for cooperative cancellation during long runs.
- Partial result preservation: if translation fails midway, the current translated PDF result is still kept when possible.
- Translation history with provider/model information.
- Editable annotation output with Bluebeam-style font size range from 2 to 72.
- Optional vision debug output for troubleshooting handwritten recognition.

## Project Structure

```text
AutoPDFTranslator/
├─ app.py
├─ desktop_app.py
├─ run_desktop.bat
├─ autopdftranslator/
│  ├─ __init__.py
│  ├─ main.py
│  ├─ pipeline.py
│  ├─ writer.py
│  ├─ extractors.py
│  ├─ providers/
│  │  ├─ __init__.py
│  │  ├─ mock.py
│  │  └─ openai_compatible.py
│  └─ models.py
├─ requirements.txt
├─ 使用说明.txt
└─ README.md
```

## Install

```bash
pip install -r requirements.txt
```

Python 3.10 or later is recommended.

## Run Streamlit UI

```bash
streamlit run app.py
```

## Run Desktop App

```bash
python desktop_app.py
```

Windows quick start:

```bash
run_desktop.bat
```

## Build Windows Installer

The recommended Windows packaging route is PyInstaller `onedir` plus Inno Setup.

Build the executable folder:

```powershell
.\packaging\build_exe.ps1 -SkipInstaller
```

Build the executable folder and installer:

```powershell
.\packaging\build_exe.ps1
```

Optional clean rebuild:

```powershell
.\packaging\build_exe.ps1 -Clean
```

Requirements:

- Windows
- Python installed
- Inno Setup 6 installed if you want the `.exe` installer

Build outputs:

- `dist\AutoPDFTranslator\AutoPDFTranslator.exe`
- `installer\AutoPDFTranslator-Setup-0.3.0.exe`

Packaged desktop mode stores user settings, translation history, corpus, memory, and output PDFs under:

```text
%LOCALAPPDATA%\AutoPDFTranslator
```

API keys are not bundled into the installer. Users should enter provider settings in the app UI after installation.

## Basic Workflow

1. Upload one or more PDF files.
2. Review the PDF type detection result and recommended mode.
3. Choose source and target language.
4. Choose translation mode:
   - Use `text extraction only` for native PDF comments or clean text-layer PDFs.
   - Use `hybrid` for mixed PDFs and most production workflows.
   - Use `vision only` for scanned/image/handwritten review sheets. API Providers are recommended.
5. Configure the provider in `Advanced`.
6. Click `Analyze / Estimate` if you want an extraction preview and cost estimate.
7. Click `Translate`.
8. Download or use the auto-saved translated PDF.

## Provider Configuration

Settings are split into two groups:

- `API Providers`: online OpenAI-compatible services.
- `Local Models`: local OpenAI-compatible services such as Ollama.

Common fields:

- `Base URL`
- `Model`
- `API key`
- `Prompt hint`

Use `Check connection` to verify the selected provider.

Provider settings are persisted to:

```text
.autopdftranslator_ui_config.json
```

## Translation Modes

### text extraction only

Uses typed PDF text and FreeText annotations. This is fastest and cheapest, but it cannot read image-only handwritten notes.

### hybrid

Recommended default for uncertain PDFs. The pipeline extracts text first and uses vision fallback when a page looks image-heavy, low-text, or likely contains handwritten comments.

### vision only

Runs visual recognition for every page. This is useful for scanned drawings and handwritten markups. API Providers are recommended because local vision models may be slower, less stable, or less accurate for handwriting.

## Local Models and TranslateGemma

Ollama local endpoints are supported through OpenAI-compatible chat completions. For example:

```text
Base URL: http://localhost:11434
Model: translategemma:latest
```

TranslateGemma uses a dedicated translation prompt based on its official prompt guide, with added architectural visualization context. Text translation is batched to improve consistency across related comments.

Notes:

- Local models may not reliably support vision even if the model name suggests it.
- Local vision recognition can be CPU/GPU intensive.
- For handwriting-heavy PDFs, Kimi/Gemini/OpenAI vision-capable API providers usually perform better.

## Progress and Cancellation

The progress panel shows:

- current stage
- pages translated / remaining
- pages written / remaining
- overall percentage

The `终止` button stops the current job cooperatively. A single blocking model request may need to return before the stop takes effect.

If a job fails or is cancelled, the app keeps partial results where possible instead of discarding the whole run.

## Output

The translated PDF keeps the original pages and adds editable FreeText annotations.

Annotation options include:

- font size: 2-72
- placement: top-left stacked or top-right stacked
- vision DPI
- max pages per PDF

The auto-save folder can be set in the UI. Results are also available in the result area after translation.

## Translation History

The history panel records:

- translation date
- source filename
- translated filename
- mode
- provider group
- provider name
- model
- duration
- result status

Failed runs are marked with the reason, such as no comments detected or partial translation failure.

## Useful Environment Variables

```bash
set AUTOPDFTRANSLATOR_TEXT_BATCH_SIZE=5000
set AUTOPDFTRANSLATOR_TRANSLATEGEMMA_BATCH_SIZE=40
set AUTOPDFTRANSLATOR_VISION_DEBUG_DIR=C:\tmp\autopdftranslator_vision_debug
set AUTOPDFTRANSLATOR_VISION_TIMEOUT_SECONDS=180
set AUTOPDFTRANSLATOR_VISION_MAX_TOKENS=4096
set AUTOPDFTRANSLATOR_VISION_MAX_SIDE=2600
```

For local vision troubleshooting, enable `AUTOPDFTRANSLATOR_VISION_DEBUG_DIR` and inspect the generated page images, ROI panels, and raw model responses.

## CLI

```bash
python -m autopdftranslator input.pdf output.pdf --target-lang zh-CN --vision auto
```

Vision options:

- `never`: text extraction only
- `auto`: hybrid
- `always`: vision only

## Optional ODL Fallback Extractor

For complex canvases such as Miro exports, an experimental OpenDataLoader fallback can be enabled.

Install optional dependency:

```bash
pip install opendataloader-pdf
```

Runtime requirements:

- Java 11+

Environment flag:

```bash
set AUTOPDFTRANSLATOR_ODL_FALLBACK=1
```
