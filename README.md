# Contract PDF Parser CLI

Simple CLI to parse a single contract PDF into structured JSON with sections, clauses, labels, and effective date.

## Quick start

1) Create a virtual environment (optional but recommended)

```
python -m venv .venv
./.venv/Scripts/Activate.ps1  # PowerShell on Windows
```

2) Install dependencies

```
pip install -r requirements.txt
```

3) Run

```
python sanika_choudhary.py <path-to.pdf> <output.json>
```

## OCR fallback (optional)

If a PDF is scanned and contains no extractable text, the tool tries OCR via Tesseract.

You will need:

- Tesseract OCR installed and on PATH
- Poppler (for pdf2image) installed

Windows notes:

- Tesseract: download installer from the official repo and add the install folder to PATH
- Poppler: download a Windows build (e.g., `poppler` for Windows), add its `bin/` folder to PATH

If OCR dependencies are missing, the tool still emits valid best-effort JSON.

## Optional LLM assist (offline via Ollama)

This tool can optionally refine the parsed JSON using a local Ollama model. It is disabled by default and the parser works fully offline with no LLM.

Enable with environment variables:

```
# PowerShell examples
$Env:OLLAMA_ENABLE="1"
$Env:OLLAMA_MODEL="llama3"           # default
$Env:OLLAMA_URL="http://localhost:11434/api/chat"  # default
$Env:OLLAMA_TIMEOUT="25"             # seconds
$Env:OLLAMA_CONTEXT_PAGES="6"        # how many pages to send (default 6)
$Env:OLLAMA_CONTEXT_CHARS="30000"     # character cap for context (default 30000)
```

Run Ollama locally (`https://ollama.com`) and pull a model, e.g.:

```
ollama pull llama3
```

If Ollama is not running or the env flag is not set, the parser simply skips this step and still emits valid JSON.

## Output schema (exact)

This tool normalizes whitespace, preserves punctuation, and produces deterministic ordering.


