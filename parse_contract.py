"""
Author: Sanika Choudhary

How to run (CLI):
  python sanika_choudhary.py <input.pdf> <output.json>

What this does:
  - Reads a contract PDF and produces a clean, deterministic JSON with
    title, contract_type, effective_date, and a list of sections → clauses.
  - Tries text extraction first (pdfminer). If that fails (e.g., scanned PDFs),
    it falls back to OCR (Tesseract + pdf2image) when available.
  - Always returns valid JSON. If something goes wrong, it emits a small but
    schema-correct JSON so downstream systems keep working.

Optional local LLM assist (Ollama):
  - Disabled by default. Turn on with OLLAMA_ENABLE=1 to lightly refine titles
    and stitch broken sentences without inventing content.

Performance notes:
  - Designed to be snappy on typical 5–30 page PDFs. OCR and LLM can be slower; both are optional.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Optional, Sequence, Tuple, Dict, Any

# Optional text-fixing (mojibake cleanup)
try:
    import ftfy  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    ftfy = None


# ===============
# Utilities
# ===============


def fix_mojibake(text: str) -> str:
    """Fix common encoding artifacts (e.g., â€™ → ’) using ftfy when available."""
    if not text:
        return text
    if ftfy is None:
        return text
    try:
        return ftfy.fix_text(text)
    except Exception:
        return text


def normalize_whitespace(text: str) -> str:
    """Fix encoding artifacts, collapse internal whitespace to single spaces, trim edges."""
    # First, try to fix encoding artifacts
    text = fix_mojibake(text)
    # Replace all whitespace (including newlines and tabs) with spaces; then collapse.
    collapsed = re.sub(r"\s+", " ", text, flags=re.UNICODE)
    return collapsed.strip()


def safe_read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


# ===============
# Optional Ollama LLM refinement (graceful + opt-in)
# ===============


def _env_truthy(name: str, default: str = "0") -> bool:
    val = os.environ.get(name, default).strip().lower()
    return val in {"1", "true", "yes", "on"}


def maybe_refine_with_ollama(pages: Sequence[str], draft_output: Dict[str, Any]) -> Dict[str, Any]:
    """Optionally refine the draft JSON using a local Ollama model.

    Controlled via env:
      OLLAMA_ENABLE: set to 1/true to enable; disabled by default.
      OLLAMA_MODEL:  model name (default: "llama3").
      OLLAMA_URL:    chat endpoint (default: http://localhost:11434/api/chat)
      OLLAMA_TIMEOUT: seconds (default: 25)

    Always returns a dict. On any failure, returns the original draft_output.
    """
    if not _env_truthy("OLLAMA_ENABLE", "0"):
        return draft_output

    model = os.environ.get("OLLAMA_MODEL", "llama3").strip() or "llama3"
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/chat").strip()
    try:
        timeout = float(os.environ.get("OLLAMA_TIMEOUT", "25").strip())
    except Exception:
        timeout = 25.0

    # Control context size via env: number of pages and max chars
    try:
        ctx_pages = int(os.environ.get("OLLAMA_CONTEXT_PAGES", "6").strip())
    except Exception:
        ctx_pages = 6
    try:
        ctx_chars = int(os.environ.get("OLLAMA_CONTEXT_CHARS", "30000").strip())
    except Exception:
        ctx_chars = 30000

    pages_excerpt = "\n\n".join(pages[: max(1, ctx_pages)])
    if len(pages_excerpt) > ctx_chars:
        pages_excerpt = pages_excerpt[:ctx_chars]

    system_prompt = (
        "You are a contract structure normalizer. You will improve a draft parse of a contract into "
        "a very strict JSON schema with sections and clauses. Do not invent content. Only refine titles, "
        "fix clause stitching when a sentence is broken across lines, and preserve ordering. Keep labels when present, "
        "otherwise use empty string. Use null for section numbers that are absent. Keep clause index 0-based per section. "
        "Normalize whitespace to single spaces and preserve punctuation. Return only JSON in the exact same schema."
    )
    user_prompt = (
        "Source excerpt (first pages):\n" + pages_excerpt +
        "\n\nDraft JSON to refine (keep schema and keys exactly):\n" + json.dumps(draft_output, ensure_ascii=False)
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # Ask for concise deterministic output
        "options": {"temperature": 0.0},
        # Some Ollama models support JSON formatting; if unsupported, we'll parse best-effort.
        "format": "json",
        "stream": False,
    }

    try:
        # Prefer requests if available; otherwise fall back to urllib
        try:
            import requests  # type: ignore

            resp = requests.post(url, json=payload, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            import urllib.request
            import urllib.error

            req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:  # type: ignore
                raw = r.read().decode("utf-8", "replace")
            try:
                data = json.loads(raw)
            except Exception:
                return draft_output

        # Ollama chat response format: { message: { content: "..." } }
        content = ""
        try:
            content = data.get("message", {}).get("content", "")
        except Exception:
            content = ""
        if not content:
            return draft_output

        # Parse the JSON content. If it fails, return original.
        try:
            refined = json.loads(content)
            if isinstance(refined, dict) and "sections" in refined:
                return refined
        except Exception:
            pass
        return draft_output
    except Exception:
        return draft_output


# ===============
# PDF Text Extraction
# ===============


def extract_text_with_pdfminer(pdf_path: str) -> List[str]:
    """Extract text per page using pdfminer.six. Returns a list of page texts."""
    try:
        from pdfminer.high_level import extract_text
    except Exception:
        # pdfminer not installed; return empty to trigger OCR fallback
        return []

    try:
        # pdfminer returns the whole doc; we'll split on form feed when available or page heuristics
        full_text = extract_text(pdf_path) or ""
        # Split by form feed if present; otherwise split by two or more newlines as a heuristic
        if "\f" in full_text:
            pages = [t for t in full_text.split("\f") if t.strip()]
        else:
            # Heuristic fallback: not perfect, but avoids a giant single string for many docs
            raw = full_text.split("\n\n\n")
            pages = [t for t in raw if t.strip()]
        return pages
    except Exception:
        return []


def ocr_pdf_to_text(pdf_path: str) -> List[str]:
    """OCR the PDF into per-page text using pdf2image + pytesseract.

    Gracefully returns [] if dependencies are missing.
    """
    try:
        from pdf2image import convert_from_path
        import pytesseract
    except Exception:
        return []

    try:
        images = convert_from_path(pdf_path)
    except Exception:
        return []

    pages: List[str] = []
    for img in images:
        try:
            text = pytesseract.image_to_string(img)
        except Exception:
            text = ""
        pages.append(text or "")
    return pages


def get_pages_text(pdf_path: str) -> List[str]:
    """Attempt text-first, then OCR fallback. Always returns a list of page texts."""
    pages = extract_text_with_pdfminer(pdf_path)
    if not any(p.strip() for p in pages):
        # Try OCR fallback for scanned PDFs
        pages = ocr_pdf_to_text(pdf_path)
    # Ensure we always return at least one page string (possibly empty)
    if not pages:
        pages = [""]
    # Apply encoding fixes early to improve downstream detection and matching
    return [fix_mojibake(p or "") for p in pages]


# ===============
# Metadata Extraction (Title, Contract Type, Effective Date)
# ===============


AGREEMENT_KEYWORDS = [
    "Agreement",
    "Contract",
    "Order",
    "Statement of Work",
    "SOW",
    "Addendum",
    "Amendment",
    "License",
    "Services Agreement",
    "Master Services Agreement",
    "Master Subscription Agreement",
]


def pick_first_nonempty(lines: Sequence[str]) -> str:
    for line in lines:
        if line and line.strip():
            return line.strip()
    return ""


def detect_title_and_type(pages: Sequence[str]) -> Tuple[str, str]:
    """Heuristic: choose a strong-looking first-page heading as title and infer contract type.

    Improvements:
      - Prefer known specific agreement titles when present (e.g., "Individual Contributor License Agreement").
      - Otherwise, fall back to scoring lines near the top of page 1.
      - contract_type derives from identified title keywords; else a reasonable default.
    """
    first_page = pages[0] if pages else ""
    # Normalize whitespace for robust matching of titles that may be split across lines
    first_page_normalized = normalize_whitespace(first_page)
    candidate_lines = [l.strip() for l in first_page.splitlines()[:60]]
    candidate_lines = [l for l in candidate_lines if l]

    # Prefer exact known titles if present anywhere on page 1 (case-insensitive)
    KNOWN_TITLES = [
        "Individual Contributor License Agreement",
        "Corporate Contributor License Agreement",
        "Contributor License Agreement",
        "Master Services Agreement",
        "Master Subscription Agreement",
        "Non-Disclosure Agreement",
        "Mutual Non-Disclosure Agreement",
        "Data Processing Addendum",
        "Statement of Work",
        "Order Form",
        "Service Order",
        "Software License Agreement",
        "Professional Services Agreement",
    ]

    lowered_page = first_page.lower()
    lowered_page_normalized = first_page_normalized.lower()
    matched_titles: List[str] = []
    for phrase in KNOWN_TITLES:
        # Match against both raw and whitespace-normalized text to handle multi-line headings
        if (phrase.lower() in lowered_page) or (phrase.lower() in lowered_page_normalized):
            matched_titles.append(phrase)
    if matched_titles:
        # Choose the longest match to be most specific
        chosen = max(matched_titles, key=len)
        title = chosen
        # Derive contract_type from title when it clearly denotes an agreement type
        if re.search(r"\bAgreement\b", chosen, flags=re.I):
            contract_type = chosen
        else:
            # Fallback to keyword-based type
            contract_type = "Agreement"
        return normalize_whitespace(title), normalize_whitespace(contract_type)

    def score_line(line: str) -> int:
        score = 0
        if len(line) <= 120:
            score += 2
        if re.search(r"\b(Agreement|Contract|Order|Amendment|Addendum|License|SOW)\b", line, re.I):
            score += 4
        if line.isupper():
            score += 3
        if re.match(r"^[A-Z][A-Za-z0-9\-\s&()/:]+$", line):
            score += 2
        # Penalize lines that look like addresses or tables of contents
        if re.search(r"Page\s+\d+|Table of Contents|\.{3,}", line, re.I):
            score -= 3
        return score

    best = max(candidate_lines, key=score_line, default=pick_first_nonempty(candidate_lines))
    title = best or ""

    contract_type = ""
    for kw in AGREEMENT_KEYWORDS:
        if re.search(rf"\b{re.escape(kw)}\b", title, flags=re.I):
            contract_type = kw
            break

    if not contract_type:
        # Try scanning first page for a line containing a type keyword
        for line in candidate_lines:
            for kw in AGREEMENT_KEYWORDS:
                if re.search(rf"\b{re.escape(kw)}\b", line, flags=re.I):
                    contract_type = kw
                    break
            if contract_type:
                break

    if not contract_type and title:
        # Fallback: last capitalized word that looks like a type
        words = re.findall(r"[A-Za-z]+", title)
        for w in reversed(words):
            if w.lower() in {"agreement", "contract", "order", "amendment", "addendum", "license"}:
                contract_type = w.capitalize()
                break

    return normalize_whitespace(title), normalize_whitespace(contract_type or "Agreement")


DATE_PATTERNS = [
    # Effective Date formats: "This Agreement is effective as of January 1, 2024"
    r"effective\s+as\s+of\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})",
    r"effective\s+date\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
    r"dated\s+as\s+of\s+([A-Za-z]+\s+\d{1,2},\s*\d{4})",
    r"dated\s*[:\-]?\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})",
    # Numeric formats: 2024-01-31, 01/31/2024, 31/01/2024
    r"effective\s+date\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
    r"effective\s+date\s*[:\-]?\s*(\d{1,2}/\d{1,2}/\d{4})",
]


def parse_date_to_iso(date_str: str) -> Optional[str]:
    """Parse a variety of date strings to YYYY-MM-DD, else None."""
    date_str = date_str.strip()
    # Try known formats fast path
    known_formats = ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y"]
    for fmt in known_formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    # Fallback to dateutil if available
    try:
        from dateutil import parser as dateutil_parser

        dt = dateutil_parser.parse(date_str, dayfirst=False, fuzzy=True)
        return dt.strftime("%Y-%m-%d")
    except Exception:
        return None


def detect_effective_date(pages: Sequence[str]) -> Optional[str]:
    """Search early pages for effective date references and normalize to ISO.

    Returns ISO date string or None if not found.
    """
    search_space = "\n".join(pages[:2])  # focus on first couple of pages
    lowered = search_space.lower()
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, lowered, flags=re.I):
            raw = m.group(1)
            # Recover original casing slice if possible
            start = m.start(1)
            end = m.end(1)
            original_slice = search_space[start:end]
            iso = parse_date_to_iso(original_slice) or parse_date_to_iso(raw)
            if iso:
                return iso
    # As a last resort, scan for any month name + day + year near the top
    m = re.search(r"([A-Za-z]+\s+\d{1,2},\s*\d{4})", search_space)
    if m:
        iso = parse_date_to_iso(m.group(1))
        if iso:
            return iso
    return None


def detect_effective_date_from_acroform(pdf_path: str) -> Optional[str]:
    """Attempt to read effective date from AcroForm fields when present.

    Looks for common date field names (case-insensitive): Date, Effective Date, Date of Agreement, Effective
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception:
        return None

    try:
        reader = PdfReader(pdf_path)
        # Access /AcroForm fields directly for broad compatibility
        root = reader.trailer.get("/Root", {})
        acro = root.get("/AcroForm") if isinstance(root, dict) else None
        if not acro:
            # Some pypdf versions expose via reader.trailer["/Root"].get_object()
            try:
                root_obj = reader.trailer["/Root"].get_object()
                acro = root_obj.get("/AcroForm") if isinstance(root_obj, dict) else None
            except Exception:
                acro = None
        if not acro:
            return None
        fields = acro.get("/Fields")
        if not fields:
            return None

        field_values: Dict[str, str] = {}
        for f in fields:
            try:
                fo = f.get_object()
                name = fo.get("/T")
                value = fo.get("/V")
                if name is None:
                    continue
                name_str = str(name)
                # Value may be a TextStringObject or empty (e.g., <FEFF>)
                value_str = "" if value is None else str(value)
                # Strip BOM markers that appear as "<FEFF>" or equivalents
                value_str = value_str.replace("\ufeff", "").strip().strip("<> ")
                field_values[name_str] = value_str
            except Exception:
                continue

        # Search for date-ish keys first, then parse
        date_keys = [
            "effective date",
            "date of agreement",
            "date",
        ]
        for key in date_keys:
            for fname, fval in field_values.items():
                if key in fname.lower() and fval:
                    iso = parse_date_to_iso(fval)
                    if iso:
                        return iso
        # As a fallback, attempt to parse any field value that looks like a date
        for fval in field_values.values():
            if not fval:
                continue
            iso = parse_date_to_iso(fval)
            if iso:
                return iso
    except Exception:
        return None
    return None


# ===============
# Section and Clause Segmentation
# ===============


SECTION_NUMBER_RE = r"(?P<num>(?:\d+(?:\.\d+)*|[IVXLCM]+))"  # 1, 1.2.3 or Romans
SECTION_TITLE_RE = r"(?P<title>[A-Z][A-Za-z0-9 \-\(\)/,&\.]{2,})"
SECTION_HEADER_RE = re.compile(
    rf"^\s*{SECTION_NUMBER_RE}\.?\s+{SECTION_TITLE_RE}\s*$"
)
UPPERCASE_HEADER_RE = re.compile(r"^[A-Z][A-Z0-9 \-\(\)/,&\.]{3,}$")


@dataclass
class Section:
    title: str
    number: Optional[str]
    body_lines: List[str]


def detect_sections(pages: Sequence[str]) -> List[Section]:
    """Detect section headers in page order.

    Heuristics:
      - Prefer lines like "1. Definitions" or "1.2 Scope of Services".
      - Also accept ALL-CAPS headings as section titles (no number → number=None).
      - Otherwise create a single catch-all section with the entire document body.
    """
    lines: List[Tuple[int, str]] = []
    for page_index, page in enumerate(pages):
        for raw_line in page.splitlines():
            line = raw_line.rstrip()
            lines.append((page_index, line))

    headers: List[Tuple[int, int, Optional[str], str]] = []  # (global_idx, page_idx, number, title)

    def clean_section_title(title: str) -> str:
        # If the matched line accidentally includes the start of the body (e.g., "Grant of ... . Subject to ..."),
        # keep the short, title-like part before the first period when it looks like a header.
        parts = [p.strip() for p in title.split(".")]
        if len(parts) >= 2:
            head = parts[0]
            # Heuristic: short and title-cased head is likely the real section title
            if 2 <= len(head.split()) <= 8 and re.match(r"^[A-Z][A-Za-z0-9 \-\(\)/,&]+$", head):
                return head + "."
        return title.strip()
    for idx, (page_idx, line) in enumerate(lines):
        if not line.strip():
            continue
        m = SECTION_HEADER_RE.match(line)
        if m:
            number = m.group("num")
            raw_title = m.group("title")
            title = clean_section_title(raw_title)
            headers.append((idx, page_idx, number, title))
            continue
        # Consider uppercase header lines (without numbering)
        if len(line) <= 140 and UPPERCASE_HEADER_RE.match(line) and not re.search(r"Page\s+\d+", line):
            headers.append((idx, page_idx, None, line.strip()))

    sections: List[Section] = []
    if not headers:
        body = "\n".join(l for _, l in lines)
        sections.append(Section(title="Body", number=None, body_lines=body.splitlines()))
        return sections

    # Build sections by slicing between headers
    for i, (global_idx, page_idx, number, title) in enumerate(headers):
        next_global = headers[i + 1][0] if i + 1 < len(headers) else len(lines)
        body_slice = [l for _, l in lines[global_idx + 1 : next_global]]
        sections.append(Section(title=title.strip(), number=(number.strip() if number else None), body_lines=body_slice))

    return sections


CLAUSE_LABEL_RE = re.compile(
    r"^\s*(?:(?P<num>\d+(?:\.\d+)*)(?:[\).])?\s+|\((?P<alpha>[a-zA-Z]{1,3})\)\s+|\((?P<roman>i{1,3}|iv|v|vi{0,3}|ix|x)\)\s+)",
    flags=re.I,
)
TITLE_LIKE_PREFIX_RE = re.compile(r"^\s*(?P<title>[A-Z][A-Za-z0-9 \-]{2,})(?:\:|\-\s)\s+")


def split_clauses(body_lines: Sequence[str]) -> List[Tuple[str, str]]:
    """Split a section's body into clauses and extract labels.

    Returns a list of (clause_text, label_string).
    """
    # Group lines into paragraphs first (blank-line separated)
    paragraphs: List[str] = []
    buff: List[str] = []
    for raw in body_lines:
        line = raw.rstrip()
        if line.strip():
            buff.append(line)
        else:
            if buff:
                paragraphs.append(" ".join(buff))
                buff = []
    if buff:
        paragraphs.append(" ".join(buff))

    # Stitch paragraphs that are likely the same clause but were split by extraction artifacts.
    # Merge when previous paragraph does not end with sentence-final punctuation and the next
    # begins with lowercase/punctuation, indicating a continuation.
    def should_merge(prev: str, nxt: str) -> bool:
        if not prev or not nxt:
            return False
        prev_trim = prev.rstrip()
        nxt_lstrip = nxt.lstrip()
        # If previous ends with ., ?, !, or closing quote/paren then do not merge
        if re.search(r'[\.!?]["\)\]]?\s*$', prev_trim):
            return False
        # Merge if next starts with lowercase letter or common punctuation
        return bool(re.match(r'^[a-z\("\'\-]', nxt_lstrip))

    stitched: List[str] = []
    for para in paragraphs:
        if stitched and should_merge(stitched[-1], para):
            stitched[-1] = normalize_whitespace(stitched[-1] + " " + para)
        else:
            stitched.append(para)
    paragraphs = stitched

    clauses: List[Tuple[str, str]] = []
    for para in paragraphs:
        text = normalize_whitespace(para)
        label = ""
        # Try numeric/alpha/roman leading labels like "1.2", "(a)", "(iv)"
        m = CLAUSE_LABEL_RE.match(text)
        if m:
            if m.group("num"):
                label = m.group("num").strip()
            elif m.group("alpha"):
                label = f"({m.group('alpha').strip()})"
            elif m.group("roman"):
                label = f"({m.group('roman').strip()})"
        else:
            # Try title-like prefixes ending with ':' or '- ' e.g., "Definitions: ..."
            m2 = TITLE_LIKE_PREFIX_RE.match(text)
            if m2:
                label = m2.group("title").strip()

        clauses.append((text, label))

    if not clauses:
        # Ensure at least one clause
        body_text = normalize_whitespace(" ".join(body_lines))
        clauses = [(body_text, "")]

    return clauses


# ===============
# Orchestration
# ===============


def parse_contract(pdf_path: str) -> dict:
    pages = get_pages_text(pdf_path)

    # Metadata
    title, contract_type = detect_title_and_type(pages)
    effective_date = detect_effective_date_from_acroform(pdf_path) or detect_effective_date(pages)

    # Sections and clauses
    sections_detected = detect_sections(pages)
    result_sections = []
    for section in sections_detected:
        clauses_raw = split_clauses(section.body_lines)
        clauses_out = []
        for idx, (clause_text, label) in enumerate(clauses_raw):
            clauses_out.append(
                {
                    "text": normalize_whitespace(clause_text),
                    "label": label if label else "",
                    "index": idx,
                }
            )
        result_sections.append(
            {
                "title": normalize_whitespace(section.title) if section.title else "",
                "number": (section.number if section.number else None),
                "clauses": clauses_out,
            }
        )

    draft_output = {
        "title": title,
        "contract_type": contract_type,
        "effective_date": effective_date if effective_date else None,
        "sections": result_sections,
    }
    # Optional local LLM refinement (graceful opt-in)
    try:
        refined = maybe_refine_with_ollama(pages, draft_output)
        if isinstance(refined, dict):
            return refined
    except Exception:
        pass
    return draft_output


def write_json_utf8(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def cli_main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Parse a contract PDF into structured JSON.")
    parser.add_argument("input_pdf", help="Path to input PDF (text-based or scanned)")
    parser.add_argument("output_json", help="Path to write JSON output")
    args = parser.parse_args(argv)

    input_pdf = args.input_pdf
    output_json = args.output_json

    if not os.path.exists(input_pdf):
        print(f"Input PDF does not exist: {input_pdf}", file=sys.stderr)
        return 2

    try:
        result = parse_contract(input_pdf)
        # Enforce JSON schema normalization: ensure types
        if not isinstance(result.get("sections"), list):
            result["sections"] = []
        for sec in result["sections"]:
            sec["number"] = sec["number"] if isinstance(sec.get("number"), str) else None
            clauses = sec.get("clauses") or []
            if not isinstance(clauses, list):
                clauses = []
            for c in clauses:
                c["label"] = c["label"] if isinstance(c.get("label"), str) else ""
                c["index"] = int(c.get("index", 0))
                c["text"] = normalize_whitespace(str(c.get("text", "")))
            sec["clauses"] = clauses

        write_json_utf8(output_json, result)
        return 0
    except Exception as e:
        # Graceful degradation: still emit a minimal valid JSON
        fallback = {
            "title": "",
            "contract_type": "Agreement",
            "effective_date": None,
            "sections": [
                {
                    "title": "Body",
                    "number": None,
                    "clauses": [
                        {"text": "", "label": "", "index": 0},
                    ],
                }
            ],
        }
        try:
            write_json_utf8(output_json, fallback)
        except Exception:
            pass
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(cli_main())


