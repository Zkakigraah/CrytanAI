"""
Extracts plain text from uploaded files (txt, md, docx, pdf) for use as
cipher input. Kept separate from app/ui.py -- pure functions, testable
without Streamlit.

Every format gets reduced to clean prose. Classical ciphers are
alphabetic by nature -- historically, cryptographers stripped structure
(spacing, word boundaries) rather than added it, specifically to deny an
attacker frequency-analysis footholds (see PROJECT_QA.md and the README).
So reducing markdown syntax, docx styling, and PDF layout artifacts down
to the actual words is the right normalization for this project, not a
lossy compromise.
"""
import re

SUPPORTED_EXTENSIONS = ("txt", "md", "docx", "pdf")


def extract_text(filename: str, raw_bytes: bytes) -> str:
    """Dispatches on file extension and returns extracted plain text."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if ext == "txt":
        return _decode_bytes(raw_bytes)
    if ext == "md":
        return _strip_markdown(_decode_bytes(raw_bytes))
    if ext == "docx":
        return _extract_docx(raw_bytes)
    if ext == "pdf":
        return _extract_pdf(raw_bytes)

    raise ValueError(f"Unsupported file type: .{ext} (supported: {', '.join(SUPPORTED_EXTENSIONS)})")


def _decode_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _strip_markdown(text: str) -> str:
    """Removes common Markdown syntax, leaving the underlying prose. Not
    a full CommonMark parser -- doesn't need to be, just needs to clear
    the syntax noise that would otherwise pollute cipher input: headers,
    emphasis markers, link/image syntax, code fences, list bullets, table
    pipes, horizontal rules."""
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)         # code blocks
    text = re.sub(r"`([^`]*)`", r"\1", text)                         # inline code
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", " ", text)                # images
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)             # links -> visible text
    text = re.sub(r"^[ \t]*#{1,6}[ \t]*", "", text, flags=re.MULTILINE)   # headers
    text = re.sub(r"^[ \t]*>[ \t]*", "", text, flags=re.MULTILINE)        # blockquotes
    text = re.sub(r"^[ \t]*[-*+][ \t]+", "", text, flags=re.MULTILINE)    # bullet lists
    text = re.sub(r"^[ \t]*\d+\.[ \t]+", "", text, flags=re.MULTILINE)    # numbered lists
    text = re.sub(r"^[ \t]*-{3,}[ \t]*$", " ", text, flags=re.MULTILINE)  # horizontal rules
    text = re.sub(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(\|[ \t]*:?-{2,}:?[ \t]*)+\|?[ \t]*$", " ",
                   text, flags=re.MULTILINE)                              # table separator rows
    text = re.sub(r"(\*\*\*|\*\*|\*|___|__|_)", "", text)             # emphasis markers
    text = text.replace("|", " ")                                     # table pipes
    return " ".join(text.split())  # collapse whitespace left behind


def _extract_docx(raw_bytes: bytes) -> str:
    import io
    from docx import Document

    doc = Document(io.BytesIO(raw_bytes))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    return " ".join(paragraphs)


def _extract_pdf(raw_bytes: bytes) -> str:
    import io
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return " ".join(" ".join(p.split()) for p in pages)
