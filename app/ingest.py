"""Turn an uploaded PDF (or pasted text) into plain document text."""
from __future__ import annotations

import io

MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_PAGES = 80


class IngestError(Exception):
    pass


def pdf_to_text(data: bytes) -> str:
    if len(data) > MAX_UPLOAD_BYTES:
        raise IngestError("That PDF is larger than 10 MB. Please upload a smaller file.")
    try:
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise IngestError("This PDF is password protected.")
        pages = reader.pages[:MAX_PAGES]
        text = "\n\n".join((p.extract_text() or "").strip() for p in pages).strip()
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"Could not read this PDF ({exc.__class__.__name__}).")
    if len(text) < 20:
        raise IngestError(
            "No text could be extracted. Scanned PDFs need OCR, which Verbatim does not"
            "support. Paste the text instead."
        )
    return text
