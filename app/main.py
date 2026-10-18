"""Verbatim web service: one process serves the API and the frontend.

Run:  uvicorn app.main:app --host 0.0.0.0 --port 8000
"""
from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from pathlib import Path

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from .ingest import IngestError, pdf_to_text
from .llm import LLMError, get_client, load_env
from .mock import DEMO_QUESTION, SAMPLE_PATH, MockLLM
from .pipeline import PipelineError, run

load_env()
STATIC = Path(__file__).parent / "static"
MAX_QUESTION_CHARS = 500
_CACHE: "OrderedDict[str, dict]" = OrderedDict()
_CACHE_MAX = 50


def _err(message: str, status: int) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def index(request):
    return FileResponse(STATIC / "index.html")


async def health(request):
    client = get_client()
    return JSONResponse({"status": "ok", "mode": getattr(client, "mode", "live"),
                         "providers": client.describe()})


async def sample(request):
    return JSONResponse({"text": SAMPLE_PATH.read_text(), "question": DEMO_QUESTION})


def _answer(document: str, question: str, entailment: bool) -> dict:
    key = hashlib.sha256(f"{int(entailment)}\x00{question}\x00{document}".encode()).hexdigest()
    if key in _CACHE:
        _CACHE.move_to_end(key)
        return {**_CACHE[key], "cached": True}

    client = get_client()
    fallback_note = None
    try:
        result = run(document, question, client, entailment)
    except (LLMError, PipelineError) as exc:
        # Last resort for the demo: scripted model, real verifier.
        if getattr(client, "mode", "") == "live" and MockLLM.supports(document, question):
            fallback_note = f"Live model unavailable ({exc}). Showing the scripted demo model instead."
            result = run(document, question, MockLLM(), entailment)
        else:
            raise
    if getattr(client, "mode", "") == "mock" and not MockLLM.supports(document, question):
        raise PipelineError(
            "No API key is configured, so Verbatim is in demo mode. Demo mode only answers the "
            "sample lease question. Set GROQ_API_KEY or OPENROUTER_API_KEY to use your own documents.")
    if fallback_note:
        result["notes"].append(fallback_note)
        result["fallback"] = True
    if getattr(client, "mode", "") == "mock":
        result["notes"].append("Demo mode: model replies are scripted. The evidence check is real.")

    result["cached"] = False
    _CACHE[key] = result
    if len(_CACHE) > _CACHE_MAX:
        _CACHE.popitem(last=False)
    return result


async def ask(request):
    form = await request.form()
    question = str(form.get("question") or "").strip()
    text = str(form.get("text") or "")
    entailment = str(form.get("entailment", "1")).lower() not in ("0", "false", "off", "")
    upload = form.get("file")

    if not question:
        return _err("Please type a question about the document.", 400)
    if len(question) > MAX_QUESTION_CHARS:
        return _err(f"Please keep the question under {MAX_QUESTION_CHARS} characters.", 400)

    document = text
    if upload is not None and getattr(upload, "filename", ""):
        data = await upload.read()
        try:
            document = await run_in_threadpool(pdf_to_text, data)
        except IngestError as exc:
            return _err(str(exc), 400)
    document = document.replace("\r\n", "\n").strip()
    if len(document) < 20:
        return _err("Upload a PDF or paste the document text first.", 400)

    # Mock-mode guard happens before any work so users get a fast, clear message.
    client = get_client()
    if getattr(client, "mode", "") == "mock" and not MockLLM.supports(document, question):
        return _err("No API key is configured, so Verbatim is in demo mode. Demo mode only "
                    "answers the sample lease question. Set GROQ_API_KEY or OPENROUTER_API_KEY "
                    "to use your own documents.", 400)
    try:
        result = await run_in_threadpool(_answer, document, question, entailment)
    except LLMError as exc:
        return _err(str(exc), 502)
    except PipelineError as exc:
        return _err(str(exc), 502)
    return JSONResponse(result)


routes = [
    Route("/", index),
    Route("/api/health", health),
    Route("/api/sample", sample),
    Route("/api/ask", ask, methods=["POST"]),
    Mount("/static", StaticFiles(directory=str(STATIC)), name="static"),
]

app = Starlette(routes=routes)

if __name__ == "__main__":  # pragma: no cover
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.environ.get("PORT", "8000")))
