# Verbatim

**An AI assistant that answers questions about a document, then checks its own answer against the document.**

Hacktoberfest Hack Day Bengaluru 2026 · Track 2: Best Open-Source AI Project · Problem O3: Trustworthy and Verifiable AI

People sign documents they have not fully read: rental agreements, insurance policies, government circulars. AI summaries sound confident but can be wrong, and there is no cheap way to see which parts to trust. Verbatim does not ask the AI to grade itself. The model must show word-for-word evidence for every claim, and **plain code** checks that the evidence exists in the document.

## How it works

1. Upload a PDF (or paste text) and ask a question, e.g. "When do I get my deposit back?"
2. An open-weight model (default `gpt-oss-120b` on Groq, OpenRouter as fallback) drafts an answer.
3. The model splits the answer into atomic claims, each with a verbatim quote from the document.
4. **Layer 1, code check** (`app/verifier.py`): the quote is searched in the document, exact first (ignoring whitespace, case, curly quotes), then fuzzy. A fuzzy match is rejected if numbers or negations differ from the closest passage, so "30 days" can never be quietly matched to "7 days". No match: the model gets one retry to produce a real quote, otherwise the claim is **retracted**.
5. **Layer 2, optional model check**: for claims whose quote really exists, the model confirms the passage supports the claim. If not, the claim is flagged **unsupported**.
6. The UI shows the draft beside the verified answer, highlights supported passages in the document, shows retracted claims struck through with the quote the model cited and why it failed, and gives a trust score.

Retractions are shown openly, never silently fixed.

**Trust score** = weighted share of claims that are supported (exact match 1.0, fuzzy match 0.9, unsupported or retracted 0).

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY and/or OPENROUTER_API_KEY
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and click **Load sample lease + demo question**.

With no API key the app runs in **demo mode**: the model replies are scripted, but the evidence check is the real code running on the real document. Demo mode only answers the sample lease question, and the UI says so. If a live provider fails during that sample question, the app falls back to demo mode and tells the user.

## Deploy (single service, single URL)

FastAPI-style single service: `uvicorn` serves both the API and the frontend.

- **Render**: connect the repo, it reads `render.yaml`. Add `GROQ_API_KEY` (and optionally `OPENROUTER_API_KEY`) as secret env vars.
- **Railway / any Procfile host**: uses `Procfile`.

Health check: `GET /api/health`.

## API

`POST /api/ask` (multipart form): `question`, then either `file` (PDF) or `text`, plus `entailment` (`1`/`0`).
Returns JSON: `draft_answer`, `claims[]` (`claim`, `quote`, `status`, `match`, `start`, `end`, `reason`, `retried`, `retry_quote`, `entailment`), `verified_answer`, `trust`, `document_text`, `notes`, `mode`, `provider`, `model`.
`start`/`end` are character offsets into `document_text`.

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers exact and fuzzy matching, number and negation rejection, retraction logic, trust score, provider fallback on rate limits, and the scripted demo end to end.

## Known limits

- One document and one question at a time. No accounts, no database, no OCR (scanned PDFs are rejected with a clear message).
- Documents over 60,000 characters are truncated.
- Layer 1 proves a quote exists, not that the model read it correctly. Layer 2 is a model judgement and can itself be wrong, which is why it is optional and shown separately.
- Not legal advice.

## Repo layout

```
app/verifier.py   layer 1 evidence check and trust score (no model involved)
app/pipeline.py   draft -> claims -> verify -> retry -> support check
app/llm.py        Groq / OpenRouter client with retry and fallback
app/mock.py       scripted demo model (verifier stays real)
app/ingest.py     PDF text extraction
app/main.py       Starlette app: API and frontend
app/static/       single-page frontend
samples/          sample lease (.txt and .pdf)
tests/            unit tests
```

## Licence

MIT. See `LICENSE`.
