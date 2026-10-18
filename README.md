# Verbatim

**An AI assistant that answers questions about a document, then checks its own answer against the document.**

Hacktoberfest Hack Day Bengaluru 2026 · Track 2: Best Open-Source AI Project · Problem O3: Trustworthy and Verifiable AI

**Live demo: https://verbatim-ewk5.onrender.com**

> The demo runs on Render's free plan, which sleeps when idle. The first request after a quiet period can take 50 seconds or more. If the page seems stuck, wait a minute and refresh.

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

## Try it

Open the live demo and click **Load sample lease + demo question**, or upload your own PDF. Good questions to try on the included sample insurance policy (`samples/sample_insurance.pdf`):

- **Lookups:** "How long is the initial waiting period, and does it apply to accidents?"
- **Multi-claim:** "I am 62 years old. What costs do I bear on a claim?"
- **Scenario:** "My premium was due on 1 March and I paid on 20 March. Was I covered in between?"
- **Trap (not in the document):** "Are ambulance charges covered?" A trustworthy system should say the document does not say, or retract any invented claim.

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env        # add GROQ_API_KEY and/or OPENROUTER_API_KEY
uvicorn app.main:app --port 8000
```

Open http://localhost:8000 and click **Load sample lease + demo question**.

With no API key the app runs in **demo mode**: the model replies are scripted, but the evidence check is the real code running on the real document. Demo mode only answers the sample lease question, and the UI says so. If a live provider fails during that sample question, the app falls back to demo mode and tells the user.

### Environment variables

| Variable | Purpose |
|---|---|
| `GROQ_API_KEY` | Primary provider key (console.groq.com) |
| `GROQ_MODEL` | Default `openai/gpt-oss-120b` |
| `OPENROUTER_API_KEY` | Fallback provider key (openrouter.ai) |
| `OPENROUTER_MODEL` | Default `openai/gpt-oss-120b` |
| `DOCHECK_MOCK` | Set to `1` to force the scripted demo model |

Never commit `.env`. It is listed in `.gitignore`.

## Deploy on Render (single service, single URL)

`uvicorn` serves both the API and the frontend.

1. Push the repo to GitHub (make sure `app/`, `samples/`, `tests/`, `render.yaml` and `requirements.txt` are included, and `.env` is not).
2. In Render, choose **New + → Blueprint** and select the repo. Render reads `render.yaml`.
3. Enter `GROQ_API_KEY` and `OPENROUTER_API_KEY` when prompted (they are secret env vars, never stored in the repo).
4. Click **Apply**. Once the service shows **Live**, check `GET /api/health`.

Notes:

- Every push to `main` triggers a redeploy. Editing env vars in the dashboard also redeploys.
- To pin the Python version, add `PYTHON_VERSION` (for example `3.12.7`) under `envVars` in `render.yaml`.
- Emergency switch: setting `DOCHECK_MOCK=1` in Render's Environment tab forces the scripted demo if a provider is down or rate-limited.
- To avoid cold starts, ping `/api/health` every 10 minutes with a service such as UptimeRobot.
- Any other host works too: the start command is `uvicorn app.main:app --host 0.0.0.0 --port $PORT`.

## API

`POST /api/ask` (multipart form): `question`, then either `file` (PDF) or `text`, plus `entailment` (`1`/`0`).
Returns JSON: `draft_answer`, `claims[]` (`claim`, `quote`, `status`, `match`, `start`, `end`, `reason`, `retried`, `retry_quote`, `entailment`), `verified_answer`, `trust`, `document_text`, `notes`, `mode`, `provider`, `model`.
`start`/`end` are character offsets into `document_text`.

`GET /api/health` returns OK when the server is running.

## Tests

```bash
python -m unittest discover -s tests -v
```

Covers exact and fuzzy matching, number and negation rejection, retraction logic, trust score, provider fallback on rate limits, and the scripted demo end to end.

## Known limits

- One document and one question at a time. No accounts, no database, no OCR (scanned PDFs are rejected with a clear message).
- Documents over 60,000 characters are truncated.
- Layer 1 proves a quote exists, not that the model read it correctly. Layer 2 is a model judgement and can itself be wrong, which is why it is optional and shown separately.
- The free hosting plan sleeps when idle, so the first request can be slow.
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
samples/          sample lease and sample insurance policy
tests/            unit tests
render.yaml       Render blueprint
```

## Licence

MIT. See `LICENSE`.
