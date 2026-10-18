"""The agent loop: draft -> claims with quotes -> code check -> retry ->
optional model check -> verified answer + trust score.

The model never grades itself on whether a quote exists: DocIndex (plain code)
decides that. The optional entailment layer only judges whether a quote that
really exists actually supports the claim.

Absence claims ("the document does not mention X") have no quote to check.
For those the model lists search terms and plain code searches the document:
zero matches -> "absent" (verified), any match -> "contradicted".
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor

from .verifier import DocIndex, trust_score

MAX_DOC_CHARS = 60_000

DRAFT_SYSTEM = (
    "You answer a question about a document using ONLY the document. Be concise "
    "(under 120 words). If the document does not say, say so plainly."
)

CLAIMS_SYSTEM = (
    "You break an answer into atomic factual claims and attach evidence to each.\n"
    "Rules:\n"
    "- Output JSON only, exactly: {\"claims\": [{\"claim\": \"...\", \"quote\": \"...\"}]}\n"
    "- Each claim is one checkable statement (max 4-8 claims).\n"
    "- Each quote MUST be copied character-for-character from the DOCUMENT, "
    "10-40 words, containing the key facts of the claim. Never paraphrase a quote.\n"
    "- If you cannot find a supporting passage, set the quote to an empty string.\n"
    "- ABSENCE: if the answer says the document does not mention or does not specify "
    "something, write that as ONE single claim (never several) in this form instead of "
    "a quote: {\"claim\": \"The document does not mention ...\", \"type\": \"absence\", "
    "\"terms\": [\"matern\", \"pregnan\", \"childbirth\"]}. The terms are 3 to 8 lowercase "
    "word stems for the topic and its close synonyms, specific to the topic. Never use "
    "generic words such as policy, cover, claim, insurance or document."
)

RETRY_SYSTEM = (
    "A quote you supplied could not be found in the document. Find a passage in the "
    "DOCUMENT that really supports the claim and copy it character-for-character. "
    "If no passage in the document supports the claim, answer with a null quote.\n"
    "Output JSON only: {\"quote\": \"...\"} or {\"quote\": null}"
)

ENTAIL_SYSTEM = (
    "You judge whether a passage from a document supports a claim. Be strict: the "
    "passage must state or clearly imply the claim, including any numbers, durations, "
    "conditions and exceptions. Output JSON only: "
    "{\"supported\": true|false, \"reason\": \"one short sentence\"}"
)


class PipelineError(Exception):
    pass


def parse_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply."""
    if not text:
        raise PipelineError("The model returned an empty reply.")
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else text[text.find("{"): text.rfind("}") + 1]
    try:
        data = json.loads(candidate)
    except Exception:
        raise PipelineError("The model did not return valid JSON.")
    return data if isinstance(data, dict) else {}


def _absence_terms(item: dict) -> list[str]:
    """Return the model's search terms if this item is an absence claim."""
    if str(item.get("type", "")).strip().lower() != "absence":
        return []
    terms = item.get("terms")
    if not isinstance(terms, list):
        return []
    return [str(t) for t in terms if str(t).strip()]


def run(document: str, question: str, client, entailment: bool = True) -> dict:
    notes: list[str] = []
    truncated = False
    if len(document) > MAX_DOC_CHARS:
        document, truncated = document[:MAX_DOC_CHARS], True
        notes.append(f"Document was long, so only the first {MAX_DOC_CHARS:,} characters were used.")

    index = DocIndex(document)
    used: list[tuple[str, str]] = []

    def call(system: str, user: str, stage: str) -> str:
        text, provider, model = client.complete(system, user, stage)
        used.append((provider, model))
        return text

    doc_block = f"DOCUMENT:\n\"\"\"\n{document}\n\"\"\""

    # 1. Draft answer
    draft = call(DRAFT_SYSTEM, f"{doc_block}\n\nQUESTION: {question}", "draft").strip()

    # 2. Claims with word-for-word quotes (or search terms for absence claims)
    raw = parse_json(call(
        CLAIMS_SYSTEM,
        f"{doc_block}\n\nQUESTION: {question}\n\nANSWER TO BREAK DOWN:\n{draft}",
        "claims"))
    items = [c for c in raw.get("claims", []) if isinstance(c, dict) and str(c.get("claim", "")).strip()]
    if not items:
        raise PipelineError("The model produced no checkable claims for this answer.")

    claims = []
    absence_seen = False
    for i, c in enumerate(items[:10], start=1):
        quote = str(c.get("quote") or "").strip()
        cl = {
            "id": i, "claim": str(c["claim"]).strip(), "quote": quote, "retried": False,
            "retry_quote": None, "match_obj": None, "status": None, "match": None,
            "start": None, "end": None, "score": 0.0, "reason": "", "search": None,
            "entailment": {"checked": False, "supported": None, "reason": ""},
        }

        terms = _absence_terms(c)
        if terms and not absence_seen:
            result = index.search_terms(terms)
            if result["terms"]:
                absence_seen = True  # only the first absence claim is searched
                found = sorted({h["term"] for h in result["hits"]})
                cl["search"] = {"terms": result["terms"], "count": result["count"], "found_terms": found}
                cl["quote"] = ""
                if result["count"] == 0:
                    cl["status"] = "absent"
                else:
                    first = result["hits"][0]
                    cl.update(
                        status="contradicted", start=first["start"], end=first["end"],
                        quote=index.context(first["start"], first["end"], pad=100).strip(),
                        reason=("The model said the document does not mention this, but the "
                                "document contains matching words: " + ", ".join(found) + "."))
                claims.append(cl)
                continue

        cl["match_obj"] = index.find(quote)
        claims.append(cl)

    quote_claims = [c for c in claims if c["status"] is None]

    # 3. Layer 1 retry: ask once more for a real quote where the code found none
    def retry(cl: dict) -> None:
        reply = call(
            RETRY_SYSTEM,
            f"{doc_block}\n\nCLAIM: {cl['claim']}\n\nQUOTE THAT FAILED: {cl['quote'] or '(none given)'}",
            "retry")
        try:
            new_quote = parse_json(reply).get("quote")
        except PipelineError:
            new_quote = None
        cl["retried"] = True
        cl["retry_quote"] = str(new_quote).strip() if new_quote else None
        if cl["retry_quote"]:
            m2 = index.find(cl["retry_quote"])
            if m2.found:
                cl["match_obj"] = m2
            else:
                cl["match_obj"].reason = (
                    f"{cl['match_obj'].reason} A retry also failed: {m2.reason}").strip()
        else:
            cl["match_obj"].reason = (
                f"{cl['match_obj'].reason} On retry the model could not find any passage "
                "that supports this claim.").strip()

    failing = [c for c in quote_claims if not c["match_obj"].found]
    if failing:
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(retry, failing))

    for cl in quote_claims:
        m = cl["match_obj"]
        if m.found:
            cl.update(status="supported", match=m.status, start=m.start, end=m.end, score=m.score)
        else:
            cl.update(status="retracted", match=None, score=m.score, reason=m.reason)

    # 4. Layer 2 (optional): does the real passage actually support the claim?
    def check(cl: dict) -> None:
        passage = index.context(cl["start"], cl["end"])
        reply = call(ENTAIL_SYSTEM, f"CLAIM: {cl['claim']}\n\nPASSAGE:\n{passage}", "entail")
        try:
            data = parse_json(reply)
        except PipelineError:
            data = {}
        ok = data.get("supported")
        if isinstance(ok, bool):
            cl["entailment"] = {"checked": True, "supported": ok,
                                "reason": str(data.get("reason", "")).strip()}
            if not ok:
                cl["status"] = "unsupported"
                cl["reason"] = ("The quote exists in the document but does not support the claim: "
                                + cl["entailment"]["reason"]).strip()
        else:
            notes.append(f"Support check for claim {cl['id']} returned an unreadable reply and was skipped.")

    if entailment:
        to_check = [c for c in quote_claims if c["status"] == "supported"]
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(check, to_check))

    for cl in claims:
        cl.pop("match_obj", None)

    ok_claims = [c["claim"] for c in claims if c["status"] in ("supported", "absent")]
    verified = " ".join(ok_claims) if ok_claims else \
        "No claim in the draft answer could be verified against the document."

    providers = sorted({p for p, _ in used})
    models = sorted({m for _, m in used})
    return {
        "mode": getattr(client, "mode", "live"),
        "provider": ", ".join(providers),
        "model": ", ".join(models),
        "question": question,
        "document_text": document,
        "truncated": truncated,
        "draft_answer": draft,
        "claims": claims,
        "verified_answer": verified,
        "trust": trust_score(claims),
        "entailment_enabled": entailment,
        "notes": notes,
    }