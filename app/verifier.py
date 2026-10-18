"""Layer 1: deterministic, code-only evidence check.

Given a document and a quote the model claims to have copied from it, decide
whether the quote really exists. No model is involved anywhere in this file.

Order of checks:
  1. Exact match after light normalisation (whitespace, curly quotes, case).
  2. Fuzzy match (rapidfuzz if installed, difflib otherwise) to tolerate PDF
     extraction noise, but ONLY if numbers and negations in the quote agree
     with the matched passage. A fuzzy match that changes "30 days" to
     "7 days" or drops a "not" is exactly the kind of error we must catch.

Offsets returned are character positions in the ORIGINAL document text, so
the UI can highlight them directly.
"""
from __future__ import annotations

import difflib
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

try:  # optional, faster and better alignment
    from rapidfuzz import fuzz as _rf_fuzz
except Exception:  # pragma: no cover - exercised when rapidfuzz is absent
    _rf_fuzz = None

FUZZY_THRESHOLD = 90      # percent similarity needed for a fuzzy match
MIN_QUOTE_WORDS = 3       # shorter "quotes" are not meaningful evidence

_CHAR_MAP = {
    "\u2018": "'", "\u2019": "'", "\u201a": "'", "\u201b": "'",
    "\u201c": '"', "\u201d": '"', "\u201e": '"',
    "\u2010": "-", "\u2011": "-", "\u2012": "-", "\u2013": "-", "\u2014": "-",
    "\u00ad": "",  # soft hyphen
}

NUMBER_WORDS = {
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
    "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
    "thousand", "lakh", "crore", "million", "first", "second", "third",
    "half", "twice",
}
NEGATIONS = {"not", "no", "never", "without", "cannot", "neither", "nor", "none"}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Normalise text and return (normalised, idx) where idx[i] is the index
    in the original text of normalised character i."""
    out: list[str] = []
    idx: list[int] = []
    prev_space = True
    for i, ch in enumerate(text):
        ch = _CHAR_MAP.get(ch, ch)
        ch = unicodedata.normalize("NFKC", ch)
        for c in ch:
            if c.isspace():
                if prev_space:
                    continue
                out.append(" ")
                idx.append(i)
                prev_space = True
            else:
                for lc in c.lower():
                    out.append(lc)
                    idx.append(i)
                prev_space = False
    while out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalize(text: str) -> str:
    return normalize_with_map(text)[0]


def clean_quote(quote: str | None) -> str:
    """Strip wrapper quote marks and leading/trailing ellipses the model adds."""
    q = (quote or "").strip()
    q = q.strip("\"'\u201c\u201d\u2018\u2019 ")
    q = re.sub(r"^(\.{3}|\u2026)+\s*", "", q)
    q = re.sub(r"\s*(\.{3}|\u2026)+$", "", q)
    return q.strip()


def _numeric_tokens(s: str) -> Counter:
    toks = _TOKEN_RE.findall(s)
    return Counter(t for t in toks if t[0].isdigit() or t in NUMBER_WORDS)


def _negations(s: str) -> Counter:
    return Counter(t for t in _TOKEN_RE.findall(s) if t in NEGATIONS)


def critical_mismatch(quote_norm: str, span_norm: str) -> str | None:
    """Return a human-readable reason if the quote and passage disagree on
    numbers or negations, else None."""
    qn, sn = _numeric_tokens(quote_norm), _numeric_tokens(span_norm)
    if qn != sn:
        return ("Numbers or amounts in the quote do not match the closest passage "
                f"in the document (quote: {sorted(qn.elements())}, "
                f"document: {sorted(sn.elements())}).")
    if _negations(quote_norm) != _negations(span_norm):
        return "Negation words (not/no/never/without) differ from the closest passage."
    return None


@dataclass
class Match:
    status: str            # "exact" | "fuzzy" | "none"
    start: int | None = None
    end: int | None = None
    score: float = 0.0
    reason: str = ""

    @property
    def found(self) -> bool:
        return self.status in ("exact", "fuzzy")


class DocIndex:
    """Normalise a document once, then check many quotes against it."""

    def __init__(self, document: str):
        self.document = document
        self.norm, self.idx = normalize_with_map(document)

    # -- public ---------------------------------------------------------
    def find(self, quote: str | None) -> Match:
        q = normalize(clean_quote(quote))
        if not q:
            return Match("none", reason="The model gave no quote for this claim.")
        if len(q.split()) < MIN_QUOTE_WORDS:
            return Match("none", reason="The quote is too short to count as evidence.")

        pos = self.norm.find(q)
        if pos >= 0:
            return self._match("exact", pos, pos + len(q), 100.0)

        cand = self._fuzzy(q)
        if cand is None:
            return Match("none", reason="This quote does not appear anywhere in the document.")
        ns, ne, score = cand
        ns, ne = self._expand_to_words(ns, ne)
        problem = critical_mismatch(q, self.norm[ns:ne])
        if problem:
            return Match("none", score=score, reason=problem)
        return self._match("fuzzy", ns, ne, score)

    def context(self, start: int, end: int, pad: int = 250) -> str:
        a, b = max(0, start - pad), min(len(self.document), end + pad)
        return self.document[a:b]

    def search_terms(self, terms: list[str], max_terms: int = 12) -> dict:
        """Look for any of the given words in the document (prefix match, so
        'matern' finds 'maternity'). Used to check claims like 'the document
        does not mention X'. Returns the terms searched and every hit."""
        cleaned: list[str] = []
        for t in terms or []:
            t = normalize(str(t))
            if len(t) >= 3 and t not in cleaned:
                cleaned.append(t)
        cleaned = cleaned[:max_terms]
        hits = []
        for t in cleaned:
            for m in re.finditer(r"\b" + re.escape(t) + r"\w*", self.norm):
                hits.append({
                    "term": t,
                    "start": self.idx[m.start()],
                    "end": self.idx[m.end() - 1] + 1,
                })
        hits.sort(key=lambda h: h["start"])
        return {"terms": cleaned, "count": len(hits), "hits": hits}

    # -- internals ------------------------------------------------------
    def _match(self, status: str, ns: int, ne: int, score: float) -> Match:
        start = self.idx[ns]
        end = self.idx[ne - 1] + 1
        return Match(status, start, end, round(score, 1))

    def _expand_to_words(self, ns: int, ne: int) -> tuple[int, int]:
        n = len(self.norm)
        while ns > 0 and self.norm[ns - 1] != " ":
            ns -= 1
        while ne < n and self.norm[ne] != " ":
            ne += 1
        return ns, ne

    def _fuzzy(self, q: str) -> tuple[int, int, float] | None:
        if _rf_fuzz is not None:
            res = _rf_fuzz.partial_ratio_alignment(q, self.norm, score_cutoff=FUZZY_THRESHOLD)
            if res is None:
                return None
            return res.dest_start, res.dest_end, float(res.score)
        return self._fuzzy_difflib(q)

    def _fuzzy_difflib(self, q: str) -> tuple[int, int, float] | None:
        text, n = self.norm, len(q)
        if n > len(text):
            return None
        step = max(1, n // 10)
        best = (0, 0, 0.0)
        floor = FUZZY_THRESHOLD - 10
        for s in range(0, len(text) - n + 1, step):
            sm = difflib.SequenceMatcher(None, q, text[s:s + n], autojunk=False)
            if sm.real_quick_ratio() * 100 < floor or sm.quick_ratio() * 100 < floor:
                continue
            r = sm.ratio() * 100
            if r > best[2]:
                best = (s, s + n, r)
        if best[2] <= 0:
            return None
        # refine around the best window (shifts and small length changes)
        s0 = best[0]
        for s in range(max(0, s0 - step), min(len(text) - n, s0 + step) + 1):
            for L in range(max(1, n - 6), n + 7):
                if s + L > len(text):
                    continue
                r = difflib.SequenceMatcher(None, q, text[s:s + L], autojunk=False).ratio() * 100
                if r > best[2]:
                    best = (s, s + L, r)
        if best[2] < FUZZY_THRESHOLD:
            return None
        return best


def trust_score(claims: list[dict]) -> dict:
    """0-100 score over claims that rest on a quote. Verified-absent claims
    ('not stated in the document') have no quote, so they are counted
    separately and left out of the score. A claim of absence that the search
    contradicts counts as a failure. If every claim is an absence claim, the
    score is None (shown as n/a)."""
    absent = sum(1 for c in claims if c["status"] == "absent")
    scored = [c for c in claims if c["status"] != "absent"]
    total = len(scored)
    supported = sum(1 for c in scored if c["status"] == "supported")
    unsupported = sum(1 for c in scored if c["status"] == "unsupported")
    retracted = sum(1 for c in scored if c["status"] == "retracted")
    contradicted = sum(1 for c in scored if c["status"] == "contradicted")
    weight = sum((1.0 if c.get("match") == "exact" else 0.9)
                 for c in scored if c["status"] == "supported")
    if total:
        score = round(100 * weight / total)
    else:
        score = None if absent else 0
    return {"score": score, "supported": supported, "unsupported": unsupported,
            "retracted": retracted, "contradicted": contradicted,
            "absent": absent, "total": total}
