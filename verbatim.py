from dataclasses import dataclass
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class Claim:
    text: str
    quote: str


@dataclass(frozen=True)
class VerifiedClaim:
    text: str
    quote: str
    supported: bool
    failure_reason: Optional[str] = None


UNSUPPORTED_QUOTE_REASON = "Quote was not found word-for-word in the source document."


def verify_claims(document: str, claims: Iterable[Claim]) -> List[VerifiedClaim]:
    verified: List[VerifiedClaim] = []
    for claim in claims:
        supported = claim.quote in document
        verified.append(
            VerifiedClaim(
                text=claim.text,
                quote=claim.quote,
                supported=supported,
                failure_reason=None if supported else UNSUPPORTED_QUOTE_REASON,
            )
        )
    return verified


def highlight_supported_claims(document: str, verified_claims: Iterable[VerifiedClaim]) -> str:
    highlighted = document
    supported_quotes = sorted(
        {claim.quote for claim in verified_claims if claim.supported and claim.quote},
        key=len,
        reverse=True,
    )

    for quote in supported_quotes:
        highlighted = highlighted.replace(quote, f"<mark>{quote}</mark>")

    return highlighted


def render_claims(verified_claims: Iterable[VerifiedClaim]) -> str:
    lines = []
    for claim in verified_claims:
        if claim.supported:
            lines.append(f'- ✅ {claim.text}\n  - quote: "{claim.quote}"')
        else:
            lines.append(
                f'- ❌ ~~{claim.text}~~\n'
                f'  - attempted quote: "{claim.quote}"\n'
                f"  - reason: {claim.failure_reason}"
            )
    return "\n".join(lines)


def verify_and_render(document: str, claims: Iterable[Claim]) -> dict[str, str]:
    verified = verify_claims(document, claims)
    return {
        "highlighted_source": highlight_supported_claims(document, verified),
        "claims": render_claims(verified),
    }
