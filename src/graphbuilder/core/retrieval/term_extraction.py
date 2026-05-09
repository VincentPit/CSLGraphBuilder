"""Cheap term extraction for the BM25 and Cypher channels.

We don't want a real NER model on the query path — that would add a
heavy dependency and a synchronous LLM call to every turn. Instead we
extract candidate terms with a few cheap signals:

1. **Quoted phrases** — anything inside ``"…"`` or ``'…'`` is preserved
   verbatim (drug names with hyphens, gene aliases with digits, etc.).
2. **Capitalised tokens** outside the very first word — proper nouns
   carry signal in biomedical queries (gene symbols, drug brand names).
3. **All-caps tokens** of 2+ letters — gene symbols (TP53, BRCA1).
4. **Hyphenated compounds** (TNF-alpha, IL-6) and identifier-like tokens
   (ENSG00000…, CHEMBL123).

Stopwords + punctuation are stripped. The output is deduplicated, in
the order extracted, capped at ``max_terms`` so we don't fan-out the
fulltext channel into a giant OR query.
"""

from __future__ import annotations

import re
from typing import List

# Reserve a small biomedical / English stopword list. Anything more
# elaborate (full Snowball) is overkill — the fulltext index handles
# weighting and we only want to discard obviously useless tokens.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on",
    "for", "with", "without", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "what", "which", "who", "when", "where", "why", "how",
    "tell", "me", "about", "show", "find", "list", "give",
    "this", "that", "these", "those", "it", "its", "their",
    "can", "could", "should", "would", "may", "might",
    "from", "into", "between", "among", "vs", "versus",
})

# A token is a run of letters / digits / hyphens / underscores / dots / slashes.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/\-]*")
_QUOTED_RE = re.compile(r"\"([^\"]+)\"|'([^']+)'")


def extract_terms(query: str, max_terms: int = 8) -> List[str]:
    """Pull candidate retrieval terms out of *query*.

    Returns a deduplicated list (preserving first-seen order). Stopwords
    are dropped only when they appear as bare tokens — quoted phrases
    are kept verbatim. The cap of ``max_terms`` is a soft fan-out limit
    for the BM25 channel (each term becomes a substring match).
    """
    if not query:
        return []

    seen: set[str] = set()
    out: List[str] = []

    # 1) Quoted phrases first — highest signal.
    for m in _QUOTED_RE.finditer(query):
        phrase = (m.group(1) or m.group(2) or "").strip()
        if phrase and phrase not in seen:
            seen.add(phrase)
            out.append(phrase)
            if len(out) >= max_terms:
                return out

    # Strip quotes from the query before token-level passes so the
    # quoted phrases don't get re-extracted as fragments.
    stripped = _QUOTED_RE.sub(" ", query)

    tokens = _TOKEN_RE.findall(stripped)
    for i, tok in enumerate(tokens):
        # Trailing/leading punctuation rarely belongs in a search term
        # ("migraine." → "migraine"). We only strip from the boundaries —
        # internal dots / hyphens (TNF-alpha, ENSG…123) are preserved.
        tok = tok.strip("._/-")
        if not tok:
            continue
        lower = tok.lower()

        # Drop obvious stopwords (case-insensitive) but only if they're
        # bare lowercase tokens — keep "IT" if it's all caps (could be
        # an acronym).
        if lower in _STOPWORDS and not tok.isupper():
            continue
        # Single-character tokens that aren't digits (drop "x" but
        # keep "5" in case it's a pathway number).
        if len(tok) == 1 and not tok.isdigit():
            continue

        # Heuristics: keep if any of these hold.
        keep = False
        if "-" in tok or "_" in tok or "/" in tok or "." in tok:
            # Hyphenated / identifier-like — biomedical signal.
            keep = True
        elif tok.isupper() and len(tok) >= 2:
            # Gene symbols, acronyms.
            keep = True
        elif any(c.isdigit() for c in tok):
            # Mixed alphanumeric — likely an identifier.
            keep = True
        elif tok[0].isupper() and i > 0:
            # Capitalised noun later in the sentence.
            keep = True
        elif len(tok) >= 4 and lower not in _STOPWORDS:
            # Long enough to be a content word.
            keep = True

        if not keep:
            continue
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= max_terms:
            break

    return out


__all__ = ["extract_terms"]
