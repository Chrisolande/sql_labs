from __future__ import annotations

__all__ = ["MATCHING_STYLES", "NO_MATCH_STYLE"]

MATCHING_STYLES: frozenset[str] = frozenset({"exact_terms", "paraphrase", "distractor"})
NO_MATCH_STYLE: str = "no_match"
