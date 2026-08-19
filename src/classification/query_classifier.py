COMPARISON_KEYWORDS = ['compare', 'vs', 'versus', 'difference between', ' or ', 'which is better']
RECENT_KEYWORDS = ['latest', 'recent', 'new developments', 'current state', 'newest', '2024', '2025', '2026']
PAPER_KEYWORDS = ['papers on', 'papers about', 'seminal work', 'key research', 'foundational paper', 'what papers']
APPLICATION_KEYWORDS = ['how is', 'how are', 'applications of', 'used for', 'used in']
# Technique-based is the fallback default — "best method", "how does X work", "what is X"


def classify_query(query: str) -> str:
    """Lightweight keyword-based classifier matching the 5 query types from the
    project brief. Order matters: more specific patterns are checked first,
    falling back to Technique-based as the default catch-all."""
    q = query.lower()

    if any(kw in q for kw in COMPARISON_KEYWORDS):
        return "Comparison-based"
    if any(kw in q for kw in RECENT_KEYWORDS):
        return "Recent-based"
    if any(kw in q for kw in PAPER_KEYWORDS):
        return "Paper-based"
    if any(kw in q for kw in APPLICATION_KEYWORDS):
        return "Application-based"
    return "Technique-based"
