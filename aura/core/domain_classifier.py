"""Shared domain/topic classification for AURA subsystems.

Single authoritative source for keyword-to-domain mapping. Used by:
- consciousness/metacognition.py (get_domain_for_query)
- consciousness/self_improvement.py (_infer_domain)
- consciousness/strategy_bandit.py (ProblemClassifier)
- patterns/pattern_prophet.py (_classify_topic)
"""

from typing import Dict, List

# Canonical domain keyword map
DOMAIN_KEYWORDS: Dict[str, List[str]] = {
    "coding": [
        "code", "function", "class", "variable", "bug", "error", "debug",
        "python", "javascript", "typescript", "rust", "java", "html", "css",
        "api", "endpoint", "database", "sql", "git", "deploy", "docker",
        "test", "refactor", "compile", "import", "module", "package",
        "algorithm", "data structure", "regex", "syntax",
    ],
    "math": [
        "calculate", "equation", "formula", "math", "number", "statistics",
        "probability", "algebra", "calculus", "geometry", "proof", "theorem",
        "integral", "derivative", "matrix", "vector", "linear",
    ],
    "science": [
        "experiment", "hypothesis", "research", "paper", "study",
        "biology", "chemistry", "physics", "neuroscience", "genome",
        "molecule", "quantum", "evolution", "climate", "astronomy",
    ],
    "writing": [
        "write", "essay", "article", "blog", "story", "email", "letter",
        "draft", "edit", "proofread", "grammar", "paragraph", "outline",
        "summarize", "paraphrase", "tone", "style",
    ],
    "reasoning": [
        "analyze", "compare", "evaluate", "explain", "why", "how",
        "cause", "effect", "argument", "logic", "evidence", "conclusion",
        "pros", "cons", "trade-off", "decision",
    ],
    "creative": [
        "idea", "brainstorm", "imagine", "design", "creative", "art",
        "music", "poem", "fiction", "game", "invent", "novel",
    ],
    "casual": [
        "hello", "hi", "hey", "thanks", "how are you", "what's up",
        "good morning", "joke", "fun", "chat",
    ],
}


def classify_domain(text: str) -> str:
    """Classify text into a domain based on keyword matching.

    Returns the best-matching domain name, or 'general' if no strong match.
    """
    if not text:
        return "general"
    text_lower = text.lower()
    scores: Dict[str, int] = {}
    for domain, keywords in DOMAIN_KEYWORDS.items():
        scores[domain] = sum(1 for kw in keywords if kw in text_lower)
    if not scores:
        return "general"
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "general"


def get_domain_keywords(domain: str) -> List[str]:
    """Get keywords for a specific domain."""
    return DOMAIN_KEYWORDS.get(domain, [])
