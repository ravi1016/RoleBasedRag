import re

PII_PATTERNS = [
    r"\b\d{10}\b",
    r"\b\d{12}\b",
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}"
]

def validate_query(query: str):

    for p in PII_PATTERNS:
        if re.search(p, query):
            return False, "PII detected in query"

    if "ignore previous instructions" in query.lower():
        return False, "Prompt injection detected"

    return True, None