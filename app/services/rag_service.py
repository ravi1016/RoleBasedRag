from app.services.retrieval_service import retrieve
from app.services.llm_service import generate_answer
from app.utils.guardrails import validate_query


def run_rag(query: str, role: str, session_id: str | None = None):

    ok, msg = validate_query(query)
    if not ok:
        return msg

    docs = retrieve(query, role)
    context = "\n\n".join(docs)

    return generate_answer(query, context, session_id)