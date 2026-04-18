from app.services.retrieval_service import retrieve
from app.services.llm_service import generate_answer
from app.utils.guardrails import validate_query


def run_rag(query: str, role: str, session_id: str | None = None, llm=None):

    # 1. Guardrails
    ok, msg = validate_query(query)
    if not ok:
        return msg

    # 2. Retrieval (NO manual chat memory)
    docs = retrieve(query, role)

    # 3. Build context
    context = "\n\n".join(d.payload.get("text", "") for d in docs)

    # 4. Generate answer (LangChain handles memory internally)
    answer = generate_answer(query, context, session_id)

    return answer