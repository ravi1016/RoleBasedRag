from app.services.retrieval_service import retrieve
from app.services.llm_service import generate_answer
from app.utils.guardrails import validate_query
from app.utils.logger import get_logger

logger = get_logger(__name__)


def run_rag(query: str, role: str, session_id: str | None = None, llm=None):

    logger.info(f"RAG start | query='{query}' | role='{role}' | session_id='{session_id}'")

    # 1. Guardrails
    ok, msg = validate_query(query)
    if not ok:
        logger.warning(f"Guardrail blocked query: {query}")
        return msg

    # 2. Retrieval
    try:
        docs = retrieve(query, role)
        logger.info(f"Retrieved documents: {len(docs)}")
    except Exception as e:
        logger.error(f"Retrieval failed: {str(e)}", exc_info=True)
        return "Error during retrieval"

    if not docs:
        logger.warning("No documents found for query")
        return "Not available in internal documents"

    # 3. Build context
    try:
        context = "\n\n".join(d.payload.get("text", "") for d in docs)
        logger.debug(f"Context length: {len(context)}")
        logger.debug(f"Top doc preview: {docs[0].payload.get('text', '')[:150]}")
    except Exception as e:
        logger.error(f"Context building failed: {str(e)}", exc_info=True)
        return "Error building context"

    # 4. Generate answer
    try:
        answer = generate_answer(query, context, session_id)
        logger.info("Answer generated successfully")
    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}", exc_info=True)
        return "Error generating answer"

    logger.info("RAG completed successfully")

    return answer