from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.utils.logger import get_logger
import os
import time

# -----------------------------
# Logger
# -----------------------------
logger = get_logger(__name__)

# -----------------------------
# In-memory session store
# -----------------------------
session_store = {}


# -----------------------------
# Session memory handler
# -----------------------------
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        logger.debug(f"Creating new session memory | session_id={session_id}")
        session_store[session_id] = InMemoryChatMessageHistory()

    return session_store[session_id]


# -----------------------------
# LLM init
# -----------------------------
def get_llm():
    logger.debug("Initializing Groq LLM (llama-3.1-8b-instant)")

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )


# -----------------------------
# Main generation function
# -----------------------------
def generate_answer(query: str, context: str, session_id: str | None = None):

    start_time = time.time()

    logger.info(f"LLM request started | session_id={session_id}")

    llm = get_llm()

    # -----------------------------
    # Logging inputs (safe)
    # -----------------------------
    logger.debug(f"Query: {query}")
    logger.debug(f"Context length: {len(context)} chars")

    # -----------------------------
    # Prompt template
    # -----------------------------
    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are an enterprise assistant.\n\n"
            "RULES:\n"
            "- Use ONLY the context below\n"
            "- If not found, say 'Not available in internal documents'\n\n"
            "Context:\n{context}"
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])

    chain = prompt | llm

    # -----------------------------
    # Memory wrapper
    # -----------------------------
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="history",
    )

    sid = session_id if session_id else "default"

    logger.debug(f"Using session_id={sid}")

    # -----------------------------
    # Execution
    # -----------------------------
    try:
        response = with_history.invoke(
            {"query": query, "context": context},
            config={"configurable": {"session_id": sid}}
        )

        latency_ms = round((time.time() - start_time) * 1000, 2)

        logger.info(f"LLM response generated | latency={latency_ms}ms")

        logger.debug(f"Response preview: {response.content[:200]}")

        return response.content

    except Exception as e:
        logger.error(f"LLM generation failed: {str(e)}", exc_info=True)
        raise