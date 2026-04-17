from langchain_groq import ChatGroq
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import os

session_store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = InMemoryChatMessageHistory()
    return session_store[session_id]

def get_llm():
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )

def generate_answer(query: str, context: str, session_id: str | None = None):
    llm = get_llm()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise assistant.\n\nRULES:\n- Use ONLY the context below\n- If not found, say 'Not available in internal documents'\n\nContext:\n{context}"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{query}")
    ])
    
    chain = prompt | llm
    
    with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="query",
        history_messages_key="history",
    )
    
    sid = session_id if session_id else "default"
    
    response = with_history.invoke(
        {"query": query, "context": context},
        config={"configurable": {"session_id": sid}}
    )
    return response.content