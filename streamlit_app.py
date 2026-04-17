import streamlit as st
import requests
import uuid

# Configuration
API_URL = "http://127.0.0.1:8000/chat"

st.set_page_config(page_title="Enterprise RAG Chat", page_icon="🏢")

# 1. Provide Beautiful Styling Option for Web Application Development
st.markdown("""
    <style>
        .stChatInput {
            bottom: 2rem;
        }
        [data-testid="stSidebar"] {
            background-color: #f8f9fc;
            padding-top: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Enterprise Knowledge Base Chat")

# Session State for Conversation Memory Tracking
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135692.png", width=100)
    st.header("Settings")
    role = st.selectbox(
        "Select your Role",
        ("hr", "finance", "marketing", "engineering", "exec"),
        index=0,
        help="Simulates basic Role-Based Access Control logic for the retriever."
    )
    
    st.markdown("---")
    st.write(f"**Session ID:** `{st.session_state.session_id[:8]}`...")
    
    if st.button("Reset Session Memory"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

# Render Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask about company policies, finances, or systems..."):
    # Render user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Fetch from API
    with st.chat_message("assistant"):
        with st.spinner("Searching internal documents (with Re-ranking)..."):
            try:
                payload = {
                    "query": prompt,
                    "role": role,
                    "session_id": st.session_state.session_id
                }
                response = requests.post(API_URL, json=payload)
                response.raise_for_status()
                
                answer = response.json().get("answer", "No answer provided")
                st.markdown(answer)
                
                # Append to history
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
            except requests.exceptions.RequestException as e:
                st.error(f"Error communicating with backend API: {e}")
