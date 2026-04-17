from app.utils.vectorstore import vectorstore
from app.utils.rbac import get_allowed_depts

from qdrant_client import models
from sentence_transformers import CrossEncoder


# -----------------------------
# Step 1: Load reranker (ONCE)
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -----------------------------
# Step 2: RBAC filter builder
# -----------------------------
def build_filter(role: str):
    allowed = get_allowed_depts(role)

    return models.Filter(
        must=[
            models.FieldCondition(
                key="metadata.department",
                match=models.MatchAny(any=allowed)
            )
        ]
    )


# -----------------------------
# Step 3: Qdrant retrieval (recall stage)
# -----------------------------
def qdrant_search(query: str, role: str):

    qfilter = build_filter(role)

    docs = vectorstore.similarity_search(
        query,
        k=15,
        filter=qfilter
    )

    return docs


# -----------------------------
# Step 4: Cross-Encoder reranking
# -----------------------------
def rerank(query: str, docs):

    if not docs:
        return []

    pairs = [(query, doc.page_content) for doc in docs]

    scores = reranker.predict(pairs)

    scored_docs = sorted(
        zip(docs, scores),
        key=lambda x: x[1],
        reverse=True
    )

    top_docs = [doc for doc, score in scored_docs[:5]]

    return top_docs


# -----------------------------
# Step 5: Public function
# -----------------------------
def retrieve(query: str, role: str):

    docs = qdrant_search(query, role)
    docs = rerank(query, docs)

    return [d.page_content for d in docs]