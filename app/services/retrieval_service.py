from qdrant_client import models
from qdrant_client import QdrantClient

from sentence_transformers import CrossEncoder

from app.utils.vectorstore import client, collection_name
from app.utils.embeddings import embeddings
from app.utils.sparse_encoder import encode_sparse_single
from app.utils.rbac import get_allowed_depts

# -----------------------------
# Reranker (load once)
# -----------------------------
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# -----------------------------
# RBAC filter
# -----------------------------
def build_filter(role: str):
    allowed = get_allowed_depts(role)

    return models.Filter(
        must=[
            models.FieldCondition(
                key="department",  # ✅ correct
                match=models.MatchAny(any=allowed)
            )
        ]
    )


# -----------------------------
# Dense search (FIXED QDRANT 1.x)
# -----------------------------
def dense_search(query: str, role: str, k: int = 10):

    vector = embeddings.embed_query(query)

    results = client.query_points(
        collection_name=collection_name,
        query=models.NearestQuery(nearest=vector),
        using="dense",
        limit=k,
        query_filter=build_filter(role)
    ).points

    return results


# -----------------------------
# Sparse search (BM25)
# -----------------------------
def sparse_search(query: str, role: str, k: int = 10):

    sparse = encode_sparse_single(query)

    results = client.query_points(
        collection_name=collection_name,
        query=models.NearestQuery(nearest=models.SparseVector(**sparse)),
        using="bm25",
        limit=k,
        query_filter=build_filter(role)
    ).points

    return results


# -----------------------------
# RRF fusion (SAFE TEXT OUTPUT)
# -----------------------------
def rrf_fusion(dense, sparse, k=10):

    scores = {}
    docs = {}

    def get_id(d):
        return d.id

    def get_text(d):
        return d.payload.get("text", "")

    for rank, d in enumerate(dense):
        _id = get_id(d)
        scores[_id] = scores.get(_id, 0) + 1 / (rank + 1)
        docs[_id] = d

    for rank, d in enumerate(sparse):
        _id = get_id(d)
        scores[_id] = scores.get(_id, 0) + 1 / (rank + 1)
        docs[_id] = d

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    return [docs[_id] for _id, _ in ranked[:k]]


# -----------------------------
# Reranking (FIXED INPUT TYPE)
# -----------------------------
def rerank(query: str, docs, top_k: int = 5):

    if not docs:
        return []

    # ✅ ALWAYS extract text safely
    texts = []
    for d in docs:
        if isinstance(d, str):
            texts.append(d)
        else:
            texts.append(d.payload.get("text", ""))

    # ✅ Build proper pairs
    pairs = [(query, t) for t in texts]

    # 🔥 Important: convert to list (avoid generator issues)
    scores = reranker.predict(pairs)

    ranked = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)

    return [doc for doc, _ in ranked[:top_k]]
    


# -----------------------------
# Query rewriting (conversation-aware)
# -----------------------------
def rewrite_query(query: str, chat_history: list, llm):

    if not chat_history:
        return query

    history_text = "\n".join(
        f"{h['role'].upper()}: {h['content']}"
        for h in chat_history[-5:]
    )

    from langchain_core.prompts import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = PromptTemplate.from_template("""
You are a query rewriting assistant.

Convert the user question into a standalone search query.

Chat History:
{history}

User Query:
{query}

Standalone Query:
""")

    chain = prompt | llm | StrOutputParser()

    rewritten = chain.invoke({
        "history": history_text,
        "query": query
    })

    print("\n🧠 Rewritten Query:", rewritten)

    return rewritten.strip()


# -----------------------------
# MAIN PIPELINE
# -----------------------------
def retrieve(query: str, role: str, chat_history=None, llm=None):

    dense = dense_search(query, role, k=50)
    sparse = sparse_search(query, role, k=50)

    print(f"\n✅ Dense count: {len(dense)}")
    print(f"✅ Sparse count: {len(sparse)}")

    fused_docs = rrf_fusion(dense, sparse, k=20)
    final_docs = rerank(query, fused_docs, top_k=5)

    return final_docs