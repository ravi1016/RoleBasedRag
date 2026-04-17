import os
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_qdrant import QdrantVectorStore
from app.utils.embeddings import embeddings

client = QdrantClient(url=os.getenv("QDRANT_URL"))

# Ensure the collection exists before initializing QdrantVectorStore. Qdrant doesn't create it automatically on init.
collection_name = os.getenv("QDRANT_COLLECTION_NAME")
if not client.collection_exists(collection_name):
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=int(os.getenv("EMBEDDING_DIMENSIONS", 384)), distance=Distance.COSINE),
    )

vectorstore = QdrantVectorStore(
    client=client,
    collection_name=collection_name,
    embedding=embeddings
)