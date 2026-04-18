import os
from qdrant_client import QdrantClient
from qdrant_client.http import models

client = QdrantClient(url=os.getenv("QDRANT_URL"))

collection_name = os.getenv("QDRANT_COLLECTION_NAME")


def init_collection():
    if not client.collection_exists(collection_name):

        client.create_collection(
            collection_name=collection_name,

            vectors_config={
                "dense": models.VectorParams(
                    size=int(os.getenv("EMBEDDING_DIMENSIONS", 384)),
                    distance=models.Distance.COSINE
                )
            },

            sparse_vectors_config={
                "bm25": models.SparseVectorParams()
            }
        )


# IMPORTANT: expose properly
vectorstore = client