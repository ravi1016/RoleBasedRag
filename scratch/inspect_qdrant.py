from qdrant_client import QdrantClient
import inspect
sig = inspect.signature(QdrantClient.query_points)
for param in sig.parameters.values():
    print(param)
