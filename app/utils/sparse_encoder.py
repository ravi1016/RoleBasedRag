from fastembed import SparseTextEmbedding

# Use FastEmbed's pre-trained BM25 instead of a local TfidfVectorizer
# which would otherwise need to be fitted and saved to disk.
model = SparseTextEmbedding(model_name="Qdrant/bm25")

def encode_sparse(texts):
    sparse_vectors = []
    
    for res in model.embed(texts):
        sparse_vectors.append({
            "indices": res.indices.tolist(),
            "values": res.values.tolist()
        })
        
    return sparse_vectors

def encode_sparse_single(text: str):
    res = list(model.embed([text]))[0]
    
    return {
        "indices": res.indices.tolist(),
        "values": res.values.tolist()
    }