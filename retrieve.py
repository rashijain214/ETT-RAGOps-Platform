import numpy as np
from .rag_store import open_store

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def search(query_emb, top_k=5):
    """
    Perform semantic search over stored document embeddings.

    Args:
        query_emb (np.ndarray): Embedding of the user query
        top_k (int): Number of top relevant chunks to return

    Returns:
        List of top-k most relevant document chunks
    """
    store = open_store()
    results = []

    for _, entry in store.items():
        emb = np.array(entry["embedding"])
        sim = cosine_sim(query_emb, emb)
        results.append((sim, entry))

    store.close()
    results.sort(key=lambda x: x[0], reverse=True)

    return [entry for _, entry in results[:top_k]]
