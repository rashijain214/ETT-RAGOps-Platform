import numpy as np
from sqlitedict import SqliteDict

DB_PATH = "rag_store.sqlite"

def open_store():
    return SqliteDict(DB_PATH, autocommit=True)

def add_chunk(store, chunk_id, text, embedding, metadata=None):
    store[chunk_id] = {
        "text": text,
        "embedding": np.array(embedding).tolist(),
        "metadata": metadata or {}
    }

def get_all_chunks(store):
    return list(store.items())

def search_similar(query_embedding, top_k=5):
    store = open_store()
    results = []
    for k, v in store.items():
        emb = np.array(v["embedding"])
        sim = np.dot(query_embedding, emb) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(emb)
        )
        results.append({
            "score": float(sim),
            "text": v["text"],
            "metadata": v.get("metadata", {}),
            "chunk_id": k
        })
    store.close()
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]
