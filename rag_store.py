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

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def search_similar(query_embedding, top_k=5):
    store = open_store()
    results = []

    q = np.array(query_embedding, dtype=float)

    for k, v in store.items():
        emb = np.array(v["embedding"], dtype=float)
        sim = _cosine_similarity(q, emb)
        results.append({
            "score": float(sim),
            "text": v["text"],
            "metadata": v.get("metadata", {}),
            "chunk_id": k
        })

    store.close()
    results.sort(key=lambda x: x["score"], reverse=True)
    return results[:top_k]

def delete_document_chunks(doc_name: str) -> dict:
    """
    Delete all chunks whose metadata['doc'] == doc_name.
    Works with your ingest.py which stores: {"doc": doc_id}.
    """
    store = open_store()
    keys_to_delete = []

    for k, v in store.items():
        doc = (v.get("metadata") or {}).get("doc")
        if doc == doc_name:
            keys_to_delete.append(k)

    for k in keys_to_delete:
        del store[k]

    store.close()
    return {"doc_name": doc_name, "deleted_chunks": len(keys_to_delete)}