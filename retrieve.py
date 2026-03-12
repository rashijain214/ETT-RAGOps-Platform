import numpy as np
from rag_store import open_store


def cosine_sim(a, b):
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def keyword_score(query, text):
    q = set(query.lower().split())
    t = set(text.lower().split())

    if not q:
        return 0

    overlap = q.intersection(t)
    return len(overlap) / len(q)


def search(query_text, query_emb, top_k=5, threshold=0.30):
    """
    Hybrid search: semantic + keyword filtering
    """

    store = open_store()
    results = []

    for _, entry in store.items():

        text = entry["text"]
        emb = np.array(entry["embedding"])

        semantic = cosine_sim(query_emb, emb)
        keyword = keyword_score(query_text, text)

        score = 0.7 * semantic + 0.3 * keyword

        if score >= threshold:
            results.append((score, entry))

    store.close()

    results.sort(key=lambda x: x[0], reverse=True)

    output = []

    for score, entry in results[:top_k]:
        text = entry["text"]
        snippet = text if len(text) <= 300 else text[:300] + "..."

        output.append({
            "text": snippet,
            "similarity": round(score * 100, 2),
            "source": entry.get("metadata", {}).get("doc", "Unknown")
        })

    return output
