from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str):
    return embed_model.encode([text])[0]
