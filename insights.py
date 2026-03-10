import os
import json
import google.generativeai as genai

from .retrieve import search
from .embeddings import get_embedding

genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT = """
You are a careful research assistant. Given ONLY the following passages from a PDF,
extract insights for each category below. Do not use outside knowledge.

User highlighted this text:
"{highlight}"

Related passages:
{related}

Return JSON in this schema:
{
    "key_takeaways": [],
    "did_you_know": [],
    "contradictions": [],
    "examples": [],
    "inspirations": []
}
"""


def generate_insights(highlight):
    # embed query
    q_emb = get_embedding(highlight)

    # retrieve related chunks
    neighbors = search(q_emb, top_k=5)
    related_texts = [n["text"] for n in neighbors]

    related = "\n\n".join(related_texts)

    prompt = PROMPT.format(highlight=highlight, related=related)

    # call LLM
    response = genai.GenerativeModel("gemini-2.0-flash").generate_content(
        prompt,
        generation_config={"response_mime_type": "application/json"}
    )

    try:
        result = json.loads(response.text)
    except json.JSONDecodeError:
        result = {
            "key_takeaways": [],
            "did_you_know": [],
            "contradictions": [],
            "examples": [],
            "inspirations": []
        }

    return result