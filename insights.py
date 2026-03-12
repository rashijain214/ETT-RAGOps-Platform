import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

from retrieve import search
from embeddings import get_embedding

# Load .env variables
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))

PROMPT = """
You are a careful research assistant. Given ONLY the following passages from a PDF,
extract insights for each category below. Do not use outside knowledge.

User highlighted this text:
"{highlight}"

Related passages:
{related}

Return ONLY valid JSON. Do not include explanations, markdown, or extra text.

{{
    "key_takeaways": [],
    "did_you_know": [],
    "contradictions": [],
    "examples": [],
    "inspirations": []
}}
"""


def generate_insights(highlight):
    try:
        # Create embedding for query
        q_emb = get_embedding(highlight)
        # Retrieve related passages
        neighbors = search(highlight, q_emb, top_k=5)
        related_texts = [n["text"] for n in neighbors]

        related = "\n\n".join(related_texts)

        # Fill prompt
        prompt = PROMPT.format(highlight=highlight, related=related)

        # Call Gemini
        model = genai.GenerativeModel("gemini-2.5-flash")
        response = model.generate_content(prompt)

        text = response.text

        # Extract JSON from response safely
        json_match = re.search(r"\{.*\}", text, re.DOTALL)

        if json_match:
            result = json.loads(json_match.group())
        else:
            raise ValueError("No JSON found in response")

    except Exception as e:
        print("Insight generation error:", e)

        result = {
            "key_takeaways": [],
            "did_you_know": [],
            "contradictions": [],
            "examples": [],
            "inspirations": []
        }

    return result
