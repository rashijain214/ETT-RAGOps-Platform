from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import List
import os
import uuid

from .insights import generate_insights
from .ingest import ingest_document, extract_text_from_pdf
from .embeddings import get_embedding
from .rag_store import search_similar

router = APIRouter()


class HighlightRequest(BaseModel):
    highlight: str


class InsightsResponse(BaseModel):
    key_takeaways: List[str]
    did_you_know: List[str]
    contradictions: List[str]
    examples: List[str]
    inspirations: List[str]


@router.post("/rag/insights", response_model=InsightsResponse)
async def get_insights(req: HighlightRequest):
    result = generate_insights(req.highlight)

    return {
        "key_takeaways": result.get("key_takeaways", []),
        "did_you_know": result.get("did_you_know", []),
        "contradictions": result.get("contradictions", []),
        "examples": result.get("examples", []),
        "inspirations": result.get("inspirations", [])
    }


@router.post("/rag/search_snippets")
async def search_snippets(req: HighlightRequest):
    query_emb = get_embedding(req.highlight)
    results = search_similar(query_emb, top_k=5)

    return {"results": results}


@router.post("/rag/ingest_pdfs")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    total_chunks = 0

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = extract_text_from_pdf(file_path)
        chunks = ingest_document(file.filename, text)

        total_chunks += chunks

        os.remove(file_path)

    return {"message": f"Ingested {len(files)} PDFs with {total_chunks} chunks"}