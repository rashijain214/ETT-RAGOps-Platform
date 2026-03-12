from fastapi import APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import List
import os
import uuid

from insights import generate_insights
from ingest import ingest_document, extract_text_from_pdf
from embeddings import get_embedding
from retrieve import search
from rag_store import open_store

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
    results = search(req.highlight, query_emb, top_k=5)

    return {"results": results}


@router.post("/rag/ingest_pdfs")
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    total_chunks = 0
    uploaded_files = []

    upload_dir = os.path.join(os.path.dirname(__file__), "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    for file in files:
        file_path = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")

        with open(file_path, "wb") as f:
            f.write(await file.read())

        text = extract_text_from_pdf(file_path)
        chunks = ingest_document(file.filename, text)

        total_chunks += chunks
        uploaded_files.append({
            "filename": file.filename,
            "chunks": chunks
        })

        os.remove(file_path)

    return {
        "message": f"Successfully ingested {len(files)} PDF(s) with {total_chunks} total chunks",
        "files": uploaded_files,
        "total_chunks": total_chunks
    }


@router.get("/rag/list_documents")
async def list_documents():
    """List all documents in the database with their chunk counts."""
    store = open_store()
    doc_counts = {}
    
    for key, entry in store.items():
        doc_name = entry.get("metadata", {}).get("doc", "Unknown")
        doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
    
    store.close()
    
    documents = [{"name": name, "chunks": count} for name, count in doc_counts.items()]
    return {"documents": documents, "total_documents": len(documents)}


@router.delete("/rag/clear_database")
async def clear_database():
    """Clear all documents from the database."""
    try:
        store = open_store()
        store.clear()
        store.commit()
        store.close()
        return {"message": "Database cleared successfully"}
    except Exception as e:
        return {"error": str(e)}