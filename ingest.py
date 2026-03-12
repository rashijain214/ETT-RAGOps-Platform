import numpy as np
from rag_store import open_store, add_chunk
from embeddings import get_embedding
from PyPDF2 import PdfReader


def chunk_text(text, chunk_size=120, overlap=30):

    words = text.split()
    chunks = []
    i = 0

    while i < len(words):
        chunk = words[i:i + chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap

    return chunks


def ingest_document(doc_id, text):

    store = open_store()

    chunks = chunk_text(text)

    for idx, chunk in enumerate(chunks):

        emb = get_embedding(chunk)

        add_chunk(
            store,
            f"{doc_id}_chunk_{idx}",
            chunk,
            emb,
            {
                "doc": doc_id   # this stores the PDF name
            }
        )

    store.close()

    return len(chunks)


def extract_text_from_pdf(file_path):

    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    return text