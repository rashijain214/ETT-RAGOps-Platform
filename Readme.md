# RAGOps Platform – Core RAG Engine
## Overview

RAGOps Platform is a Retrieval-Augmented Generation (RAG) based document intelligence system designed to analyze unstructured PDF documents and generate context-grounded insights using Large Language Models (LLMs).

The platform combines semantic retrieval with controlled generation, ensuring that responses are derived strictly from relevant document context instead of hallucinated model knowledge.

This repository currently contains the core RAG engine, responsible for document ingestion, vector indexing, semantic retrieval, and insight generation.

Future iterations will extend the system with containerization, CI/CD pipelines, scalable vector databases, and Kubernetes-based deployment.

## System Architecture

The platform follows a Retrieval-Augmented Generation (RAG) pipeline to ensure accurate and context-aware responses.

PDF Document
     │
     ▼
Text Extraction (PyPDF2)
     │
     ▼
Text Chunking
     │
     ▼
Embedding Generation (Sentence Transformers)
     │
     ▼
Vector Storage (SQLite)
     │
     ▼
Semantic Retrieval (Cosine Similarity)
     │
     ▼
Context Injection
     │
     ▼
LLM Insight Generation (Google Gemini)

### Core Concept

Instead of sending entire documents to an LLM:

Documents are divided into semantic chunks

Each chunk is converted into a vector embedding

The system retrieves relevant chunks using semantic similarity

Only these chunks are given to the LLM as context

This approach significantly reduces hallucination and improves response accuracy.

## Key Features
### PDF Document Ingestion

The system supports ingestion of PDF documents and converts them into machine-readable text.

Capabilities include:

- PDF text extraction

- text preprocessing and cleaning

- document chunk segmentation

Future improvements may include:

- OCR for scanned PDFs

- multi-format document ingestion

### Semantic Chunking

Large documents are divided into smaller semantic chunks to improve retrieval performance.

Benefits include:

- improved embedding quality

- faster similarity search

- more precise context retrieval

Typical chunk size ranges between 300–800 tokens.

### Embedding Generation

Each chunk is converted into a high-dimensional semantic embedding using Sentence Transformers.

Example models: all-MiniLM-L6-v2 and mpnet-base-v2

These embeddings capture:

- contextual meaning

- semantic similarity

- relationships between concepts

### Vector Storage

Embeddings are stored in a SQLite-based vector store, enabling efficient similarity search.

Each stored entry includes:

chunk_id
document_id
text_chunk
embedding_vector
metadata

Similarity search is performed using cosine similarity.

### Semantic Retrieval

When a user submits a query:

1. The query is converted into an embedding

2. The system searches the vector store

3. The top-K most relevant chunks are retrieved

These chunks serve as context for the LLM prompt.

### Context-Aware Insight Generation

Retrieved document chunks are passed to the Google Gemini LLM to generate insights such as:

- document summaries

- explanations

- key findings

- question answering

Example prompt structure:

Context:
[retrieved document chunks]

User Query:
[question]

Instruction:
Answer strictly using the provided context.

This ensures responses remain grounded in the source document.

## REST API Interface

The system exposes its functionality through a FastAPI-based REST API.

Example Endpoints
Upload and Process Documents
POST /ingest

Uploads a PDF and processes it through the ingestion pipeline.

Query the System
POST /query

Allows users to ask questions about indexed documents.

List Documents
GET /documents

Returns all ingested documents.

FastAPI provides:

high performance

asynchronous request handling

automatic API documentation

## Technology Stack
### Core Technologies

- Python 3

- FastAPI

- Sentence Transformers

- Google Gemini LLM

- SQLite (vector store)

- PyPDF2

### AI / ML Concepts

- Retrieval-Augmented Generation (RAG)

- Semantic Embeddings

- Vector Similarity Search

- Prompt Engineering

## Project Structure

```
ragops-platform/

├── rag_store.py
│   Vector database implementation and similarity search
│
├── embeddings.py
│   Sentence embedding generation
│
├── retrieve.py
│   Semantic retrieval logic
│
├── ingest.py
│   PDF ingestion pipeline
│
├── insights.py
│   LLM-based insight generation
│
├── api.py
│   FastAPI REST API server
│
└── requirements.txt
    Project dependencies
```

### Example Workflow
Step 1: Upload a Document
```
POST /ingest
```

The system performs:

- text extraction from the PDF

- document chunking

- embedding generation

- vector storage

Step 2: Query the System
```
POST /query
```

Example question: What are the key findings of the report?

Processing pipeline:

1. query embedding generation

2. vector similarity search

3. top-K context retrieval

4. LLM response generation

### Future Enhancements

The long-term goal is to evolve RAGOps into a scalable document intelligence platform.

Planned improvements include:

Infrastructure

Docker containerization

Kubernetes deployment

CI/CD pipelines

Vector Databases

Replace SQLite with scalable vector databases such as:

Pinecone

Weaviate

Milvus

Retrieval Improvements

hybrid search (BM25 + vector search)

reranking models

query expansion

Document Processing

OCR for scanned PDFs

table extraction

multi-document reasoning

Observability

retrieval quality metrics

hallucination monitoring

query tracing

### Running the Project
Install Dependencies
pip install -r requirements.txt
Start the API Server
uvicorn api:app --reload
Open API Documentation
http://localhost:8000/docs
