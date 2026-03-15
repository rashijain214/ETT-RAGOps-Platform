# RAGOps Platform – Core RAG Engine
## Overview

RAGOps Platform is a Retrieval-Augmented Generation (RAG) based document intelligence system designed to analyze unstructured PDF documents and generate context-grounded insights using Large Language Models (LLMs).

The platform combines semantic retrieval with controlled generation, ensuring that responses are derived strictly from relevant document context instead of hallucinated model knowledge.

This repository currently contains the core RAG engine, responsible for document ingestion, vector indexing, semantic retrieval, and insight generation.

Future iterations will extend the system with containerization, CI/CD pipelines, scalable vector databases, and Kubernetes-based deployment.

## Key Features (Current Stage)

- PDF document ingestion and text extraction

- Text chunking and semantic embedding generation

- Vector storage with cosine similarity search

- Retrieval-Augmented Generation (RAG) pipeline

- Context-restricted LLM-based insight generation

- REST APIs using FastAPI

- RAG Architecture

The system follows a standard Retrieval-Augmented Generation workflow:

PDF → Text Extraction → Chunking → Embeddings
    → Vector Store → Semantic Retrieval
    → Context-Aware LLM Generation

## Technologies Used

- Python 3

- FastAPI

- Sentence Transformers

- Google Gemini LLM

- Vector Similarity Search (Cosine Similarity)

- SQLite-based Vector Store

- PyPDF2

## Project Structure (Current)

```text
ragops-platform/
├── rag_store.py      # Vector database and similarity search
├── embeddings.py     # Sentence embedding generation
├── retrieve.py       # Semantic retrieval logic
├── ingest.py         # PDF ingestion and chunk embedding
├── insights.py       # LLM-based insight generation
├── api.py            # REST API orchestration
└── requirements.txt
