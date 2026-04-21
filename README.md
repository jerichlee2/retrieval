# Google Drive RAG

A local retrieval-augmented generation (RAG) system for a Google Drive archive.

This project turns a large historical Google Drive corpus into a searchable knowledge base. It retrieves relevant archived documents and uses a local Llama model to answer questions grounded in those documents.

## What this does

The system:

1. inventories a Google Drive archive
2. filters likely useful documents
3. downloads and exports candidate files
4. extracts text from supported file types
5. chunks documents into retrieval-sized units
6. embeds those chunks into vector space
7. stores the vectors in a FAISS index
8. retrieves the top relevant chunks for a question
9. feeds retrieved context into a local Llama model
10. returns a grounded natural-language answer

The goal is not just “chat with an LLM,” but to build a domain-specific knowledge assistant that can answer from archived project evidence.

## Why this exists

Many Google Drive archives contain years of scattered project history across:
- meeting notes
- design reviews
- technical reports
- procedures
- spreadsheets
- onboarding docs
- safety documentation
- source-of-truth documents

A normal LLM does not know this archive. This project makes the archive searchable and usable.

## Current architecture

### Retrieval side
- document inventory from Google Drive
- filtered candidate corpus
- extracted Tier A text corpus
- chunked JSONL corpus
- embedding model for semantic search
- FAISS vector index

### Generation side
- local Llama 3.1 8B Instruct checkpoint
- prompt assembly using retrieved chunks
- answer generation grounded in retrieved evidence

## Repository structure

```text
.
├── config/
│   ├── credentials.json
│   └── token.json
├── data/
│   ├── inventory/
│   ├── manifests/
│   ├── extracted/
│   ├── chunks/
│   └── embeddings/
├── experiments/
├── outputs/
├── scripts/
│   ├── inventory/
│   ├── download/
│   ├── extract/
│   ├── chunk/
│   ├── embed/
│   └── chat/
└── README.md
