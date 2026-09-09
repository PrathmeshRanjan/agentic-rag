# Agentic RAG

An Agentic Retrieval-Augmented Generation (RAG) project built with LangGraph, LangChain, Pinecone, Google Generative AI embeddings, and Tavily web search. The system routes user questions to the best evidence source, evaluates whether the retrieved context is strong enough to answer, and falls back to web search when the private knowledge base is insufficient.

This repo is a practical implementation of the architecture and workflow shown in the included reference diagrams:

- Architecture diagram: `architecture.png`
- Workflow diagram: `workflow.png`

## About this project

This project demonstrates a production-style agentic RAG pattern:

- The router decides whether a question should be answered from a private knowledge base or handled as a direct conversational request.
- A retriever queries a Pinecone vector database built from documentation chunks.
- An evidence grader decides whether the retrieved KB context is strong enough.
- If the private KB is weak, the workflow searches the web with Tavily.
- A second grader evaluates the web results before generating the final answer.
- If both sources are insufficient, the agent transparently states that it cannot answer confidently.

The result is a more reliable and explainable answer system than a simple retrieval pipeline because it includes routing, grading, and fallback logic designed for real-world knowledge tasks.

## Architecture overview

The system follows the architecture shown in `architecture.png`.

### High-level flow

1. User asks a question.
2. Router LLM decides whether the question belongs to the KB workflow or should be answered directly.
3. If KB is relevant, the system retrieves top matches from Pinecone.
4. A grading step checks whether the KB evidence is sufficient.
5. If evidence is weak, the system performs a Tavily web search.
6. Another grader checks the web evidence.
7. If neither source is adequate, the agent rewrites the query or returns an explicit insufficient-evidence answer.
8. The final answer is generated from the best available source and returned to the user.

### Components

- User / Router: interprets the incoming question and selects the best route.
- Knowledge base: private documents stored as embeddings in Pinecone.
- Retriever: queries the vector database for semantically relevant chunks.
- Evidence grading: checks whether retrieved content is strong enough to answer.
- Web search tool: Tavily provides live web results and snippets.
- LLM answer generation: synthesizes a final answer using only the relevant evidence.

## Workflow overview

The decision graph in `workflow.png` follows this logic:

```text
start
  -> route_question
      -> retrieve_kb / direct_answer
      -> grade_kb_evidence
          -> generate_from_kb (good)
          -> search_web (weak)
              -> grade_web_evidence
                  -> generate_from_web (good)
                  -> rewrite_query (weak, retry allowed)
                  -> answer_insufficient (final fallback)
```

This graph highlights the core agentic pattern:

- route the query,
- fetch evidence,
- evaluate quality,
- retry or fall back,
- generate an answer grounded in the best evidence.

## Features

- Agentic question routing between private KB and direct chat
- Semantic retrieval from a Pinecone vector database
- Document chunking and embedding pipeline
- Evidence grading before answer generation
- Tavily web fallback for incomplete or missing KB coverage
- Query rewriting for better retrieval on retry
- Explicit handling of insufficient-evidence cases
- Beginner-friendly answer generation grounded in source context

## Tech stack

- Python 3.10+
- LangChain
- LangGraph
- Pinecone
- Google Generative AI Embeddings
- Tavily Search
- BeautifulSoup for web loading
- LangChain document splitting and retrieval components

## Project structure

```text
.
├── README.md
├── architecture.png
├── workflow.png
├── agentic-rag.ipynb
├── pyproject.toml
├── requirements.txt
├── uv.lock
├── .env
├── .venv/
└── .gitignore
```

This project is currently notebook-driven rather than relying on a standalone `main.py` entrypoint. The interactive workflow and the indexing logic live in `agentic-rag.ipynb`.

## Setup

### Prerequisites

Before running the project, make sure you have:

- Python 3.10 or newer
- A Pinecone account and API key
- A Google AI API key for Gemini embeddings
- A Tavily API key
- Access to the LangChain / LangGraph ecosystem packages

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

Using pip:

```bash
pip install -r requirements.txt
```

Or using uv:

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root with the following values:

```env
PINECONE_API_KEY=your_pinecone_api_key
GOOGLE_API_KEY=your_google_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Depending on the model setup you use, you may also need additional environment variables for the specific LLM provider.

## How the RAG pipeline works

The repository is centered on the notebook-based implementation, which does the following:

> The main logic is implemented in `agentic-rag.ipynb`; there is no separate Python application entrypoint in this repo at the moment.

### 1. Load source documents

The notebook loads documentation from a public source URL using a web loader.

### 2. Chunk and embed documents

Documents are split into chunks and embedded with Google Generative AI embeddings before being stored in Pinecone.

### 3. Create the vector database

A Pinecone index is created if it does not exist, and the document chunks are uploaded into a named namespace.

### 4. Build the LangGraph workflow

The workflow uses a state graph with nodes for:

- routing the question,
- retrieving from KB,
- grading evidence,
- searching the web,
- rewriting the query,
- generating final answers,
- ending in the direct or fallback path.

### 5. Answer with citations and grounded evidence

The generated answer is based on the best available evidence source:

- private KB when sufficient,
- web results when KB is weak,
- direct response for simple conversational queries,
- explicit insufficient-evidence response when reliability is too low.

## Example usage

The workflow is designed to answer questions like:

- What is Agentic RAG?
- How does a LangGraph RAG workflow decide between KB and web search?
- How do retrievers and graders work together?
- When should the system fall back to web evidence?

The notebook provides a complete interactive example for building the index and running the workflow.