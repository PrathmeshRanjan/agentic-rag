from pathlib import Path
from app.services.ingestion import load_file, chunk_documents
from app.rag.vectorstore import add_documents

folder = Path('data/sample_kb')
files = [p for p in folder.iterdir() if p.is_file()]


all_docs = []
for path in files:
    all_docs.extend(load_file(path))
chunks = chunk_documents(all_docs)
ids = add_documents(chunks)
print(f"Indexed {len(files)} files -> {len(chunks)} chunks -> {len(ids)} Pinecone vectors")