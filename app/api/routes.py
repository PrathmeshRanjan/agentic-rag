from pathlib import Path
import os

from dotenv import load_dotenv
from fastapi import APIRouter, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.rag.workflow import ask
from app.rag.vectorstore import add_documents
from app.services.audit import init_db, write_audit
from app.services.ingestion import SUPPORTED, chunk_documents, load_file

load_dotenv()
init_db()

router = APIRouter(prefix="/api")


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


@router.get("/health")
def health():
    return {"status": "ok", "service": "HR Chatbot", "healthy": True}


@router.post("/chat")
def chat(payload: ChatRequest):
    try:
        result = ask(payload.question)
        write_audit(payload.question, result["source_used"], result.get("trace", []))
        return {
            "answer": result["answer"],
            "source_used": result["source_used"],
            "trace": result.get("trace", []),
            "citations": result.get("citations", []),
            "rewritten_query": result.get("current_query", payload.question),
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/ingest")
async def ingest(file: UploadFile = File(...), x_admin_key: str = Header(default="")):
    admin_api_key = os.getenv("ADMIN_API_KEY")
    if not admin_api_key or x_admin_key != admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")

    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="A filename is required")

    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported: {', '.join(sorted(SUPPORTED))}")

    upload_dir = Path("uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(filename).name
    dest.write_bytes(await file.read())
    docs = load_file(dest)
    chunks = chunk_documents(docs)
    ids = add_documents(chunks)
    return {"message": "Document indexed", "file": dest.name, "chunks": len(chunks), "ids_created": len(ids)}