import time
import os

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

INDEX_NAME = "hr-agentic-rag-kb"
NAMESPACE = "agentic-rag"
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072
TOP_K = 4

_embeddings = None
_vectorstore = None

load_dotenv()


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing")
        _embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL,
        )
    return _embeddings


def ensure_index():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")

    pc = Pinecone(api_key=api_key)
    index_names = [index_info["name"] for index_info in pc.list_indexes()]

    if INDEX_NAME not in index_names:
        pc.create_index(
            name=INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        while not pc.describe_index(INDEX_NAME).status["ready"]:
            time.sleep(1)

    return pc.Index(INDEX_NAME)


def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = PineconeVectorStore(
            index=ensure_index(),
            embedding=get_embeddings(),
            namespace=NAMESPACE,
        )
    return _vectorstore


def get_retriever():
    return get_vectorstore().as_retriever(
        search_kwargs={"k": TOP_K, "namespace": NAMESPACE}
    )


def add_documents(chunks):
    return get_vectorstore().add_documents(chunks)