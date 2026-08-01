import os
import logging
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Import RAG utilities from existing query module
from query import (
    load_env,
    get_vector_store,
    similarity_search,
    build_prompt_messages,
)

# Import LLM provider abstractions
from llmProvider import get_chat_model, generate_answer

# Import incremental sync from ingest module
from ingest import IndexerConfig, run_incremental_sync

# ---------------------------------------------------------------------------
# FastAPI application setup
# ---------------------------------------------------------------------------
app = FastAPI(title="Vault RAG", description="Local RAG assistant over your Obsidian vault")

origins = ["http://localhost", "http://127.0.0.1", "http://localhost:8000", "http://127.0.0.1:8000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str
    sources: List[str]

class SyncResponse(BaseModel):
    added: int
    changed: int
    deleted: int
    unchanged: int

# ---------------------------------------------------------------------------
# Shared resources — built once at startup, never duplicated
# ---------------------------------------------------------------------------
db_dir, collection_name, embedding_model, llm_config = load_env()

from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

# Construct chat model once at application startup
chat_model = get_chat_model(llm_config)

# Build a shared IndexerConfig for the /sync endpoint
_ingest_config = IndexerConfig.load_from_env()

# ---------------------------------------------------------------------------
# vector_store is kept in a mutable container so it can be refreshed
# after a sync without restarting the server.
# ---------------------------------------------------------------------------
_store: dict = {"db": None}


def _open_vector_store():
    """Open (or reopen) the Chroma collection and store it in _store['db']."""
    _store["db"] = get_vector_store(db_dir, collection_name, embeddings)


def _get_db():
    """Return the current Chroma handle, reopening if it is None."""
    if _store["db"] is None:
        _open_vector_store()
    return _store["db"]


# Initial open at startup
try:
    _open_vector_store()
except RuntimeError as e:
    logging.warning(f"Vector store not ready at startup ({e}) — run ingest.py first.")

# ---------------------------------------------------------------------------
# Sync concurrency guard — module-level boolean lock
# ---------------------------------------------------------------------------
_sync_in_progress: bool = False

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """Answer a question using notes retrieved from the vector store."""
    if not request.question or not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        db = _get_db()
    except Exception as e:
        raise HTTPException(
            status_code=503,
            detail="Vector store unavailable — run ingest.py to build the index first.",
        )

    results = similarity_search(db, request.question)
    if not results:
        raise HTTPException(status_code=404, detail="No relevant notes found in the vault")

    prompt_messages = build_prompt_messages(request.question, results)
    answer = generate_answer(chat_model, prompt_messages, llm_config.provider, llm_config.model)

    sources = list({doc.metadata.get("title", "unknown") for doc, _ in results})
    return AskResponse(answer=answer, sources=sources)


@app.post("/sync", response_model=SyncResponse)
def sync_endpoint():
    """
    Incrementally sync the vault into the Chroma vector store.

    Returns a summary of how many files were added, changed, deleted,
    or left unchanged. Returns 409 if a sync is already in progress.

    After the sync finishes, the in-memory Chroma handle is refreshed
    so the server picks up any collection changes without a restart.
    """
    global _sync_in_progress

    if _sync_in_progress:
        raise HTTPException(
            status_code=409,
            detail="A sync is already in progress. Please wait for it to complete.",
        )

    _sync_in_progress = True
    try:
        summary = run_incremental_sync(
            config=_ingest_config,
            embeddings=embeddings,
            db=None,
        )
    finally:
        _sync_in_progress = False

    try:
        _open_vector_store()
    except RuntimeError:
        _store["db"] = None

    return SyncResponse(**summary)


# ---------------------------------------------------------------------------
# Serve the static UI at /  (must be mounted AFTER all API routes)
# ---------------------------------------------------------------------------
app.mount("/", StaticFiles(directory="static", html=True), name="static")
