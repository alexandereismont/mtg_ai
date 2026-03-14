"""
MTG Rules RAG — FastAPI server
==============================
Exposes the RAG query pipeline over HTTP so the React frontend can call it.

Endpoints:
    POST /api/query   {"question": "..."}  → SSE stream of answer chunks

Usage:
    python server.py          # starts on http://localhost:8000
    # or
    uvicorn server:app --reload
"""

import json
from contextlib import asynccontextmanager

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mtg_rag import OLLAMA_MODEL, SYSTEM_PROMPT, load_index, retrieve

# ---------------------------------------------------------------------------
# App lifecycle — load the index once at startup
# ---------------------------------------------------------------------------

_index: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index
    try:
        _index = load_index()
        print("[server] TF-IDF index loaded.")
    except RuntimeError as e:
        print(f"[server] WARNING: {e}")
        print("[server] Run `python mtg_rag.py setup` then restart.")
    yield


app = FastAPI(title="MTG Rules RAG", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class QueryRequest(BaseModel):
    question: str


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def _sse(data: str) -> str:
    return f"data: {data}\n\n"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.post("/api/query")
async def query_endpoint(req: QueryRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    if _index is None:
        raise HTTPException(
            status_code=503,
            detail="Index not loaded. Run `python mtg_rag.py setup` and restart the server.",
        )

    hits = retrieve(req.question, _index)

    if not hits:
        context = "(No closely matching rules found.)"
    else:
        parts = [f"[Rule {c['id']}]\n{c['text']}" for c in hits]
        context = "\n\n---\n\n".join(parts)

    prompt = (
        f"Relevant Rules:\n\n{context}\n\n"
        f"---\n\nQuestion: {req.question}\n\n"
        "Answer based on the rules above, citing rule numbers."
    )

    def generate():
        stream = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            stream=True,
        )
        for chunk in stream:
            token = chunk["message"]["content"]
            if token:
                yield _sse(json.dumps({"token": token}))
        yield _sse("[DONE]")

    return StreamingResponse(generate(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
