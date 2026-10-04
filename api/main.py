# The web API.
#   GET  /                  the web page: upload a document, then ask questions about it
#   POST /upload?name=...   add one document (the request body is the file) and run ingestion on it
#   POST /query             ask the agent a question, get a cited answer back
#   GET  /images/{id}       one stored photo, so the page can show the photo an answer cites
#   GET  /health            check that the database and Ollama are reachable
#
# Run:  uvicorn api.main:app --port 8000
# Then open http://localhost:8000 for the page, or http://localhost:8000/docs to try the API directly.
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from pydantic import BaseModel

import config
from agent.graph import ask
from ingestion import store
from ingestion.router import route_file
from ingestion.run import main as run_ingestion

MAX_QUESTION_CHARS = 500
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB
RAW_FOLDER = Path("data/raw")        # where ingestion reads its input files
PAGE = Path(__file__).parent / "static" / "index.html"
LOG_FILE = Path("logs/requests.jsonl")  # one JSON line per request

logging.basicConfig(level=logging.INFO, format="%(levelname)s:     %(name)s: %(message)s")  # show our info lines
logging.getLogger("httpx").setLevel(logging.WARNING)  # hide one line per model download check
log = logging.getLogger("api")
app = FastAPI(title="Vision RAG Agent")


class Query(BaseModel):
    question: str  # FastAPI checks that the request body has this field and that it is text


def check_question(question):
    """The guardrail: refuse empty or oversized questions before they reach any model."""
    question = question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="The question is empty.")
    if len(question) > MAX_QUESTION_CHARS:
        raise HTTPException(
            status_code=400,
            detail=f"The question is too long ({len(question)} characters; the limit is {MAX_QUESTION_CHARS}).",
        )
    return question


def write_log(entry):
    LOG_FILE.parent.mkdir(exist_ok=True)
    with LOG_FILE.open("a") as f:
        f.write(json.dumps(entry) + "\n")


@app.post("/query")
def query(body: Query):
    request_id = uuid.uuid4().hex[:12]  # a short unique id, so one request can be found in the logs
    start = time.time()
    entry = {"request_id": request_id, "time": datetime.now(timezone.utc).isoformat(timespec="seconds")}

    try:
        question = check_question(body.question)
    except HTTPException as error:
        write_log({**entry, "blocked": error.detail, "question_chars": len(body.question)})
        raise

    result = ask(question)

    usage = result["usage"]
    tokens = usage["prompt_tokens"] + usage["answer_tokens"]
    entry.update({
        "question": question,
        "path": result["path"],
        "prompt_tokens": usage["prompt_tokens"],
        "answer_tokens": usage["answer_tokens"],
        "latency_ms": round((time.time() - start) * 1000),
        "cost_usd": round(tokens / 1000 * config.COST_PER_1K_TOKENS, 6),
        "cited_sources": len(result["citations"]),
    })
    write_log(entry)
    log.info("request %s: %s path, %d tokens, %d ms", request_id, entry["path"], tokens, entry["latency_ms"])

    # Attach the text each citation points at (a passage, or a photo's description), so the page can show it
    for citation in result["citations"]:
        citation["quote"] = result["sources"][citation["n"] - 1]["text"]

    return {
        "request_id": request_id,
        "answer": result["answer"],
        "citations": result["citations"],
        "path": result["path"],
        "tokens": tokens,
        "latency_ms": entry["latency_ms"],
        "cost_usd": entry["cost_usd"],
    }


@app.get("/")
def page():
    return FileResponse(PAGE)


@app.post("/upload")
async def upload(name: str, request: Request):
    """Save one document into data/raw and run the ingestion pipeline, which only processes what's new."""
    name = Path(name).name  # keep just the file name, so "../../x.pdf" can't write outside data/raw
    if name.startswith(".") or route_file(name) not in ("pdf", "text"):
        raise HTTPException(status_code=400, detail="Please upload a PDF, TXT or Markdown file.")
    data = await request.body()  # the request body is the file itself
    if not data:
        raise HTTPException(status_code=400, detail="The file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="The file is too big (the limit is 20 MB).")

    start = time.time()
    path = RAW_FOLDER / name
    path.write_bytes(data)
    await run_in_threadpool(run_ingestion)  # slow, blocking work: run it outside the server's main loop

    with store.connect() as conn:
        row = conn.execute("SELECT pages FROM sources WHERE source_id = %s", (name,)).fetchone()
        if row is None:  # ingestion skips files it can't read
            path.unlink()
            raise HTTPException(status_code=400, detail="Couldn't read this file. Is it a valid PDF?")
        passages = conn.execute(
            "SELECT count(*) FROM text_chunks WHERE source_id = %s AND kind = 'text'", (name,)
        ).fetchone()[0]
        documents, photos = conn.execute(
            "SELECT count(*) FILTER (WHERE file_type <> 'image'), count(*) FILTER (WHERE file_type = 'image') FROM sources"
        ).fetchone()

    seconds = round(time.time() - start, 1)
    write_log({"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "upload": name,
               "bytes": len(data), "passages": passages, "seconds": seconds})
    log.info("upload %s: %d passages, %.1f s", name, passages, seconds)
    return {"name": name, "pages": row[0], "passages": passages, "seconds": seconds,
            "library": {"documents": documents, "photos": photos}}


@app.get("/images/{image_id}")
def image(image_id: str):
    with store.connect() as conn:
        row = conn.execute("SELECT blob_path FROM images WHERE image_id = %s", (image_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="No image with that id.")
    return FileResponse(row[0])


@app.get("/health")
def health():
    status = {}
    try:
        with store.connect() as conn:
            conn.execute("SELECT 1")
        status["database"] = "ok"
    except Exception as error:
        status["database"] = f"down: {error}"
    try:
        requests.get(f"{config.OLLAMA_URL}/api/tags", timeout=5).raise_for_status()
        status["ollama"] = "ok"
    except Exception as error:
        status["ollama"] = f"down: {error}"
    status["status"] = "ok" if status["database"] == status["ollama"] == "ok" else "degraded"
    return status
