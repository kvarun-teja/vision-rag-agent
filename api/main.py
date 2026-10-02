# The web API.
#   POST /query   ask the agent a question, get a cited answer back
#   GET  /health  check that the database and Ollama are reachable
#
# Run:  uvicorn api.main:app --port 8000
# Then open http://localhost:8000/docs to try it in the browser.
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import config
from agent.graph import ask
from ingestion import store

MAX_QUESTION_CHARS = 500
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

    return {
        "request_id": request_id,
        "answer": result["answer"],
        "citations": result["citations"],
        "path": result["path"],
        "tokens": tokens,
        "latency_ms": entry["latency_ms"],
        "cost_usd": entry["cost_usd"],
    }


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
