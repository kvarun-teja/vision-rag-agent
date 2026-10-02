# Vision RAG Agent

Ask questions in plain English over a mix of **documents and photos**, and get short answers that **cite their sources**: a PDF page or a specific photo.

An agent decides for each question how to answer it:

- **Text questions** ("Who flew the first airplane?") search the PDF text and answer from the best passages.
- **Image questions** ("Which photo shows a dog holding a frisbee?") search the photos, use an **object detector (YOLO) as a tool** to keep only photos that really contain the right objects, then have a **vision-language model** describe the best one.
- **Questions the data can't answer** get "I couldn't find this in the documents" instead of a made-up answer.

Everything runs **locally on a 4 GB laptop GPU**: no API keys, no data leaves the machine. It is served as a **FastAPI** service with request logging and a guardrail, and measured with **RAGAS** plus exact checks on a hand-built eval set.

## Example

```
$ curl -s -X POST localhost:8000/query -H "Content-Type: application/json" \
       -d '{"question": "Where and when was the steam locomotive invented?"}'
{"answer": "The steam locomotive was invented in the United Kingdom in 1802 [1].",
 "citations": [{"n": 1, "source": "data/raw/Train.pdf", "page": 3}],
 "path": "text", "tokens": 1404, "latency_ms": 1589, ...}

$ curl -s -X POST localhost:8000/query -H "Content-Type: application/json" \
       -d '{"question": "Which photo shows a cat wearing a hat, and what colour is the hat?"}'
{"answer": "This photo shows a black and white cat wearing a pink knit hat ... [1]",
 "citations": [{"n": 1, "image_id": "000000286708.jpg"}],
 "path": "image", ...}
```

## Architecture

```
INGESTION (offline, once)
  data/raw/ ──► router ──┬── PDF   ──► PyMuPDF: page text + embedded pictures ─┐
                         ├── photo ──► loaded directly (skips the PDF reader) ──┼─► OCR (only images that contain text)
                         └── other ──► skipped and logged                       ┘
            ──► chunk (≈150 words, sentence boundaries, 1-sentence overlap)
            ──► fingerprint (SHA-256) and skip anything already stored
            ──► embed: text with BGE-small (384 numbers), images with CLIP (512 numbers)
            ──► store in Postgres + pgvector (two tables: text_chunks, images)

SERVING (per question)
  POST /query ──► guardrail (empty / >500 characters → HTTP 400)
              ──► LangGraph agent:
                    decide ──"text"──► search chunks ──► keep strong matches ──────────────────────────┐
                           ──"image"─► search photos (CLIP) ──► YOLO tool keeps photos with the right   │
                                       objects ──► moondream describes the best one ────────────────────┤
                                                                                                         ▼
                                                                     qwen2.5:3b answers from the sources, with [n] citations
              ──► log one JSON line: request id, path, tokens, latency, cost

DATA
  pgvector (vectors + labels) · data/ (raw files, extracted pictures) · logs/requests.jsonl
```

**Models** (all through [Ollama](https://ollama.com) or sentence-transformers, all local):

| Job | Model | Why |
|---|---|---|
| Text embeddings | `BAAI/bge-small-en-v1.5` | Small and fast; strong retrieval quality |
| Image + image-search embeddings | `clip-ViT-B-32` | Puts photos and text in the same vector space |
| Vision tool | YOLO11m (COCO, 80 objects) | ~24 ms per image; the nano version missed small objects |
| Describing photos | `moondream` (1.8B) | Fits on a 4 GB GPU; LLaVA 7B needs 5–6 GB |
| Deciding, tool calls, answers | `qwen2.5:3b` | Follows instructions and supports tool calling; moondream can't answer text-only questions |

## Results

Measured on [eval/eval_set.json](eval/eval_set.json): 30 hand-checked questions (15 text, 13 image, 2 not answerable from the data). Full per-question output: [eval/results.json](eval/results.json).

**Exact checks** (no model involved, so these are the numbers to trust):

| | Text (15) | Image (13) | Not answerable (2) |
|---|---|---|---|
| Agent picked the right path | 15/15 | 13/13 | 2/2 |
| Cited a correct source / declined | **15/15** | **9/13** | **2/2** |
| Average time per question | 0.7 s | 2.2 s | 0.1 s |

**RAGAS** (graded locally by `qwen2.5:3b`):

| | Text | Image |
|---|---|---|
| Faithfulness (claims backed by the context) | 0.60 | 0.38 |
| Answer relevancy (answers the question asked) | 0.78 | 0.58 |
| Context precision (retrieved context is useful) | 0.76 | 0.46 |
| Context recall (context holds the needed facts) | 0.93 | 0.23 |

**Before vs. after the last round of fixes** (image-answer prompt, a relative score margin for text chunks, cleaning tool inputs in code): text went from 14/15 to 15/15, image stayed at 9/13 (one question fixed, another lost), and average time per image question dropped from 5.0 s to 2.2 s. Image RAGAS scores rose (faithfulness 0.21 → 0.38, relevancy 0.42 → 0.58).

**How much to trust RAGAS here:** re-grading the *same unchanged* text answers moved the scores by up to ±0.06, so differences smaller than that are grader noise. Image context recall is low partly because the "context" is moondream's description, while the reference answer is a COCO caption written in different words. Three of the four image misses are retrieval misses: the right photo never reached the top 10.

## Design choices

- **pgvector behind a retriever interface.** [retrieval/retriever.py](retrieval/retriever.py) defines `search_text()` / `search_images()`; the agent and API only call those. Postgres keeps vectors and their labels (file, page, OCR text) in one place; switching to Qdrant or OpenSearch means writing one new class.
- **YOLO before the vision-language model.** Image search (CLIP) is fuzzy; YOLO is precise and cheap. Screening 10 candidates with YOLO costs ~0.25 s, while describing one photo with moondream costs ~0.5–1 s and ~740 of its 2,048 tokens. Describing only the top YOLO-confirmed photo was both cheaper and more accurate on the eval set than describing the top search result or letting the language model pick between three.
- **The agent calls the tool through real tool calling.** qwen is shown a tool description and replies with `find_objects(objects=["dog", "frisbee"])`; our code validates those inputs against YOLO's labels and runs the detector.
- **Two layers of grounding.** If no chunk scores above a measured cutoff, the model is never called and the system declines. Otherwise the prompt allows only the numbered sources and requires `[n]` citations, which are mapped back to files and pages (numbers that don't exist are ignored).
- **Thresholds come from measurements, not guesses.** The 0.70 cutoff sits in the gap between answerable (≥ 0.74) and unanswerable (≤ 0.65) questions; top-k came from [retrieval/tune_top_k.py](retrieval/tune_top_k.py).

## Limitations

- **The RAGAS grader is the same 3B model.** It sometimes gives 0 faithfulness to answers copied word for word from the source, so the exact checks (right path, right source cited, declined when it should) are the more reliable numbers.
- **CLIP confuses similar photos** (a moped vs. a row of motorcycles), and **YOLO can't see colours**, so colour-specific image questions can pick the wrong photo.
- **Small models are literal.** The image-answer prompt was tuned on the eval set; with only 13 image questions, treat image scores as indicative.
- **Single-user.** One shared database connection and models loaded in one process; fine for a demo, not for heavy traffic.

## How to run it

**You need:** Linux with an NVIDIA GPU (4 GB is enough), [conda](https://docs.conda.io), [Docker](https://docs.docker.com/engine/install/), [Ollama](https://ollama.com), and Tesseract (`sudo apt install tesseract-ocr`).

```bash
# 1. Python environment (pick the PyTorch CUDA build your driver supports: see nvidia-smi and pytorch.org)
conda create -n RAG python=3.11 -y && conda activate RAG
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 2. Local models
ollama pull moondream
ollama pull qwen2.5:3b

# 3. Database (reachable only from this computer)
docker run -d --name rag-postgres -e POSTGRES_PASSWORD=localdev -p 127.0.0.1:5432:5432 pgvector/pgvector:pg16

# 4. Settings, data, ingestion
cp .env.example .env
python scripts/download_data.py      # 10 Wikipedia PDFs + 200 COCO photos (~300 MB)
python -m ingestion.run              # ~45 s on a GPU

# 5. Serve
uvicorn api.main:app --port 8000     # then open http://localhost:8000/docs
```

**Or with Docker Compose** (API on the CPU, Ollama still on the host GPU; after steps 2 and 4's download):

```bash
docker compose up -d db
docker compose run --rm api python -m ingestion.run   # first time only
docker compose up -d api
```

**Tests and evaluation:**

```bash
python -m pytest tests/              # ingestion checks
python -m retrieval.tune_top_k       # how often the right source is in the top k
python -m eval.run_eval              # agent + RAGAS on the 30 eval questions (~10 min)
```

## Project layout

```
ingestion/   router, PDF parsing, images, OCR, chunking, embedding, storing, run.py (the pipeline)
retrieval/   the retriever interface + pgvector implementation, top-k tuning
agent/       llm.py (Ollama calls), generate.py (grounded answers), vision_tool.py (YOLO), graph.py (LangGraph agent)
api/         FastAPI app: /query, /health, guardrail, request log
eval/        eval_set.json, run_eval.py, results.json
scripts/     dataset download and three small concept demos (calling a model, text and image embeddings)
tests/       pytest checks for ingestion
config.py    all settings, read from .env
```

## Data and licences

Text: Wikipedia articles (CC BY-SA 4.0), downloaded as PDFs through the Wikimedia REST API. Photos and captions: [COCO 2017 validation set](https://cocodataset.org) (annotations CC BY 4.0; images under their Flickr licences). The data is downloaded by the script and not stored in this repository.
