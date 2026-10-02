# Run the eval set through the agent and score the answers two ways:
#   1. Our own checks (exact, no model needed): right path? right source cited? declined when it should?
#   2. RAGAS (a language model acts as the grader): faithfulness, answer relevancy, context precision and recall
# Results are saved to eval/results.json.
#
# Run:  python -m eval.run_eval
import json
import re
import time
import warnings
from pathlib import Path

from langchain_core.embeddings import Embeddings
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

import config
from agent.generate import NO_ANSWER
from agent.graph import ask
from ingestion.embed import QUERY_DEVICE, get_model

warnings.filterwarnings("ignore")  # RAGAS prints many deprecation warnings
RESULTS_FILE = Path("eval/results.json")
METRICS = [faithfulness, answer_relevancy, context_precision, context_recall]


class BGEEmbeddings(Embeddings):
    """RAGAS calls embed_query() and embed_documents(); this gives it those, backed by our BGE model.
    (The attribute is called "encoder", not "model": RAGAS reads "model" and expects a name, not a model.)"""

    def __init__(self):
        self.encoder = get_model(config.TEXT_EMBED_MODEL, device=QUERY_DEVICE)

    def embed_documents(self, texts):
        return self.encoder.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, text):
        return self.encoder.encode(text, normalize_embeddings=True).tolist()


def strip_citations(text):
    # The [1] markers are ours, not part of the answer's content; the small grader counted them as claims
    return re.sub(r"\s*\[\d+\]", "", text).strip()


def run_agent(items):
    rows = []
    for item in items:
        start = time.time()
        result = ask(item["question"])
        cited = [c.get("image_id") or c["source"].split("/")[-1] for c in result["citations"]]
        correct = [item["source"]] + item.get("also_correct", []) if item["source"] else []
        declined = result["answer"] == NO_ANSWER
        row = {
            "id": item["id"],
            "type": item["type"],
            "question": item["question"],
            "expected_answer": item["expected_answer"],
            "path": result["path"],
            "answer": result["answer"],
            "cited": cited,
            "contexts": [s["text"] for s in result.get("sources", [])],
            "right_path": result["path"] == ("image" if item["type"] == "image" else "text"),
            # for answerable questions: was a correct source cited? for "none" questions: did it decline?
            "right_outcome": declined if item["type"] == "none" else bool(set(cited) & set(correct)),
            "seconds": round(time.time() - start, 1),
        }
        rows.append(row)
        print(f"#{row['id']:<2} {row['type']:<5} {'OK ' if row['right_outcome'] else 'MISS'} {row['seconds']:>5}s  {row['answer'][:70]}")
    return rows


def add_ragas_scores(rows):
    # RAGAS needs a context to grade against, so "none" questions (and answers with no context) are left out
    scored = [r for r in rows if r["type"] != "none" and r["contexts"]]
    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=strip_citations(r["answer"]),
            reference=r["expected_answer"],
        )
        for r in scored
    ]
    # Ollama also speaks the OpenAI API format at /v1, so the standard OpenAI client can talk to our local model
    judge = ChatOpenAI(base_url=f"{config.OLLAMA_URL}/v1", api_key="ollama", model=config.TEXT_MODEL, temperature=0)
    result = evaluate(
        EvaluationDataset(samples=samples),
        metrics=METRICS,
        llm=judge,
        embeddings=BGEEmbeddings(),
        run_config=RunConfig(timeout=300, max_workers=2),
        show_progress=False,
    )
    table = result.to_pandas()
    for row, (_, scores) in zip(scored, table.iterrows()):
        # NaN (the grader's reply couldn't be read) becomes None; NaN is the only value not equal to itself
        row["ragas"] = {m.name: (round(float(scores[m.name]), 3) if scores[m.name] == scores[m.name] else None) for m in METRICS}


def average(values):
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else None


def summarise(rows):
    summary = {"routing": f"{sum(r['right_path'] for r in rows)}/{len(rows)}"}
    for kind in ("text", "image", "none"):
        group = [r for r in rows if r["type"] == kind]
        summary[kind] = {
            "right_outcome": f"{sum(r['right_outcome'] for r in group)}/{len(group)}",
            "avg_seconds": average([r["seconds"] for r in group]),
        }
        if kind != "none":
            graded = [r for r in group if "ragas" in r]
            summary[kind]["ragas_graded"] = len(graded)
            for m in METRICS:
                summary[kind][m.name] = average([r["ragas"][m.name] for r in graded])
    return summary


if __name__ == "__main__":
    items = json.load(open("eval/eval_set.json"))
    print(f"Running {len(items)} questions through the agent...")
    rows = run_agent(items)
    print("\nGrading with RAGAS (this takes a while)...")
    add_ragas_scores(rows)
    summary = summarise(rows)
    RESULTS_FILE.write_text(json.dumps({"summary": summary, "questions": rows}, indent=2))
    print("\n" + json.dumps(summary, indent=2))
    print(f"\nSaved to {RESULTS_FILE}")
