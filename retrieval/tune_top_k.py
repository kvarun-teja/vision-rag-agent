# How often does the right source show up in the top k results? Uses the eval set as the answer key.
#   text questions:  is a chunk from the expected PDF among the top k chunks?
#   image questions: is the expected photo among the top k images?
# Run:  python -m retrieval.tune_top_k
import json

from retrieval.retriever import PgVectorRetriever

K_VALUES = [1, 3, 5, 10]

retriever = PgVectorRetriever()
questions = [q for q in json.load(open("eval/eval_set.json")) if q["type"] in ("text", "image")]

hits = {"text": {k: 0 for k in K_VALUES}, "image": {k: 0 for k in K_VALUES}}
counts = {"text": 0, "image": 0}
misses = []

for q in questions:
    counts[q["type"]] += 1
    if q["type"] == "text":
        found = [hit["source"].split("/")[-1] for hit in retriever.search_text(q["question"], k=max(K_VALUES))]
    else:
        found = [hit["image_id"] for hit in retriever.search_images(q["question"], k=max(K_VALUES))]

    correct = [q["source"]] + q.get("also_correct", [])  # some questions have more than one right answer
    positions = [found.index(c) + 1 for c in correct if c in found]  # 1 = top result
    position = min(positions) if positions else None
    for k in K_VALUES:
        if position and position <= k:
            hits[q["type"]][k] += 1
    if not position or position > 3:
        misses.append((q["id"], q["question"], position))

for kind in ("text", "image"):
    scores = "  ".join(f"top-{k}: {hits[kind][k]}/{counts[kind]}" for k in K_VALUES)
    print(f"{kind:<6} {scores}")

print("\nNot in the top 3:")
for qid, question, position in misses:
    print(f"  #{qid} {question}  -> position {position or 'not in top 10'}")
