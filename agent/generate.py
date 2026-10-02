# Answer a question from retrieved sources, with citations.
#   question -> retrieve chunks -> drop weak matches -> numbered-sources prompt -> qwen2.5:3b -> answer + citations
# The agent (agent/graph.py) reuses answer_from_sources() for photo descriptions too.
#
# Try it:  python -m agent.generate "What do giraffes eat?"
import re
import sys

import config
from agent.llm import ask_model
from retrieval.retriever import PgVectorRetriever

MIN_SCORE = 0.70     # answerable eval questions scored 0.74+, unanswerable ones 0.65 or less
SCORE_MARGIN = 0.05  # also drop chunks this much weaker than the best one: near-misses confused the model's citations
NO_ANSWER = "I couldn't find this in the documents."
NO_USAGE = {"prompt_tokens": 0, "answer_tokens": 0}

SYSTEM_PROMPT = f"""You answer questions using ONLY the numbered sources you are given.
Rules:
- Use only facts stated in the sources. Never use outside knowledge.
- After each fact, cite the source number in square brackets, like [1] or [2].
- If the sources do not contain the answer, reply exactly: {NO_ANSWER}
- Keep the answer short: one to three sentences."""


# The image path hands over ONE photo: the best search result that the object detector confirmed.
# What we learned tuning this on the eval set (Days 8 and 11):
#   - given three photos to choose from, this small model judged too literally and declined good matches
#   - with the decline sentence anywhere in the prompt, it copied that sentence out for 12 of 12 photos
# So choosing (and declining when nothing matches) is done by search + YOLO, and the model only describes.
IMAGE_SYSTEM_PROMPT = """You are shown the description of one photo. It is the best match for the user's question:
it was found by image search and checked by an object detector.
Answer the user's question about this photo in one or two full sentences, using only the description
and its "Objects detected" list (that list comes from the detector, so trust it).
End your answer with [1]."""


def keep_good_chunks(chunks):
    """Keep chunks that clear the fixed cutoff AND are close to the best match.
    Example: for "how many teeth does a dog have?" the dog chunk scored 0.77 and four elephant/horse
    teeth chunks scored about 0.70; the model took the fact from the dog chunk but cited a horse one."""
    good = [c for c in chunks if c["score"] >= MIN_SCORE]
    if not good:
        return []
    best = max(c["score"] for c in good)
    return [c for c in good if c["score"] >= best - SCORE_MARGIN]


def text_sources(chunks):
    # Each source has: a label the model sees, the text, and what to show as the citation
    return [
        {
            "label": f"{c['doc_id']}, page {c['page']}",
            "text": c["text"],
            "cite": {"source": c["source"], "page": c["page"], "chunk_id": c["chunk_id"]},
        }
        for c in chunks
    ]


def build_prompt(question, sources):
    # Number each source so the model can point at it: [1], [2], ...
    numbered = "\n\n".join(f"[{n}] ({s['label']}) {s['text']}" for n, s in enumerate(sources, start=1))
    # Small models pay most attention to the end of the prompt, so the citation rule is repeated here
    return (
        f"Sources:\n{numbered}\n\n"
        f"Question: {question}\n"
        f"Answer in one to three sentences, using only the sources, and put the source number like [1] after each fact.\n"
        f"Answer:"
    )


def find_citations(answer, sources):
    # Turn every [n] in the answer back into the source it points at (ignoring numbers that don't exist)
    numbers = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(sources)})
    return [{"n": n, **sources[n - 1]["cite"]} for n in numbers]


def answer_from_sources(question, sources, system=SYSTEM_PROMPT):
    """Ask the text model to answer from numbered sources. Returns (answer, citations, usage)."""
    if not sources:
        return NO_ANSWER, [], NO_USAGE  # grounding rule 1: nothing to go on -> don't ask, decline
    answer, usage = ask_model(config.TEXT_MODEL, build_prompt(question, sources), system=system)
    if answer.startswith(NO_ANSWER):
        return NO_ANSWER, [], usage  # a "can't find it" answer shouldn't point at any sources
    return answer, find_citations(answer, sources), usage


def answer_about_photo(question, sources):
    """The image path's answer: one photo description in, one or two sentences out."""
    if not sources:
        return NO_ANSWER, [], NO_USAGE  # YOLO found no photo with the right objects
    photo = sources[0]
    prompt = (
        f"Photo description [1]: {photo['text']}\n\n"
        f"Question: {question}\n"
        f"Write a full sentence that answers the question about this photo, then end with [1].\n"
        f"Answer:"
    )
    answer, usage = ask_model(config.TEXT_MODEL, prompt, system=IMAGE_SYSTEM_PROMPT)
    if len(re.sub(r"\[\d+\]", "", answer).split()) < 4:
        # Safety net: the model once replied with just "[1]". The description itself is a grounded answer.
        answer = f"{photo['text']} [1]"
    return answer, find_citations(answer, sources), usage


def answer_question(question, retriever):
    """The straight-line text pipeline. Returns answer, citations, the context it was given, and token usage."""
    chunks = keep_good_chunks(retriever.search_text(question, k=config.TEXT_TOP_K))
    answer, citations, usage = answer_from_sources(question, text_sources(chunks))
    return {"answer": answer, "citations": citations, "contexts": [c["text"] for c in chunks], "usage": usage}


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What do giraffes eat?"
    result = answer_question(question, PgVectorRetriever())
    print("Question:", question)
    print("Answer:  ", result["answer"])
    for c in result["citations"]:
        print(f"  [{c['n']}] {c['source']}, page {c['page']}")
