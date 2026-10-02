# The agent: a LangGraph graph that decides, per question, which path to take.
#
#   START -> decide --"text"---> retrieve_text ----------------------------------------> generate -> END
#                   \--"image"--> retrieve_images -> inspect_images -> describe_images --/
#
# Each box is a "node" (a plain function). Each arrow is an "edge". The "decide" node writes
# its choice into the state, and the conditional edge reads it to pick the next node.
#
# Try it:  python -m agent.graph "Which photo shows a cat wearing a hat?"
import logging
import re
import sys
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

import config
from agent.generate import NO_USAGE, answer_about_photo, answer_from_sources, keep_good_chunks, text_sources
from agent.llm import ask_for_tool_call, ask_model
from agent.vision_tool import find_objects, object_labels, tool_description
from retrieval.retriever import PgVectorRetriever

log = logging.getLogger("agent")

SHORTLIST_SIZE = 1  # images described by the vision-language model; on the eval set, the top YOLO-confirmed
                    # search result was right 10/13 times, and letting the model pick among 3 did worse

DECIDE_PROMPT = """Decide how to answer the question below.
Reply "image" if answering it means looking at photos: for example it asks to find, show or describe a photo or picture.
Reply "text" if it can be answered from written documents: facts, history, numbers, definitions.
Reply with exactly one word: image or text.

Question: {question}"""

DESCRIBE_PROMPT = "Describe this image in two sentences. Mention the main objects, what they are doing, and their colours."

TOOL_PROMPT = """Question: {question}

Call find_objects with the objects that a matching photo must contain.
Only use labels from the allowed list. Use "person" for people, and the closest label for similar words
(for example: puppy -> dog, kitten -> cat, moped -> motorcycle, plane -> airplane)."""


# Tool inputs are checked in code, not left to the model (it once asked for ["person", "kitten", "hat"]:
# "kitten" and "hat" aren't YOLO labels, and nobody had asked about people)
SYNONYMS = {
    "kitten": "cat", "kitty": "cat", "puppy": "dog", "plane": "airplane", "aeroplane": "airplane",
    "jet": "airplane", "moped": "motorcycle", "motorbike": "motorcycle", "scooter": "motorcycle",
    "bike": "bicycle", "people": "person", "man": "person", "woman": "person", "child": "person",
}
PEOPLE_WORDS = {"person", "people", "man", "men", "woman", "women", "child", "children", "kid", "kids",
                "boy", "girl", "rider", "riders", "someone", "player", "players"}


def clean_targets(objects, question):
    """Turn the model's object list into labels YOLO knows, keeping "person" only if the question is about people."""
    if isinstance(objects, str):  # small models sometimes send "dog, frisbee" instead of a list
        objects = objects.split(",")
    labels = set(object_labels())
    question_words = set(re.findall(r"[a-z]+", question.lower()))
    targets = []
    for name in objects:
        name = name.strip().lower()
        name = SYNONYMS.get(name, name)
        if name == "person" and not question_words & PEOPLE_WORDS:
            continue
        if name in labels and name not in targets:
            targets.append(name)
    return targets


class AgentState(TypedDict, total=False):
    # Everything the nodes share. Each node reads what it needs and returns the fields it changes.
    question: str
    path: str            # "text" or "image", chosen by decide
    chunks: list         # text search results
    images: list         # image search results
    targets: list        # objects the vision tool looked for, chosen by the language model
    shortlist: list      # images picked for the vision-language model
    sources: list        # numbered sources given to the answering model
    answer: str
    citations: list
    usage: dict          # tokens used so far, summed over every model call


_retriever = None


def retriever():
    global _retriever
    if _retriever is None:
        _retriever = PgVectorRetriever()
    return _retriever


def add_usage(total, new):
    total = total or NO_USAGE
    return {key: total[key] + new[key] for key in total}


# ---- Nodes ----

def decide(state):
    reply, usage = ask_model(config.TEXT_MODEL, DECIDE_PROMPT.format(question=state["question"]))
    path = "image" if "image" in reply.lower() else "text"  # anything unclear falls back to the text path
    log.info("decide -> %s path", path)
    return {"path": path, "usage": add_usage(state.get("usage"), usage)}


def retrieve_text(state):
    chunks = keep_good_chunks(retriever().search_text(state["question"], k=config.TEXT_TOP_K))
    log.info("retrieve_text -> %d chunks", len(chunks))
    return {"chunks": chunks, "sources": text_sources(chunks)}


def retrieve_images(state):
    images = retriever().search_images(state["question"], k=config.IMAGE_TOP_K)
    log.info("retrieve_images -> %d images", len(images))
    return {"images": images}


def inspect_images(state):
    # 1. The language model is shown the tool and asks for it: "find_objects(objects=['dog', 'frisbee'])"
    inputs, usage = ask_for_tool_call(config.TEXT_MODEL, TOOL_PROMPT.format(question=state["question"]), tool_description())
    targets = clean_targets((inputs or {}).get("objects", []), state["question"])  # never trust tool inputs blindly

    # 2. Our code runs the real tool. With no valid targets it still labels the images, keeping search order.
    shortlist = find_objects(state["images"], targets, keep=SHORTLIST_SIZE)
    log.info("inspect_images -> YOLO looked for %s, kept %d of %d images", targets, len(shortlist), len(state["images"]))
    return {"targets": targets, "shortlist": shortlist, "usage": add_usage(state.get("usage"), usage)}


def describe_images(state):
    # The vision-language model looks at each shortlisted image; its description becomes a numbered source
    sources, usage = [], state.get("usage")
    for image in state["shortlist"]:
        # max_tokens stops moondream from rambling (one description once ran past 1,000 tokens)
        description, used = ask_model(config.VLM_MODEL, DESCRIBE_PROMPT, image_paths=[image["path"]], max_tokens=80)
        usage = add_usage(usage, used)
        if image.get("objects"):  # what YOLO found: facts from the detector, not the VLM's wording
            found = ", ".join(f"{label} ({confidence})" for label, confidence in image["objects"].items())
            description += f" Objects detected: {found}."
        if image.get("ocr_text"):
            description += f" Text visible in the image: {image['ocr_text']}"
        sources.append({
            "label": f"image {image['image_id']}",
            "text": description,
            "cite": {"image_id": image["image_id"], "path": image["path"]},
        })
    log.info("describe_images -> %d descriptions", len(sources))
    return {"sources": sources, "usage": usage}


def generate(state):
    if state["path"] == "image":
        answer, citations, usage = answer_about_photo(state["question"], state.get("sources", []))
    else:
        answer, citations, usage = answer_from_sources(state["question"], state.get("sources", []))
    return {"answer": answer, "citations": citations, "usage": add_usage(state.get("usage"), usage)}


# ---- The graph ----

def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("retrieve_text", retrieve_text)
    graph.add_node("retrieve_images", retrieve_images)
    graph.add_node("inspect_images", inspect_images)
    graph.add_node("describe_images", describe_images)
    graph.add_node("generate", generate)

    graph.add_edge(START, "decide")
    graph.add_conditional_edges("decide", lambda state: state["path"], {"text": "retrieve_text", "image": "retrieve_images"})
    graph.add_edge("retrieve_text", "generate")
    graph.add_edge("retrieve_images", "inspect_images")
    graph.add_edge("inspect_images", "describe_images")
    graph.add_edge("describe_images", "generate")
    graph.add_edge("generate", END)
    return graph.compile()


agent = build_agent()


def ask(question):
    return agent.invoke({"question": question})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="  %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    question = " ".join(sys.argv[1:]) or "What do giraffes eat?"
    result = ask(question)
    print("Question:", question)
    print("Path:    ", result["path"])
    print("Answer:  ", result["answer"])
    for c in result["citations"]:
        print(f"  [{c['n']}]", c.get("image_id") or f"{c['source']}, page {c['page']}")
    print("Tokens:  ", result["usage"])
