# Turn chunks and images into vectors, in batches, on the GPU.
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer

import config

BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# At question time we embed one short question, which the CPU does in milliseconds.
# Keeping those models off the GPU leaves room for Ollama and YOLO on the 4 GB card.
QUERY_DEVICE = "cpu"

_loaded = {}  # each model is loaded once per device, then reused


def get_model(name, device=DEVICE):
    if (name, device) not in _loaded:
        _loaded[(name, device)] = SentenceTransformer(name, device=device)
    return _loaded[(name, device)]


def embed_texts(texts):
    """Text chunks -> 384 numbers each (BGE)."""
    model = get_model(config.TEXT_EMBED_MODEL)
    return model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True)


def embed_query_for_text(question):
    """A question -> BGE vector, for searching text chunks.
    BGE was trained to put this instruction in front of search questions (not in front of the chunks)."""
    model = get_model(config.TEXT_EMBED_MODEL, device=QUERY_DEVICE)
    instruction = "Represent this sentence for searching relevant passages: "
    return model.encode(instruction + question, normalize_embeddings=True)


def embed_query_for_images(question):
    """A question -> CLIP vector, for searching images (CLIP's text side lives in the same space as its images)."""
    model = get_model(config.IMAGE_EMBED_MODEL, device=QUERY_DEVICE)
    return model.encode(question, normalize_embeddings=True)


def embed_images(paths):
    """Image files -> 512 numbers each (CLIP). Opens the files one batch at a time to save memory."""
    model = get_model(config.IMAGE_EMBED_MODEL)
    vectors = []
    for start in range(0, len(paths), BATCH_SIZE):
        batch = [Image.open(p).convert("RGB") for p in paths[start:start + BATCH_SIZE]]
        vectors.extend(model.encode(batch, batch_size=BATCH_SIZE, normalize_embeddings=True))
    return vectors
