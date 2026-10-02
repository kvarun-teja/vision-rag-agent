# Turn sentences into vectors (embeddings) and see that similar meanings get similar vectors.
# Run:  python scripts/text_embeddings.py
from sentence_transformers import SentenceTransformer

# BGE-small: a small, fast text embedding model (downloads about 130 MB the first time)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

sentences = [
    "A giraffe is eating leaves from a tall tree.",
    "A tall animal with a long neck feeds on branches.",
    "A pizza with cheese and tomatoes on a plate.",
]

# normalize_embeddings=True makes every vector length 1, so comparing them is a simple dot product
vectors = model.encode(sentences, normalize_embeddings=True)
print("Each sentence became", vectors.shape[1], "numbers")
print("First 5 numbers of sentence 1:", vectors[0][:5].round(3))
print()


def similarity(a, b):
    # Cosine similarity: close to 1 = same meaning, lower = less related
    return float(vectors[a] @ vectors[b])


print("1 vs 2 (giraffe vs long-neck animal): %.3f" % similarity(0, 1))
print("1 vs 3 (giraffe vs pizza):           %.3f" % similarity(0, 2))
print("2 vs 3 (long-neck animal vs pizza):  %.3f" % similarity(1, 2))
