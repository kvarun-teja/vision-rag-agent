# Put a photo and some captions into the same vector space and see which caption matches best.
# Run:  python scripts/image_embeddings.py data/raw/<some photo>.jpg
import sys

from PIL import Image
from sentence_transformers import SentenceTransformer

# CLIP was trained on photos with their captions, so it can embed both into the same space
# (downloads about 600 MB the first time)
model = SentenceTransformer("clip-ViT-B-32")

image = Image.open(sys.argv[1])
captions = [
    "a photo of a giraffe",
    "a photo of a pizza",
    "a photo of a train",
    "a photo of a dog",
]

image_vector = model.encode(image, normalize_embeddings=True)
caption_vectors = model.encode(captions, normalize_embeddings=True)
print("The photo became", image_vector.shape[0], "numbers; so did each caption")
print()

# Same cosine similarity as before, but now between a photo and text
scores = caption_vectors @ image_vector
for caption, score in sorted(zip(captions, scores), key=lambda pair: -pair[1]):
    print(f"{score:.3f}  {caption}")
