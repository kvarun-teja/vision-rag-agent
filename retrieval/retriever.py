# Given a question, find the most relevant text chunks and images.
#
# Retriever describes WHAT any retriever can do. PgVectorRetriever is HOW we do it with Postgres.
# The rest of the project only calls search_text() and search_images(), so swapping the database
# later (Qdrant, OpenSearch...) means writing one new class, without touching the agent or the API.
#
# Try it:  python -m retrieval.retriever "What do giraffes eat?"
import sys

from ingestion import store
from ingestion.embed import embed_query_for_images, embed_query_for_text

CLIP_SHARE = 0.7  # in image search, 70% of the results come from CLIP, the rest from captions/OCR text


class Retriever:
    def search_text(self, question, k=5, kinds=("text",)):
        raise NotImplementedError

    def search_images(self, question, k=5):
        raise NotImplementedError


class PgVectorRetriever(Retriever):
    def __init__(self):
        self.conn = store.connect(autocommit=True)  # read-only: never leave a transaction open

    def search_text(self, question, k=5, kinds=("text",)):
        """Text chunks closest to the question. kinds picks which chunks to search:
        "text" (document text, the default), "caption" and/or "ocr" (text describing an image)."""
        vector = embed_query_for_text(question)
        # <=> is pgvector's cosine distance (0 = same direction). 1 - distance = similarity score.
        rows = self.conn.execute(
            """SELECT chunk_id, source_id, page, kind, image_id, content, 1 - (embedding <=> %s) AS score
               FROM text_chunks
               WHERE kind = ANY(%s)
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (vector, list(kinds), vector, k),
        ).fetchall()
        return [
            {"chunk_id": r[0], "source": r[1], "page": r[2], "kind": r[3], "image_id": r[4], "text": r[5],
             "score": round(float(r[6]), 3)}
            for r in rows
        ]

    def search_images(self, question, k=5):
        """Find images two ways:
          1. CLIP: compare the question with each image's picture vector (the stronger signal)
          2. Text: compare the question with each image's caption and OCR text (BGE)

        The first 70% of the results are CLIP's best, in CLIP's order; the remaining places go to the best
        caption/OCR matches CLIP didn't already pick. Example with k=10: CLIP's top 7, then 3 caption finds.
        (On the eval set this tied CLIP alone, while merging both rankings equally did worse: 8/13 vs 11/13 in the top 3.)"""
        vector = embed_query_for_images(question)
        clip_count = round(k * CLIP_SHARE)
        best = [row[0] for row in self.conn.execute(
            "SELECT image_id FROM images ORDER BY embedding <=> %s LIMIT %s", (vector, clip_count)
        ).fetchall()]

        for hit in self.search_text(question, k=4 * k, kinds=("caption", "ocr")):
            if len(best) >= k:
                break
            if hit["image_id"] not in best:
                best.append(hit["image_id"])

        # Look up the full rows, then return them in the order chosen above
        clip_ids = set(best[:clip_count])
        rows = self.conn.execute(
            """SELECT image_id, source_id, page, blob_path, origin, caption, ocr_text
               FROM images WHERE image_id = ANY(%s)""",
            (best,),
        ).fetchall()
        found = {
            r[0]: {"image_id": r[0], "source": r[1], "page": r[2], "path": r[3], "origin": r[4], "caption": r[5],
                   "ocr_text": r[6], "found_by": "clip" if r[0] in clip_ids else "caption"}
            for r in rows
        }
        return [found[image_id] for image_id in best]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What do giraffes eat?"
    retriever = PgVectorRetriever()

    print(f"Question: {question}\n\nTop text chunks:")
    for hit in retriever.search_text(question, k=3):
        print(f"  {hit['score']:.3f}  {hit['source']} p{hit['page']}: {hit['text'][:100]}...")

    print("\nTop images:")
    for hit in retriever.search_images(question, k=3):
        print(f"  {hit['found_by']:<8} {hit['image_id']}  ({hit['origin']})  {(hit['caption'] or '')[:70]}")
