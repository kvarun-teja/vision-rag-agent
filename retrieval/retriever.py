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


class Retriever:
    def search_text(self, question, k=5):
        raise NotImplementedError

    def search_images(self, question, k=5):
        raise NotImplementedError


class PgVectorRetriever(Retriever):
    def __init__(self):
        self.conn = store.connect()

    def search_text(self, question, k=5):
        vector = embed_query_for_text(question)
        # <=> is pgvector's cosine distance (0 = same direction). 1 - distance = similarity score.
        rows = self.conn.execute(
            """SELECT chunk_id, doc_id, source, page, content, 1 - (embedding <=> %s) AS score
               FROM text_chunks
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (vector, vector, k),
        ).fetchall()
        return [
            {"chunk_id": r[0], "doc_id": r[1], "source": r[2], "page": r[3], "text": r[4], "score": round(float(r[5]), 3)}
            for r in rows
        ]

    def search_images(self, question, k=5):
        vector = embed_query_for_images(question)
        rows = self.conn.execute(
            """SELECT image_id, source, page, path, origin, ocr_text, 1 - (embedding <=> %s) AS score
               FROM images
               ORDER BY embedding <=> %s
               LIMIT %s""",
            (vector, vector, k),
        ).fetchall()
        return [
            {"image_id": r[0], "source": r[1], "page": r[2], "path": r[3], "origin": r[4], "ocr_text": r[5],
             "score": round(float(r[6]), 3)}
            for r in rows
        ]


if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "What do giraffes eat?"
    retriever = PgVectorRetriever()

    print(f"Question: {question}\n\nTop text chunks:")
    for hit in retriever.search_text(question, k=3):
        print(f"  {hit['score']:.3f}  {hit['doc_id']} p{hit['page']}: {hit['text'][:100]}...")

    print("\nTop images:")
    for hit in retriever.search_images(question, k=3):
        print(f"  {hit['score']:.3f}  {hit['image_id']}  ({hit['origin']})")
