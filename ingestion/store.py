# Save chunks and images, with their vectors and labels, in Postgres + pgvector.
import hashlib

import psycopg
from pgvector.psycopg import register_vector

import config

# Two tables: one for text chunks, one for images. Each row = a vector + labels about where it came from.
SCHEMA = """
CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id      TEXT PRIMARY KEY,
    doc_id        TEXT NOT NULL,
    source        TEXT NOT NULL,
    page          INTEGER,
    content       TEXT NOT NULL,
    content_hash  TEXT UNIQUE NOT NULL,
    embedding     vector(384) NOT NULL
);

CREATE TABLE IF NOT EXISTS images (
    image_id      TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    page          INTEGER,
    path          TEXT NOT NULL,
    origin        TEXT NOT NULL,
    ocr_text      TEXT,
    content_hash  TEXT UNIQUE NOT NULL,
    embedding     vector(512) NOT NULL
);
"""


def connect():
    conn = psycopg.connect(config.DATABASE_URL)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # switch pgvector on (only does work the first time)
    conn.commit()
    register_vector(conn)  # lets us pass numpy vectors straight into SQL
    return conn


def setup(conn):
    conn.execute(SCHEMA)
    conn.commit()


def add_hashes(chunks, images):
    # A fingerprint of the content: identical content always gives the identical hash
    for chunk in chunks:
        chunk["content_hash"] = hashlib.sha256(chunk["text"].encode()).hexdigest()
    for image in images:
        with open(image["path"], "rb") as f:
            image["content_hash"] = hashlib.sha256(f.read()).hexdigest()


def only_new(conn, table, items):
    """Keep items whose content isn't in the table yet, and only the first copy of any duplicate."""
    stored = {row[0] for row in conn.execute(f"SELECT content_hash FROM {table}")}
    new, seen = [], set(stored)
    for item in items:
        if item["content_hash"] not in seen:
            new.append(item)
            seen.add(item["content_hash"])
    return new


def save_chunks(conn, chunks, vectors):
    # All rows in one transaction: either every row is saved or none is
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO text_chunks (chunk_id, doc_id, source, page, content, content_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                [(c["chunk_id"], c["doc_id"], c["source"], c["page"], c["text"], c["content_hash"], v)
                 for c, v in zip(chunks, vectors)],
            )


def save_images(conn, images, vectors):
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO images (image_id, source, page, path, origin, ocr_text, content_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                [(i["image_id"], i["source"], i["page"], i["path"], i["origin"], i["ocr_text"], i["content_hash"], v)
                 for i, v in zip(images, vectors)],
            )
