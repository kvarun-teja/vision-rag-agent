# Save sources, images and text chunks, with their vectors and labels, in Postgres + pgvector.
import hashlib
from pathlib import Path

import psycopg
from pgvector.psycopg import register_vector

import config
from ingestion.blob_store import file_hash

# Three tables. sources is the metadata table: one row per input file.
# Every vector row links back to its source file and page; image rows also keep their caption,
# and caption/OCR chunks point at the image they describe.
SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
    source_id     TEXT PRIMARY KEY,       -- the file name, e.g. Giraffe.pdf
    file_type     TEXT NOT NULL,          -- pdf, image or text
    blob_path     TEXT NOT NULL,          -- where the raw file is kept (data/blobs/...)
    content_hash  TEXT NOT NULL,
    pages         INTEGER,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS images (
    image_id      TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL REFERENCES sources,
    page          INTEGER,
    blob_path     TEXT NOT NULL,
    origin        TEXT NOT NULL,          -- file (a photo) or pdf (a picture inside a PDF)
    caption       TEXT,                   -- written by the vision-language model
    ocr_text      TEXT,
    content_hash  TEXT UNIQUE NOT NULL,
    embedding     vector(512) NOT NULL    -- CLIP
);

CREATE TABLE IF NOT EXISTS text_chunks (
    chunk_id      TEXT PRIMARY KEY,
    source_id     TEXT NOT NULL REFERENCES sources,
    page          INTEGER,
    kind          TEXT NOT NULL,          -- text (from a document), caption or ocr (from an image)
    image_id      TEXT REFERENCES images, -- set for caption and ocr chunks
    content       TEXT NOT NULL,
    content_hash  TEXT UNIQUE NOT NULL,
    embedding     vector(384) NOT NULL    -- BGE
);
"""


def connect(autocommit=False):
    # autocommit=True is for read-only users like the retriever: each query finishes on its own.
    # Without it, the first SELECT opens a transaction that stays open, holding locks that blocked
    # "DROP TABLE" for 20 minutes when the API had been idle for 17 hours.
    conn = psycopg.connect(config.DATABASE_URL, autocommit=autocommit)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")  # switch pgvector on (only does work the first time)
    conn.commit()
    register_vector(conn)  # lets us pass numpy vectors straight into SQL
    return conn


def setup(conn):
    conn.execute(SCHEMA)
    conn.commit()


def drop_tables(conn):
    # Used by "python -m ingestion.run --fresh" to rebuild everything from data/raw
    conn.execute("DROP TABLE IF EXISTS text_chunks, images, sources CASCADE")
    conn.commit()


def add_hashes(chunks=(), images=()):
    # A fingerprint of the content: identical content always gives the identical hash
    for chunk in chunks:
        # A caption is only a duplicate if it describes the same image: two photos can share a caption
        key = chunk["text"] if chunk["kind"] == "text" else f"{chunk['image_id']}:{chunk['text']}"
        chunk["content_hash"] = hashlib.sha256(key.encode()).hexdigest()
    for image in images:
        image["content_hash"] = file_hash(image["path"])


def only_new(conn, table, items):
    """Keep items whose content isn't in the table yet, and only the first copy of any duplicate."""
    stored = {row[0] for row in conn.execute(f"SELECT content_hash FROM {table}")}
    new, seen = [], set(stored)
    for item in items:
        if item["content_hash"] not in seen:
            new.append(item)
            seen.add(item["content_hash"])
    return new


def source_id(item):
    return Path(item["source"]).name  # "data/raw/Giraffe.pdf" -> "Giraffe.pdf"


def save_sources(conn, sources):
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO sources (source_id, file_type, blob_path, content_hash, pages)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                [(s["source_id"], s["file_type"], s["blob_path"], s["content_hash"], s["pages"]) for s in sources],
            )


def save_images(conn, images, vectors):
    # All rows in one transaction: either every row is saved or none is
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO images (image_id, source_id, page, blob_path, origin, caption, ocr_text, content_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                [(i["image_id"], source_id(i), i["page"], i["blob_path"], i["origin"], i.get("caption"),
                  i.get("ocr_text"), i["content_hash"], v)
                 for i, v in zip(images, vectors)],
            )


def images_without_captions(conn):
    rows = conn.execute("SELECT image_id, source_id, page, blob_path FROM images WHERE caption IS NULL").fetchall()
    return [{"image_id": r[0], "source": r[1], "page": r[2], "path": r[3]} for r in rows]


def save_captions(conn, images):
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany("UPDATE images SET caption = %s WHERE image_id = %s",
                            [(i["caption"], i["image_id"]) for i in images])


def save_chunks(conn, chunks, vectors):
    with conn.transaction():
        with conn.cursor() as cur:
            cur.executemany(
                """INSERT INTO text_chunks (chunk_id, source_id, page, kind, image_id, content, content_hash, embedding)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                [(c["chunk_id"], source_id(c), c["page"], c["kind"], c["image_id"], c["text"], c["content_hash"], v)
                 for c, v in zip(chunks, vectors)],
            )
