# The whole ingestion pipeline in one command:
#   route -> parse -> blob store + sources table -> (skip stored images) -> OCR -> VLM captions
#   -> chunk (document text + captions + OCR text) -> embed -> store
# Run from the project folder:   python -m ingestion.run
# Rebuild everything from data/raw:  python -m ingestion.run --fresh
import logging
import sys
import time

from ingestion import blob_store, store
from ingestion.caption import add_captions
from ingestion.chunk import chunk_pages, chunks_from_images
from ingestion.embed import embed_images, embed_texts
from ingestion.ocr import add_ocr_text
from ingestion.router import load_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
log = logging.getLogger("ingestion")


def main(folder="data/raw", fresh=False):
    start = time.time()

    sources, pages, images, skipped = load_folder(folder)
    log.info("Read %d files: %d pages, %d images (%d files skipped)", len(sources), len(pages), len(images), len(skipped))

    with store.connect() as conn:
        if fresh:
            store.drop_tables(conn)
            log.info("Dropped the old tables (--fresh)")
        store.setup(conn)

        # Every raw file goes into the blob store and gets a row in the sources (metadata) table
        for source in sources:
            source["content_hash"] = blob_store.file_hash(source["path"])
            source["blob_path"] = blob_store.put(source["path"], source["content_hash"])
        store.save_sources(conn, sources)

        # Images: skip ones already stored, so a re-run doesn't redo the slow OCR, captioning and embedding
        store.add_hashes(images=images)
        new_images = store.only_new(conn, "images", images)
        for image in new_images:
            image["blob_path"] = blob_store.put(image["path"], image["content_hash"])
        if new_images:
            add_ocr_text(new_images)
            log.info("Captioning %d new images with the vision-language model...", len(new_images))
            add_captions(new_images)
            store.save_images(conn, new_images, embed_images([i["path"] for i in new_images]))

        # Repair: images stored earlier without a caption (for example, Ollama was down) get one now
        missing = store.images_without_captions(conn)
        if missing:
            log.info("Captioning %d stored images that have no caption yet...", len(missing))
            fixed = [i for i in add_captions(missing) if i["caption"]]
            store.save_captions(conn, fixed)
        else:
            fixed = []

        # Text chunks: document text, plus the captions and OCR text of the images captioned above
        chunks = chunk_pages(pages) + chunks_from_images(new_images) + chunks_from_images(fixed)
        store.add_hashes(chunks=chunks)
        new_chunks = store.only_new(conn, "text_chunks", chunks)
        if new_chunks:
            store.save_chunks(conn, new_chunks, embed_texts([c["text"] for c in new_chunks]))
        log.info("New: %d images, %d chunks (the rest were already stored)", len(new_images), len(new_chunks))

        counts = conn.execute(
            """SELECT (SELECT count(*) FROM sources), (SELECT count(*) FROM images),
                      (SELECT count(*) FROM text_chunks WHERE kind = 'text'),
                      (SELECT count(*) FROM text_chunks WHERE kind = 'caption'),
                      (SELECT count(*) FROM text_chunks WHERE kind = 'ocr')"""
        ).fetchone()

    log.info("Done in %.1f s. Database: %d sources, %d images, %d text chunks, %d caption chunks, %d OCR chunks.",
             time.time() - start, *counts)


if __name__ == "__main__":
    main(fresh="--fresh" in sys.argv)
