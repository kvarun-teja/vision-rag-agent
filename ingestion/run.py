# The whole ingestion pipeline in one command:
#   route -> parse -> (skip already-stored items) -> OCR -> chunk -> embed -> store
# Run from the project folder:  python -m ingestion.run
import logging
import time

from ingestion import store
from ingestion.chunk import chunk_pages
from ingestion.embed import embed_images, embed_texts
from ingestion.ocr import add_ocr_text
from ingestion.router import load_folder

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingestion")


def main(folder="data/raw"):
    start = time.time()

    pages, images, skipped = load_folder(folder)
    chunks = chunk_pages(pages)
    log.info("Read %d pages -> %d chunks, %d images (%d files skipped)", len(pages), len(chunks), len(images), len(skipped))

    with store.connect() as conn:
        store.setup(conn)

        # Check fingerprints first, so a re-run doesn't redo OCR and embedding for things already saved
        store.add_hashes(chunks, images)
        new_chunks = store.only_new(conn, "text_chunks", chunks)
        new_images = store.only_new(conn, "images", images)
        log.info("New: %d chunks, %d images (the rest are already stored or duplicates)", len(new_chunks), len(new_images))

        if new_images:
            add_ocr_text(new_images)
        if new_chunks:
            store.save_chunks(conn, new_chunks, embed_texts([c["text"] for c in new_chunks]))
        if new_images:
            store.save_images(conn, new_images, embed_images([i["path"] for i in new_images]))

        total_chunks = conn.execute("SELECT count(*) FROM text_chunks").fetchone()[0]
        total_images = conn.execute("SELECT count(*) FROM images").fetchone()[0]

    log.info("Done in %.1f s. Database now holds %d chunks and %d images.", time.time() - start, total_chunks, total_images)


if __name__ == "__main__":
    main()
