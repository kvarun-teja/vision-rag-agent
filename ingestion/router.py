# Decide what kind of file each input is, and send it to the right reader.
#   PDF    -> parse_pdf  (text pages + the pictures inside it)
#   image  -> load_image (photos skip the PDF reader completely)
#   text   -> read directly
import logging
from pathlib import Path

from ingestion.images import load_image, merge_images
from ingestion.parse_pdf import parse_pdf

log = logging.getLogger(__name__)

PDF_TYPES = {".pdf"}
IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
TEXT_TYPES = {".txt", ".md"}


def route_file(path):
    """Return "pdf", "image" or "text" for a file, or None if we don't handle that type."""
    extension = Path(path).suffix.lower()
    if extension in PDF_TYPES:
        return "pdf"
    if extension in IMAGE_TYPES:
        return "image"
    if extension in TEXT_TYPES:
        return "text"
    return None


def read_text_file(path):
    # A plain text file is treated as a document with a single page
    text = Path(path).read_text(encoding="utf-8").strip()
    return [{"doc_id": Path(path).stem, "source": str(path), "page": 1, "text": text}]


def load_folder(folder, extracted_dir="data/extracted"):
    """Read every file in a folder.

    Returns four lists:
      sources - one entry per file we could read (for the sources table)
      pages   - text pages from PDFs and text files
      images  - every image: standalone photos plus pictures found inside PDFs
      skipped - names of files we couldn't use (unsupported or broken)
    """
    sources, pages, standalone_images, pdf_images, skipped = [], [], [], [], []

    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.name.startswith("."):
            continue  # ignore sub-folders and hidden files like .gitkeep

        kind = route_file(path)
        if kind is None:
            log.warning("Skipping %s: unsupported file type", path.name)
            skipped.append(path.name)
            continue

        try:
            if kind == "pdf":
                doc_pages, doc_images, page_count = parse_pdf(path, extracted_dir)
                pages += doc_pages
                pdf_images += doc_images
            elif kind == "image":
                standalone_images.append(load_image(path))
                page_count = None  # a photo has no pages
            else:
                pages += read_text_file(path)
                page_count = 1
        except Exception as error:
            # One broken file shouldn't stop the whole run: note it and keep going
            log.warning("Skipping %s: could not read it (%s)", path.name, error)
            skipped.append(path.name)
            continue

        sources.append({"source_id": path.name, "file_type": kind, "path": str(path), "pages": page_count})

    images = merge_images(standalone_images, pdf_images)
    return sources, pages, images, skipped
