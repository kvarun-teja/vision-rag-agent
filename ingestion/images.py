# Standalone image files (photos) never go through the PDF reader.
# We load them here and merge them with the pictures pulled out of PDFs into one list.
from pathlib import Path

from PIL import Image


def load_image(path):
    """Check that an image file opens, and describe it the same way PDF pictures are described."""
    path = Path(path)
    with Image.open(path) as img:
        img.verify()  # checks the file isn't broken, without loading every pixel
    return {
        "image_id": path.name,
        "source": str(path),
        "page": None,  # a photo isn't part of a document, so it has no page
        "path": str(path),
        "origin": "file",
    }


def merge_images(standalone, from_pdfs):
    """One list of every image, wherever it came from."""
    return standalone + from_pdfs
