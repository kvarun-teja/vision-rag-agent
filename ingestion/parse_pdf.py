# Pull the text and the pictures out of a PDF with PyMuPDF.
from pathlib import Path

import pymupdf

MIN_IMAGE_SIZE = 100  # pixels; smaller pictures are usually icons, logos or bullet points


def parse_pdf(path, extracted_dir):
    """Return (pages, images) for one PDF.

    pages  - one entry per page that has text
    images - one entry per picture, saved as its own file in extracted_dir
    """
    path = Path(path)
    extracted_dir = Path(extracted_dir)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    pages, images = [], []
    seen = set()  # the same picture can appear on many pages (like a logo); keep it once

    with pymupdf.open(path) as pdf:
        for page_number, page in enumerate(pdf, start=1):
            text = page.get_text().strip()
            if text:
                pages.append({"doc_id": path.stem, "source": str(path), "page": page_number, "text": text})

            for n, info in enumerate(page.get_images(full=True), start=1):
                xref = info[0]  # the picture's id number inside the PDF
                if xref in seen:
                    continue
                seen.add(xref)

                picture = pdf.extract_image(xref)
                if picture["width"] < MIN_IMAGE_SIZE or picture["height"] < MIN_IMAGE_SIZE:
                    continue

                image_path = extracted_dir / f"{path.stem}_page{page_number}_{n}.{picture['ext']}"
                image_path.write_bytes(picture["image"])
                images.append({
                    "image_id": f"{path.name}#page{page_number}-{n}",
                    "source": str(path),
                    "page": page_number,
                    "path": str(image_path),
                    "origin": "pdf",
                })

    return pages, images
