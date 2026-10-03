# Day 3 checks: routing, PDF parsing, images and OCR.
# Run from the project folder:  python -m pytest tests/test_ingestion_part1.py -v
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from ingestion.ocr import add_ocr_text
from ingestion.router import load_folder, route_file

PDF = Path("data/raw/Giraffe.pdf")
PHOTO = Path("data/raw/000000459757.jpg")  # a giraffe photo with no text in it


def test_router_picks_the_right_reader():
    assert route_file("report.pdf") == "pdf"
    assert route_file("photo.JPG") == "image"  # capital letters still work
    assert route_file("notes.txt") == "text"
    assert route_file("slides.pptx") is None   # unsupported


def test_pdf_text_and_both_kinds_of_images(tmp_path):
    # One PDF and one standalone photo in a folder
    shutil.copy(PDF, tmp_path)
    shutil.copy(PHOTO, tmp_path)

    sources, pages, images, skipped = load_folder(tmp_path, extracted_dir=tmp_path / "extracted")

    assert len(pages) > 0 and "giraffe" in pages[0]["text"].lower()  # text came out of the PDF
    origins = {img["origin"] for img in images}
    assert origins == {"file", "pdf"}                                 # both kinds of image landed in the set
    assert skipped == []


def test_ocr_runs_only_where_there_is_text(tmp_path):
    # Make an image that clearly contains text
    sign = Image.new("RGB", (900, 200), "white")
    ImageDraw.Draw(sign).text((30, 60), "GIRAFFE CROSSING AHEAD", fill="black", font=ImageFont.load_default(size=60))
    sign_path = tmp_path / "sign.png"
    sign.save(sign_path)

    images = add_ocr_text([{"path": str(sign_path)}, {"path": str(PHOTO)}])

    assert "GIRAFFE" in images[0]["ocr_text"]  # text image: OCR ran and read it
    assert images[1]["ocr_text"] is None       # plain photo: no OCR text stored


def test_broken_files_are_skipped_not_fatal(tmp_path):
    shutil.copy(PHOTO, tmp_path)
    (tmp_path / "broken.pdf").write_bytes(b"this is not really a pdf")
    (tmp_path / "broken.jpg").write_bytes(b"this is not really a photo")
    (tmp_path / "slides.pptx").write_bytes(b"unsupported")

    sources, pages, images, skipped = load_folder(tmp_path, extracted_dir=tmp_path / "extracted")

    assert [img["image_id"] for img in images] == [PHOTO.name]  # the good file still made it
    assert sorted(skipped) == ["broken.jpg", "broken.pdf", "slides.pptx"]
