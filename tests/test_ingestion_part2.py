# Checks for the blob store, the sources list, and caption/OCR chunks.
# Run from the project folder:  python -m pytest tests/test_ingestion_part2.py -v
import shutil
from pathlib import Path

from ingestion import blob_store
from ingestion.chunk import chunks_from_images
from ingestion.router import load_folder
from ingestion.store import add_hashes

PDF = Path("data/raw/Giraffe.pdf")
PHOTO = Path("data/raw/000000459757.jpg")


def test_blob_store_keeps_one_copy_per_content(tmp_path, monkeypatch):
    monkeypatch.setattr(blob_store, "BLOB_DIR", tmp_path / "blobs")
    copy = tmp_path / "same_photo_other_name.jpg"
    shutil.copy(PHOTO, copy)

    first = blob_store.put(PHOTO)
    second = blob_store.put(copy)  # same bytes, different file name

    assert first == second                             # one blob for identical content
    assert len(list((tmp_path / "blobs").iterdir())) == 1
    assert Path(first).name == blob_store.file_hash(PHOTO) + ".jpg"


def test_every_readable_file_becomes_a_source(tmp_path):
    shutil.copy(PDF, tmp_path)
    shutil.copy(PHOTO, tmp_path)
    (tmp_path / "broken.pdf").write_bytes(b"not a pdf")

    sources, pages, images, skipped = load_folder(tmp_path, extracted_dir=tmp_path / "extracted")

    by_name = {s["source_id"]: s for s in sources}
    assert set(by_name) == {"Giraffe.pdf", PHOTO.name}  # the broken file is not a source
    assert by_name["Giraffe.pdf"]["file_type"] == "pdf" and by_name["Giraffe.pdf"]["pages"] == 31
    assert by_name[PHOTO.name]["file_type"] == "image" and by_name[PHOTO.name]["pages"] is None


def test_captions_and_ocr_become_linked_chunks():
    images = [
        {"image_id": "plane.jpg", "source": "data/raw/plane.jpg", "page": None,
         "caption": "A small red propeller airplane on a runway.", "ocr_text": "BAE\nSYSTEMS"},
        {"image_id": "dog.jpg", "source": "data/raw/dog.jpg", "page": None, "caption": "A dog on grass.", "ocr_text": None},
        {"image_id": "blank.jpg", "source": "data/raw/blank.jpg", "page": None, "caption": None, "ocr_text": None},
    ]

    chunks = chunks_from_images(images)

    assert [c["chunk_id"] for c in chunks] == ["plane.jpg#caption", "plane.jpg#ocr", "dog.jpg#caption"]
    assert chunks[1]["text"] == "BAE SYSTEMS"          # cleaned: the line break is gone
    assert all(c["image_id"] in ("plane.jpg", "dog.jpg") for c in chunks)


def test_same_caption_on_two_images_is_not_a_duplicate():
    images = [
        {"image_id": "a.jpg", "source": "a.jpg", "page": None, "caption": "A giraffe in a field.", "ocr_text": None},
        {"image_id": "b.jpg", "source": "b.jpg", "page": None, "caption": "A giraffe in a field.", "ocr_text": None},
    ]
    chunks = chunks_from_images(images)

    add_hashes(chunks=chunks)

    assert chunks[0]["content_hash"] != chunks[1]["content_hash"]  # each image keeps its own caption chunk
