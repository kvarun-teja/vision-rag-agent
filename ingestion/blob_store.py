# A simple blob store: every raw file is kept once, named by the fingerprint of its content.
#   data/blobs/<sha256>.pdf, data/blobs/<sha256>.jpg, ...
# Two identical files share one blob, and a blob never changes once written.
# (Moving to S3 or MinIO later would only mean rewriting put().)
import hashlib
import shutil
from pathlib import Path

BLOB_DIR = Path("data/blobs")


def file_hash(path):
    """SHA-256 fingerprint of a file's bytes."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def put(path, content_hash=None):
    """Copy a file into the blob store (if it isn't there already) and return its blob path."""
    content_hash = content_hash or file_hash(path)
    BLOB_DIR.mkdir(parents=True, exist_ok=True)
    blob_path = BLOB_DIR / f"{content_hash}{Path(path).suffix.lower()}"
    if not blob_path.exists():
        shutil.copyfile(path, blob_path)
    return str(blob_path)
