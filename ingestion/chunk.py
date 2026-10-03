# Split page text into overlapping chunks that end on sentence boundaries.
import re

MAX_WORDS = 150        # a chunk closes once it reaches about this many words
OVERLAP_SENTENCES = 1  # the last sentence of a chunk is repeated at the start of the next one
END_HEADINGS = ("See also", "References")  # in Wikipedia PDFs, the article ends here; the rest is reference lists


def remove_back_matter(pages):
    """Drop everything from a document's "See also"/"References" heading onwards."""
    kept, finished_docs = [], set()
    for page in pages:
        if page["doc_id"] in finished_docs:
            continue
        lines = page["text"].splitlines()
        for i, line in enumerate(lines):
            if line.strip() in END_HEADINGS:
                lines = lines[:i]  # keep only the part of this page above the heading
                finished_docs.add(page["doc_id"])
                break
        text = "\n".join(lines).strip()
        if text:
            kept.append({**page, "text": text})
    return kept


def clean(text):
    text = re.sub(r"\[\w{1,3}\]", "", text)  # citation markers like [12] or [a]
    text = re.sub(r"-\n(\w)", r"\1", text)    # words broken across lines: "domesti-\ncated" -> "domesticated"
    text = re.sub(r"\s+", " ", text)          # newlines and repeated spaces -> a single space
    return text.strip()


def split_sentences(text):
    # A sentence ends with . ! or ? followed by a space
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # Text without full stops (like a table) could become one giant "sentence", so cut those into pieces
    pieces = []
    for sentence in sentences:
        words = sentence.split()
        for start in range(0, len(words), MAX_WORDS):
            pieces.append(" ".join(words[start:start + MAX_WORDS]))
    return pieces


def chunk_page(text):
    chunks, current, new_sentences = [], [], 0
    for sentence in split_sentences(clean(text)):
        current.append(sentence)
        new_sentences += 1
        if sum(len(s.split()) for s in current) >= MAX_WORDS:
            chunks.append(" ".join(current))
            current = current[-OVERLAP_SENTENCES:]  # carry the overlap into the next chunk
            new_sentences = 0
    if new_sentences > 0:  # whatever is left over, unless it's only the carried-over sentence
        chunks.append(" ".join(current))
    return chunks


def chunk_pages(pages):
    """Turn pages into chunks, each one labelled with where it came from."""
    chunks = []
    for page in remove_back_matter(pages):
        for n, text in enumerate(chunk_page(page["text"]), start=1):
            chunks.append({
                "chunk_id": f"{page['doc_id']}-p{page['page']}-c{n}",
                "source": page["source"],
                "page": page["page"],
                "kind": "text",    # written text from a document
                "image_id": None,
                "text": text,
            })
    return chunks


def chunks_from_images(images):
    """Each image's caption and OCR text become text chunks of their own, linked back to the image.
    Example: the red airplane photo gets "...#caption" (moondream's description) and "...#ocr" ("BAE SYSTEMS")."""
    chunks = []
    for image in images:
        for kind in ("caption", "ocr"):
            text = image.get("ocr_text" if kind == "ocr" else "caption")
            if text:
                chunks.append({
                    "chunk_id": f"{image['image_id']}#{kind}",
                    "source": image["source"],
                    "page": image["page"],
                    "kind": kind,
                    "image_id": image["image_id"],
                    "text": clean(text),
                })
    return chunks
