# Read text that appears inside images (signs, labels, chart text) with Tesseract.
# Most photos have no text, so we first do a quick check on a small black-and-white copy,
# and only run the full OCR when that check finds real words.
import pytesseract
from PIL import Image

CHECK_SIZE = 500     # the quick check works on a copy at most this many pixels wide/tall
MIN_CONFIDENCE = 70  # Tesseract scores each word 0-100; below this we don't trust it
MIN_WORDS = 2        # how many trusted words it takes to say "this image has text"


def trusted_words(image):
    """Words Tesseract is confident about: at least 3 characters and containing a letter."""
    data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = []
    for word, confidence in zip(data["text"], data["conf"]):
        word = word.strip()
        if float(confidence) >= MIN_CONFIDENCE and len(word) >= 3 and any(c.isalpha() for c in word):
            words.append(word)
    return words


def has_text(image):
    """Quick check on a small grey copy of the image."""
    small = image.convert("L")  # "L" = greyscale
    small.thumbnail((CHECK_SIZE, CHECK_SIZE))
    return len(trusted_words(small)) >= MIN_WORDS


def add_ocr_text(images):
    """Add an "ocr_text" field to every image: the text found in it, or None."""
    for item in images:
        with Image.open(item["path"]) as img:
            img = img.convert("RGB")
            if has_text(img):
                item["ocr_text"] = pytesseract.image_to_string(img).strip()  # full OCR, full size
            else:
                item["ocr_text"] = None
    return images
