# Download a small mixed dataset into data/:
#   - 10 Wikipedia articles as PDFs (the "text" side)
#   - 200 COCO photos of the same 10 topics (the "image" side)
# Run from the project folder:  python scripts/download_data.py
import json
import random
import zipfile
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")          # everything the system will ingest goes here
COCO_DIR = Path("data/coco")        # COCO annotation files (not ingested)
CAPTIONS_FILE = Path("data/captions.json")  # human captions of our photos, used to write the eval set

# Each topic is a COCO object category with a matching Wikipedia article
TOPICS = {
    "giraffe": "Giraffe",
    "zebra": "Zebra",
    "elephant": "Elephant",
    "dog": "Dog",
    "cat": "Cat",
    "horse": "Horse",
    "pizza": "Pizza",
    "bicycle": "Bicycle",
    "train": "Train",
    "airplane": "Airplane",
}
IMAGES_PER_TOPIC = 20

# Wikipedia asks scripts to identify themselves
HEADERS = {"User-Agent": "vision-rag-agent/0.1 (student project)"}
COCO_ZIP_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
COCO_IMAGE_URL = "http://images.cocodataset.org/val2017/"


def download(url, path):
    # Save a URL to a file in small pieces, so big files don't fill up memory
    with requests.get(url, headers=HEADERS, stream=True, timeout=60) as response:
        response.raise_for_status()
        with open(path, "wb") as f:
            for piece in response.iter_content(chunk_size=1024 * 1024):
                f.write(piece)


def download_pdfs():
    for title in TOPICS.values():
        path = RAW_DIR / f"{title}.pdf"
        if path.exists():
            continue
        download(f"https://en.wikipedia.org/api/rest_v1/page/pdf/{title}", path)
        print(f"PDF   {path}  ({path.stat().st_size // 1024} KB)")


def load_coco_annotations():
    # The zip holds many files; we only need the captions and the object labels for val2017
    zip_path = COCO_DIR / "annotations_trainval2017.zip"
    if not zip_path.exists():
        print("Downloading COCO annotations (about 250 MB)...")
        download(COCO_ZIP_URL, zip_path)

    with zipfile.ZipFile(zip_path) as z:
        captions = json.loads(z.read("annotations/captions_val2017.json"))
        instances = json.loads(z.read("annotations/instances_val2017.json"))
    return captions, instances


def download_images(captions, instances):
    # Find COCO's id for each topic name, e.g. "giraffe" -> 25
    category_ids = {c["name"]: c["id"] for c in instances["categories"]}

    # For each topic, collect every photo that contains that object
    photos_by_topic = {topic: set() for topic in TOPICS}
    for a in instances["annotations"]:
        for topic in TOPICS:
            if a["category_id"] == category_ids[topic]:
                photos_by_topic[topic].add(a["image_id"])

    file_names = {img["id"]: img["file_name"] for img in instances["images"]}
    captions_by_photo = {}
    for c in captions["annotations"]:
        captions_by_photo.setdefault(c["image_id"], []).append(c["caption"].strip())

    random.seed(42)  # same "random" picks every time we run this
    chosen = {}
    for topic, photo_ids in photos_by_topic.items():
        candidates = sorted(photo_ids - set(chosen))  # skip photos already picked for another topic
        for photo_id in random.sample(candidates, IMAGES_PER_TOPIC):
            chosen[photo_id] = topic

    for photo_id, topic in chosen.items():
        path = RAW_DIR / file_names[photo_id]
        if not path.exists():
            download(COCO_IMAGE_URL + file_names[photo_id], path)

    # Save the human captions so we can write eval questions with known answers
    ground_truth = {
        file_names[photo_id]: {"topic": topic, "captions": captions_by_photo[photo_id]}
        for photo_id, topic in chosen.items()
    }
    CAPTIONS_FILE.write_text(json.dumps(ground_truth, indent=2))
    print(f"Images: {len(chosen)} photos in {RAW_DIR}, captions saved to {CAPTIONS_FILE}")


if __name__ == "__main__":
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    COCO_DIR.mkdir(parents=True, exist_ok=True)
    download_pdfs()
    captions, instances = load_coco_annotations()
    download_images(captions, instances)
