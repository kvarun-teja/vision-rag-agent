# The vision tool: a CV model (a pretrained YOLO object detector) wrapped as a function the agent can call.
#
# Image search (CLIP) is fuzzy: it returns photos that are roughly "about" the question.
# YOLO is precise and cheap: it lists the actual objects in a photo in a few milliseconds.
# So we run YOLO on all candidates first, keep only photos with the right objects,
# and only those few go to the slow vision-language model.
from pathlib import Path

from ultralytics import YOLO, settings

settings.update({"sync": False})  # turn off Ultralytics' anonymous usage reporting

# "m" = medium (40 MB, ~24 ms per image). The nano version was 3x faster but missed a frisbee in a dog's mouth.
MODEL_PATH = Path("models/yolo11m.pt")
MIN_CONFIDENCE = 0.4                     # ignore detections YOLO is less than 40% sure about

_model = None


def model():
    global _model
    if _model is None:
        MODEL_PATH.parent.mkdir(exist_ok=True)
        _model = YOLO(str(MODEL_PATH))
    return _model


def object_labels():
    """The 80 everyday objects this YOLO was trained to find (person, dog, frisbee, giraffe, ...)."""
    return list(model().names.values())


def detect_objects(image_paths):
    """For each image, the objects YOLO sees, as {label: highest confidence}."""
    all_objects = []
    for path in image_paths:
        # One image at a time: a batch of 10 ran out of memory on the 4 GB card shared with Ollama
        result = model()(path, conf=MIN_CONFIDENCE, verbose=False)[0]
        objects = {}
        for class_id, confidence in zip(result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            label = result.names[int(class_id)]
            objects[label] = round(max(confidence, objects.get(label, 0)), 2)
        all_objects.append(objects)
    return all_objects


def tool_description():
    """How the tool is described to the language model: its name, what it does, and the inputs it takes."""
    return {
        "type": "function",
        "function": {
            "name": "find_objects",
            "description": "Run an object detector on the candidate photos and keep only the photos that contain the given objects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "objects": {
                        "type": "array",
                        "items": {"type": "string", "enum": object_labels()},  # only labels YOLO actually knows
                        "description": "The objects a matching photo must contain.",
                    }
                },
                "required": ["objects"],
            },
        },
    }


def find_objects(images, targets, keep=3):
    """The tool: label every candidate image with its objects, then keep the best few containing the targets.

    Photos with ALL target objects come first; if there are none, photos with ANY target;
    if still none, nothing is kept (and the agent will say it couldn't find a match).
    """
    for image, objects in zip(images, detect_objects([img["path"] for img in images])):
        image["objects"] = objects

    with_all = [img for img in images if all(t in img["objects"] for t in targets)]
    with_any = [img for img in images if any(t in img["objects"] for t in targets)]
    return (with_all or with_any)[:keep]  # images are already in search order, best first
