# Caption every image with the vision-language model (moondream), so each image also gets a text
# description that is embedded and searched like any other text.
import logging

import config
from agent.llm import ask_model

DESCRIBE_PROMPT = "Describe this image in two sentences. Mention the main objects, what they are doing, and their colours."
RETRY_PROMPT = "Describe this image."  # moondream once returned nothing for the prompt above; this one worked

log = logging.getLogger(__name__)


def add_captions(images):
    """Add a "caption" field to every image (None if the model couldn't read it)."""
    for n, image in enumerate(images, start=1):
        try:
            # max_tokens stops moondream from rambling (one description once ran past 1,000 tokens)
            caption, _ = ask_model(config.VLM_MODEL, DESCRIBE_PROMPT, image_paths=[image["path"]], max_tokens=80)
            if not caption:
                caption, _ = ask_model(config.VLM_MODEL, RETRY_PROMPT, image_paths=[image["path"]], max_tokens=80)
            image["caption"] = caption or None
        except Exception as error:
            log.warning("No caption for %s (%s)", image["image_id"], error)
            image["caption"] = None
        if n % 50 == 0 or n == len(images):
            log.info("Captioned %d of %d images", n, len(images))
    return images
