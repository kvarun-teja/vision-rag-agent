# Talk to our local models through Ollama and look at what goes in and what comes out.
#   - moondream answers a question about a photo
#   - qwen2.5:3b answers a text-only question (moondream can't do those)
# Run:  python scripts/call_vlm.py data/raw/<some photo>.jpg
import base64
import json
import sys

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"


def ask(model, prompt, image_path=None):
    # The request is a small JSON message: which model, what to ask, and (optionally) a photo
    request = {"model": model, "prompt": prompt, "stream": False}  # stream=False: wait for the full answer
    if image_path:
        with open(image_path, "rb") as f:
            # Photos travel as text: base64 turns the image bytes into letters and digits
            request["images"] = [base64.b64encode(f.read()).decode()]

    # Print the request, with the photo cut short so it fits on screen
    shown = dict(request)
    if image_path:
        shown["images"] = [request["images"][0][:40] + "..."]
    print("REQUEST:", json.dumps(shown, indent=2))

    reply = requests.post(OLLAMA_URL, json=request, timeout=120).json()

    print("REPLY keys:", list(reply.keys()))
    print("Answer:          ", reply["response"].strip())
    print("Tokens read:     ", reply["prompt_eval_count"])
    print("Tokens written:  ", reply["eval_count"])
    print("Time taken:       %.2f s" % (reply["total_duration"] / 1e9))  # Ollama reports nanoseconds
    print()


ask("moondream", "Describe this photo in one sentence.", image_path=sys.argv[1])
ask("qwen2.5:3b", "In one sentence, what is the tallest land animal?")
