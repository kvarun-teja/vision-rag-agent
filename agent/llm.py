# One small helper for calling our local models through Ollama (same request as scripts/call_vlm.py).
import base64

import requests

import config


def ask_model(model, prompt, system=None, image_paths=None, max_tokens=None):
    """Send a prompt (and optionally images) to a model. Returns (answer, token usage)."""
    request = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0},  # no randomness: the same question gets the same answer
    }
    if max_tokens:
        request["options"]["num_predict"] = max_tokens  # stop writing after this many tokens
    if system:
        request["system"] = system  # standing instructions the model should follow for this prompt
    if image_paths:
        request["images"] = []
        for path in image_paths:
            with open(path, "rb") as f:
                request["images"].append(base64.b64encode(f.read()).decode())

    reply = requests.post(f"{config.OLLAMA_URL}/api/generate", json=request, timeout=300).json()
    usage = {"prompt_tokens": reply.get("prompt_eval_count", 0), "answer_tokens": reply.get("eval_count", 0)}
    return reply["response"].strip(), usage


def ask_for_tool_call(model, prompt, tool):
    """Offer the model one tool (a description of a function and its inputs).

    The model doesn't run anything: it replies with the tool's name and the inputs it wants.
    Our code then runs the real function. Returns (inputs or None, token usage).
    """
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [tool],
        "stream": False,
        "options": {"temperature": 0},
    }
    reply = requests.post(f"{config.OLLAMA_URL}/api/chat", json=request, timeout=300).json()
    usage = {"prompt_tokens": reply.get("prompt_eval_count", 0), "answer_tokens": reply.get("eval_count", 0)}
    calls = reply["message"].get("tool_calls") or []
    return (calls[0]["function"]["arguments"] if calls else None), usage
