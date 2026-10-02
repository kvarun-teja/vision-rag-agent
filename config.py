# All settings in one place. Values come from the .env file (copy .env.example to .env);
# the second value in each getenv() is the default used when .env doesn't set it.
import os

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:localdev@localhost:5432/postgres")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
VLM_MODEL = os.getenv("VLM_MODEL", "moondream")     # answers questions about images
TEXT_MODEL = os.getenv("TEXT_MODEL", "qwen2.5:3b")  # answers text-only questions

TEXT_EMBED_MODEL = "BAAI/bge-small-en-v1.5"  # 384 numbers per chunk
IMAGE_EMBED_MODEL = "clip-ViT-B-32"          # 512 numbers per image (and per image-search query)

# Price per 1,000 tokens, for the cost estimate in the request log. Our models run locally, so it's 0;
# set it to a paid API's price to see what the same traffic would cost there.
COST_PER_1K_TOKENS = float(os.getenv("COST_PER_1K_TOKENS", "0"))

# How many results to fetch (chosen with retrieval/tune_top_k.py)
TEXT_TOP_K = 5    # the right PDF was always the top hit; 5 chunks gives the model enough context
IMAGE_TOP_K = 10  # image search is weaker, so fetch more; the vision tool narrows them down later
