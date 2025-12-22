
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()


# -----------------------------
# Config
# -----------------------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN environment variable not set")

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

MODEL_NAME = "Qwen/Qwen3-VL-235B-A22B-Instruct:novita"
