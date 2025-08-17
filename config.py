import os
import pathlib

APP_NAME = "Telegram Style Writer"
DATA_DIR = pathlib.Path("data")
OUT_DIR = pathlib.Path("out")
POSTS_JSONL = DATA_DIR / "posts.jsonl"
EMB_PATH = DATA_DIR / "embeddings.npy"
EMB_META_PATH = DATA_DIR / "embeddings_meta.json"

DEFAULT_CHAT_MODEL = os.environ.get(
    "TSW_CHAT_MODEL", "llama3.1:8b-instruct"
)  # Базовая модель для генерации текста
DEFAULT_EMBED_MODEL = os.environ.get(
    "TSW_EMBED_MODEL", "nomic-embed-text"
)  # Модель для генерации векторов
