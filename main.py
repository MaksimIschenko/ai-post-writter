from config import (
    DATA_DIR,
    EMB_META_PATH,
    EMB_PATH,
    OUT_DIR,
    POSTS_JSONL,
)

if __name__ == "__main__":
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print(f"Posts file: {POSTS_JSONL}")
    print(f"Embeddings file: {EMB_PATH}")
    print(f"Embeddings meta file: {EMB_META_PATH}")
