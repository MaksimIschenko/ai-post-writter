"""Utils for the project."""

from config import DATA_DIR, OUT_DIR


def ensure_dirs() -> None:
    """Проверяет наличие директорий и создает их, если они отсутствуют."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)



