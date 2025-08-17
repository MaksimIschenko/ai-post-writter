"""Модуль для импорта постов из Telegram.

Для работы необходимо получить API ID и API hash в my.telegram.org
и указать их в переменных окружения (файл .env) TELEGRAM_API_ID и TELEGRAM_API_HASH.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import pathlib
import re
from typing import Any

from dotenv import load_dotenv
from rich.console import Console
from rich.prompt import Prompt

try:
    from telethon import TelegramClient
except Exception:
    TelegramClient = None  # type: ignore

from config import POSTS_JSONL
from src.utils import ensure_dirs

load_dotenv()


def sha1(text: str) -> str:
    """Хеш строки.

    :param text: Строка для хеширования.
    :type text: str

    :return: Хеш строки.
    :rtype: str
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def load_corpus(jsonl_path: pathlib.Path = POSTS_JSONL) -> list[dict[str, Any]]:
    """Загружает корпус из файла.

    :param jsonl_path: Путь к файлу корпуса.
    :type jsonl_path: pathlib.Path

    :return: Корпус.
    :rtype: list[dict[str, Any]]
    """
    if not jsonl_path.exists():
        return []
    corpus = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                text = (obj.get("text") or "").strip()
                if not text:
                    continue
                # normalize repeated newlines
                text = re.sub(r"\n{3,}", "\n\n", text)
                obj["text"] = text
                corpus.append(obj)
            except Exception:  # noqa: BLE001
                continue
    return corpus


def save_corpus(corpus: list[dict[str, Any]], jsonl_path: pathlib.Path = POSTS_JSONL) -> None:
    """Сохраняет корпус в файл.

    :param corpus: Корпус.
    :type corpus: list[dict[str, Any]]

    :param jsonl_path: Путь к файлу корпуса.
    :type jsonl_path: pathlib.Path
    """
    ensure_dirs()
    with jsonl_path.open("w", encoding="utf-8") as f:
        for obj in corpus:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

async def import_telegram_posts(
    api_id: int, api_hash: str, channel: str, limit: int = 500
) -> list[dict[str, Any]]:
    """Импортирует посты из Telegram.

    :param api_id: API ID.
    :type api_id: int
    :param api_hash: API hash.
    :type api_hash: str
    :param channel: Канал.
    :type channel: str
    :param limit: Количество постов для импорта.
    :type limit: int

    :return: Импортированные посты.
    :rtype: list[dict[str, Any]]
    """
    if TelegramClient is None:
        raise RuntimeError("Telethon is not installed. `pip install telethon`.")

    session_name = "tsw_session"
    client = TelegramClient(session_name, api_id, api_hash)

    kept = skipped_no_text = 0
    corpus: list[dict[str, Any]] = []

    try:
        await client.start()
        entity = await client.get_entity(channel)

        async for msg in client.iter_messages(entity, limit=limit):
            text = (getattr(msg, "text", None) or "").strip()
            if not text:
                skipped_no_text += 1
                continue
            corpus.append({
                "id": msg.id,
                "date": msg.date.isoformat() if msg.date else None,
                "text": text,
            })
            kept += 1
    finally:
        await client.disconnect()

    existing = load_corpus()
    seen = {sha1(x["text"]) for x in existing}
    new_items = [x for x in corpus if sha1(x["text"]) not in seen]
    save_corpus(existing + new_items)

    print(f"Fetched: {kept + skipped_no_text}, kept: {kept}, skipped(no text): {skipped_no_text}")
    return new_items


# Главная функция импорта постов из Telegram
def import_posts(console: Console) -> None:
    """Пайплайн импорта постов из Telegram.

    :param console: Консоль.
    :type console: Console
    """
    console.rule("[bold]Импорт постов из Telegram[/bold]")
    if TelegramClient is None:
        console.print("[red]Telethon не установлен. Установите: pip install telethon[/red]")
        return

    # Получаем API ID и API hash из переменных окружения
    api_id = int(os.getenv("TELEGRAM_API_ID", "0"))
    api_hash = os.getenv("TELEGRAM_API_HASH", "")
    # Если нет API ID или API hash, запрашиваем их у пользователя
    if api_id <= 0 or not api_hash:
        api_id = int(Prompt.ask("api_id (my.telegram.org)", default="0"))
        api_hash = Prompt.ask("api_hash (my.telegram.org)", password=True)
        os.environ["TELEGRAM_API_ID"] = str(api_id)
        os.environ["TELEGRAM_API_HASH"] = api_hash

    channel = Prompt.ask("@username или t.me/ссылка канала")
    limit = int(Prompt.ask("Сколько постов загрузить (до)", default="500"))

    console.print("Подключаемся к Telegram...")
    try:
        new_items = asyncio.run(import_telegram_posts(api_id, api_hash, channel, limit=limit))
    except Exception as e:  # noqa: BLE001
        console.print(f"[red]Ошибка импорта:[/red] {e}")
        return

    console.print(f"Импортировано новых постов: {len(new_items)}")
    console.print(f"Всего в корпусе: {len(load_corpus())}")
