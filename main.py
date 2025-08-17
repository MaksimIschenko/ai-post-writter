from __future__ import annotations

import time

import typer
from rich import box
from rich.console import Console
from rich.prompt import Prompt
from rich.table import Table

from config import (
    APP_NAME,
)
from src.import_post import import_posts
from src.models import ModelConfig
from src.utils import ensure_dirs

console = Console()
app = typer.Typer(add_completion=False, help=APP_NAME)

def show_header() -> None:
    """Показывает заголовок приложения."""
    console.rule(f"[bold cyan]{APP_NAME}[/bold cyan]")

def show_settings(cfg: ModelConfig) -> None:
    """Показывает текущие настройки."""
    table = Table(title="Текущие настройки", box=box.SIMPLE)
    table.add_column("Параметр", style="cyan", no_wrap=True)
    table.add_column("Значение", style="white")
    table.add_row("Chat model", cfg.chat_model)
    table.add_row("Embed model", cfg.embed_model)
    table.add_row("Температура", str(cfg.temperature))
    table.add_row("Контекст токенов (примерно)", str(cfg.max_context_tokens))
    console.print(table)

# Точка входа
@app.command()
def run() -> None:
    ensure_dirs()
    cfg = ModelConfig()
    while True:
        show_header()
        show_settings(cfg)
        #console.print(Panel.fit("Hello, [bold]world[/]!", title=APP_NAME))

        table = Table(box=box.SIMPLE, title="Действия")
        table.add_column("#", justify="right")
        table.add_column("Действие")
        table.add_row("1", "Написать пост")
        table.add_row("2", "Импортировать посты из Telegram")
        table.add_row("3", "Настройки модели")
        table.add_row("4", "Выход")
        console.print(table)

        choice = Prompt.ask("Выберите пункт", choices=["1", "2", "3", "4"], default="1")

        if choice == "1":
            print("Написать пост")
        elif choice == "2":
            import_posts(console)
        elif choice == "3":
            show_settings(cfg)
        elif choice == "4":
            break
        time.sleep(5)


if __name__ == "__main__":
    app()
