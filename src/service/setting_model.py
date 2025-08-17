"""Модуль для настройки модели."""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.prompt import Confirm, Prompt
from rich.table import Table

from src.models import ModelConfig


def show_settings(cfg: ModelConfig, console: Console) -> None:
    """Показывает текущие настройки."""
    table = Table(title="Текущие настройки", box=box.SIMPLE)
    table.add_column("Параметр", style="cyan", no_wrap=True)
    table.add_column("Значение", style="white")
    table.add_row("Chat model", cfg.chat_model)
    table.add_row("Embed model", cfg.embed_model)
    table.add_row("Температура", str(cfg.temperature))
    table.add_row("Контекст токенов (примерно)", str(cfg.max_context_tokens))
    console.print(table)

def choose_settings(cfg: ModelConfig, console: Console) -> ModelConfig:
    """Выбор настроек модели."""
    show_settings(cfg, console)
    if Confirm.ask("Изменить настройки?", default=False):
        chat_model = Prompt.ask("Chat model (Ollama)", default=cfg.chat_model)
        embed_model = Prompt.ask("Embed model (Ollama)", default=cfg.embed_model)
        temp = float(Prompt.ask("Температура", default=str(cfg.temperature)))
        maxctx = int(Prompt.ask(
            "Макс. контекст токенов (примерно)",
            default=str(cfg.max_context_tokens)
            )
        )
        cfg = ModelConfig(
            chat_model=chat_model,
            embed_model=embed_model,
            temperature=temp,
            max_context_tokens=maxctx
        )
        console.print("Настройки обновлены.\n")
    return cfg

# Главная функция настройки модели
def setting_model(cfg: ModelConfig, console: Console) -> ModelConfig:
    """Настройки модели."""
    console.rule("[bold]Настройки[/bold]")
    return choose_settings(cfg, console)
