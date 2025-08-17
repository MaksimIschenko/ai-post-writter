# flake8: noqa: E501
"""Модуль для написания поста."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import textwrap
from typing import Any

import numpy as np
import ollama
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from slugify import slugify

from config import OUT_DIR
from src.models import ModelConfig, RetrievedStyle
from src.service.import_post import load_corpus
from src.utils import ensure_dirs


class LLMBackend:
    """Бэкенд для работы с LLM."""

    def __init__(self, cfg: ModelConfig) -> None:
        """Инициализация бэкенда.

        :param cfg: Конфигурация модели.
        :type cfg: ModelConfig
        """
        self.cfg = cfg

    # Chat with JSON-mode helper
    def chat_json(self, system_prompt: str, user_prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
        """Спрашивает модель в формате JSON.

        Ask model to return JSON. We'll try to parse; if it fails, we retry once.
        `schema_hint` is a short string included in the prompt to improve format compliance.

        :param system_prompt: Системный промпт.
        :type system_prompt: str
        :param user_prompt: Пользовательский промпт.
        :type user_prompt: str
        :param schema_hint: Подсказка формата.
        :type schema_hint: str | None
        :return: Ответ модели.
        :rtype: dict[str, Any]
        """
        sp = system_prompt.strip()
        up = textwrap.dedent(
            f"""
            {user_prompt.strip()}

            ТРЕБОВАНИЕ ФОРМАТА: ответь **строго** в JSON без каких-либо префиксов, суффиксов или комментариев.
            {schema_hint or ''}
            """
        ).strip()
        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": up},
        ]
        resp = ollama.chat(model=self.cfg.chat_model, messages=messages, options={"temperature": self.cfg.temperature})
        text = resp["message"]["content"].strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Try to extract JSON between braces
            m = re.search(r"\{[\s\S]*\}$", text)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:  # noqa: BLE001
                    pass
            # one retry with stronger instruction
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Повтори ответ СТРОГО как корректный JSON."})
            resp2 = ollama.chat(model=self.cfg.chat_model, messages=messages, options={"temperature": 0})
            text2 = resp2["message"]["content"].strip()
            return json.loads(re.search(r"\{[\s\S]*\}$", text2).group(0))  # type: ignore[arg-type]

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        """Спрашивает модель в текстовом формате.

        :param system_prompt: Системный промпт.
        :type system_prompt: str
        :param user_prompt: Пользовательский промпт.
        :type user_prompt: str
        :return: Ответ модели.
        :rtype: str
        """
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
        resp = ollama.chat(model=self.cfg.chat_model, messages=messages, options={"temperature": self.cfg.temperature})
        return resp["message"]["content"].strip()

    def embed(self, texts: list[str]) -> np.ndarray:
        """Вычисляет embedding для текстов.

        :param texts: Тексты.
        :type texts: list[str]
        :return: Embedding.
        :rtype: np.ndarray
        """
        # Ollama embeddings: returns dict with 'embeddings': [[...], ...]
        res = ollama.embeddings(model=self.cfg.embed_model, input=texts)
        arr = np.array(res["embeddings"], dtype=np.float32)
        # normalize rows
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr / norms

def top_k_similar(query: str, docs: list[str], backend: LLMBackend, k: int = 5) -> list[int]:
    """Находит k наиболее похожих текстов.

    :param query: Запрос.
    :type query: str
    :param docs: Тексты.
    :type docs: list[str]
    :param backend: Бэкенд.
    :type backend: LLMBackend
    :param k: Количество наиболее похожих текстов.
    :type k: int
    :return: Индексы наиболее похожих текстов.
    :rtype: list[int]
    """
    if not docs:
        return []
    # compute embeddings
    q_emb = backend.embed([query])  # (1, d)
    d_emb = backend.embed(docs)     # (N, d)
    sims = (q_emb @ d_emb.T).flatten()  # cosine similarity since normalized
    idx = np.argsort(-sims)[:k]
    return idx.tolist()

def build_style_pack(draft: str, corpus: list[dict[str, Any]], backend: LLMBackend, k: int = 5) -> RetrievedStyle:
    """Получает k наиболее похожих текстов и спрашивает модель о стиле.
    И спрашивает модель о стиле.

    Retrieve k nearest posts and ask model for a compact style note.

    :param draft: Черновик поста.
    :type draft: str
    :param corpus: Корпус постов.
    :type corpus: list[dict[str, Any]]
    :param backend: Бэкенд.
    :type backend: LLMBackend
    :param k: Количество наиболее похожих текстов.
    :type k: int
    :return: Стиль поста.
    :rtype: RetrievedStyle
    """
    docs = [c["text"] for c in corpus]
    if not docs:
        return RetrievedStyle(examples=[], note="Нет примеров стиля: опирайся на нейтральный тон телеграм-каналов.")
    idxs = top_k_similar(draft, docs, backend, k=k)
    examples = [docs[i] for i in idxs]
    # derive style note
    system = """
    Ты опытный редактор. На основе примеров выдели голос и стиль канала в 6-10 кратких пунктах: тональность, лексика, формат, длина, эмодзи/хэштеги, структура.
    Отвечай кратко, маркдаун-списком.
    """
    user = "Примеры постов:\n\n" + "\n\n---\n\n".join(examples)
    note = backend.chat_text(system, user)
    return RetrievedStyle(examples=examples, note=note)

def generate_questions(draft: str, style: RetrievedStyle, backend: LLMBackend, n: int = 3) -> list[str]:
    """Генерирует уточняющие вопросы.

    :param draft: Черновик поста.
    :type draft: str
    :param style: Стиль поста.
    :type style: RetrievedStyle
    :param backend: Бэкенд.
    :type backend: LLMBackend
    :param n: Количество вопросов.
    :type n: int
    :return: Уточняющие вопросы.
    :rtype: list[str]
    """
    schema = "Ожидаемый JSON: {\n  \"questions\": [\"...\"]  // ровно N вопросов, коротких, без лишних слов\n}"
    sys_p = f"""
    Ты телеграм-редактор. На основе черновика и стиля канала задай РОВНО {n} уточняющих вопросов,
    чтобы лучше понять контекст, целевую аудиторию, цель поста, конкретику (суммы, сроки, CTA) и желаемый тон.
    Вопросы должны быть предельно конкретными и независимыми друг от друга. Коротко.
    """.strip()
    user_p = (
        "Черновик:\n" + draft.strip() +
        "\n\nКраткая заметка о стиле канала:\n" + style.note.strip() +
        "\n\nВажно: верни только JSON."
    )
    out = backend.chat_json(sys_p, user_p, schema)
    qs = [q.strip() for q in out.get("questions", []) if isinstance(q, str) and q.strip()]
    if len(qs) != n:
        # pad or trim
        qs = (qs + ["Уточните цель поста и желаемый результат?"] * n)[:n]
    return qs

def compose_post(draft: str, qa_pairs: list[tuple[str, str]], style: RetrievedStyle, images: list[str], backend: LLMBackend) -> str:
    """Составляет итоговый пост.

    :param draft: Черновик поста.
    :type draft: str
    :param qa_pairs: Пары вопросов и ответов.
    :type qa_pairs: list[tuple[str, str]]
    :param style: Стиль поста.
    :type style: RetrievedStyle
    :param images: Изображения.
    :type images: list[str]
    :param backend: Бэкенд.
    :type backend: LLMBackend
    :return: Итоговый пост.
    :rtype: str
    """
    examples_block = "\n\n".join([f"<пример>\n{ex}\n</пример>" for ex in style.examples]) if style.examples else ""
    imgs_block = "\n".join([f"- {os.path.basename(p)}" for p in images]) if images else "(без изображений)"

    sys_p = """
    Ты редактор телеграм-канала. Твоя задача — написать готовый к публикации пост на русском в стиле канала.
    Требования:
    - Соблюдай голос канала по примерам и стилевой записке.
    - Пиши компактно и структурированно, выдерживай редактирование под Telegram (короткие абзацы, смысл сверху).
    - Не выдумывай факты. Если каких-то данных нет — формулируй нейтрально, избегай категоричных утверждений.
    - В конце при необходимости предложи 3–5 релевантных хэштегов (без # если у канала не принято) и 1 краткий CTA, если логично.
    - Не добавляй никаких служебных комментариев.
    """.strip()

    qa_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]) if qa_pairs else "(уточнений нет)"

    user_p = f"""
    Черновик пользователя:
    ---
    {draft.strip()}
    ---

    Уточняющие вопросы и ответы:
    {qa_text}

    Изображения (пути файлов):
    {imgs_block}

    Стилевая записка:
    {style.note.strip()}

    Примеры постов канала:
    {examples_block}

    Сконструируй итоговый текст поста в этом стиле.
    """.strip()

    text = backend.chat_text(sys_p, user_p)
    # gentle post-cleanup: trim excess markdown fencing if model added
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    return text

# Главная функция написания поста при помощи LLM.
def write_post(cfg: ModelConfig, console: Console) -> None:
    """Написать пост при помощи LLM.

    :param cfg: Конфигурация модели.
    :type cfg: ModelConfig
    :param console: Консоль.
    :type console: Console
    """
    console.rule("[bold]Написать пост[/bold]")
    backend = LLMBackend(cfg)

    console.print(Panel("Вставьте черновой текст. Завершите ввод пустой строкой и строкой `END`.", style="dim"))

    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    draft = "\n".join(lines).strip()
    if not draft:
        console.print("[red]Пустой черновик. Отмена.[/red]")
        return

    # Build style pack (RAG)
    corpus = load_corpus()
    style = build_style_pack(draft, corpus, backend, k=5)

    # Ask clarifying questions sequentially
    questions = generate_questions(draft, style, backend, n=3)
    answers: list[str] = []
    for i, q in enumerate(questions, 1):
        console.print(Panel(f"Вопрос {i}/3: {q}", style="cyan"))
        a = Prompt.ask("Ваш ответ", default="")
        answers.append(a)

    qa_pairs = list(zip(questions, answers))

    # Ask for images (paths)
    console.print(Panel("Укажите пути к изображениям через запятую (или просто Enter)", style="dim"))
    img_line = input().strip()
    images: list[str] = []
    if img_line:
        images = [p.strip() for p in img_line.split(",") if p.strip()]

    # Compose final post
    final_post = compose_post(draft, qa_pairs, style, images, backend)

    # Save
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    title = slugify(draft.splitlines()[0] if draft else "post")
    out_path = OUT_DIR / f"{ts}_{title}.txt"
    ensure_dirs()
    out_path.write_text(final_post, encoding="utf-8")

    console.print(Panel("Готово! Итоговый текст сохранён:", style="green"))
    console.print(str(out_path))
    console.print()
    console.print(Panel("Предпросмотр:") )
    console.print(final_post)
