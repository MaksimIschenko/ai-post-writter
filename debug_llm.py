# flake8: noqa: E501
"""Модуль для написания поста (отладочная версия без rich.Console)."""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import textwrap
from typing import Any

import numpy as np
import ollama
from slugify import slugify

from config import OUT_DIR
from src.models import ModelConfig, RetrievedStyle
from src.service.import_post import load_corpus
from src.utils import ensure_dirs


def _strip_code_fences(s: str) -> str:
    s = s.strip()
    s = re.sub(r"^\s*```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*```\s*$", "", s)
    return s

def _parse_json_loose(s: str) -> dict[str, Any] | None:
    """Попытаться распарсить JSON разными способами."""
    if s is None:
        return None
    if isinstance(s, dict):
        return s  # уже JSON
    if not isinstance(s, str):
        return None

    s0 = s
    # 1) прямой loads
    try:
        return json.loads(s0)
    except Exception:
        pass

    # 2) убрать кодовые ограждения
    s1 = _strip_code_fences(s0)
    try:
        return json.loads(s1)
    except Exception:
        pass

    # 3) самый широкий блок { ... }
    start = s1.find("{")
    end = s1.rfind("}")
    if start != -1 and end != -1 and end > start:
        cand = s1[start : end + 1]
        try:
            return json.loads(cand)
        except Exception:
            pass

    # 4) «жадный» регекс
    m = re.search(r"\{[\s\S]*\}", s1)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            pass

    return None


class LLMBackend:
    """Бэкенд для работы с LLM."""

    def __init__(self, cfg: ModelConfig) -> None:
        self.cfg = cfg

    def chat_json(self, system_prompt: str, user_prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
        """Запрос к LLM с требованием вернуть JSON (устойчивый парсер + ретрай)."""
        sp = system_prompt.strip()
        up = textwrap.dedent(
            f"""
            {user_prompt.strip()}

            ВАЖНО: верни СТРОГО корректный JSON без комментариев, преамбул и пояснений.
            {schema_hint or ''}
            """
        ).strip()

        messages = [
            {"role": "system", "content": sp},
            {"role": "user", "content": up},
        ]

        # Первая попытка: просим json-формат
        resp = ollama.chat(
            model=self.cfg.chat_model,
            messages=messages,
            options={"temperature": self.cfg.temperature, "format": "json"},
        )
        c1 = resp["message"]["content"]
        parsed = _parse_json_loose(c1)
        if parsed is not None:
            return parsed

        # Ретрай: максимально жёстко
        retry_messages = messages + [
            {"role": "assistant", "content": c1 if isinstance(c1, str) else json.dumps(c1)},
            {"role": "user", "content": "Повтори ответ СТРОГО как корректный JSON. Только JSON, без пояснений."},
        ]
        resp2 = ollama.chat(
            model=self.cfg.chat_model,
            messages=retry_messages,
            options={"temperature": 0, "format": "json"},
        )
        c2 = resp2["message"]["content"]
        parsed2 = _parse_json_loose(c2)
        if parsed2 is not None:
            return parsed2

        snippet = str(c2 if c2 is not None else c1)
        snippet = _strip_code_fences(snippet)
        snippet = snippet[:800].replace("\n", "\\n")
        raise ValueError(f"LLM не вернул корректный JSON. Фрагмент ответа: {snippet}")

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        messages = [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ]
        resp = ollama.chat(model=self.cfg.chat_model, messages=messages, options={"temperature": self.cfg.temperature})
        return resp["message"]["content"].strip()

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors: list[list[float]] = []
        for t in texts:
            tt = t if isinstance(t, str) and t.strip() else " "
            res = ollama.embeddings(model=self.cfg.embed_model, prompt=tt)
            vec = res.get("embedding")
            if vec is None:
                raise RuntimeError(f"Unexpected embeddings response: {res!r}")
            vectors.append(vec)

        arr = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr / norms


def top_k_similar(query: str, docs: list[str], backend: LLMBackend, k: int = 5) -> list[int]:
    if not docs:
        return []
    q_emb = backend.embed([query])
    d_emb = backend.embed(docs)
    sims = (q_emb @ d_emb.T).flatten()
    idx = np.argsort(-sims)[:k]
    return idx.tolist()


def build_style_pack(draft: str, corpus: list[dict[str, Any]], backend: LLMBackend, k: int = 5) -> RetrievedStyle:
    docs = [c["text"] for c in corpus]
    if not docs:
        return RetrievedStyle(examples=[], note="Нет примеров стиля: опирайся на нейтральный тон.")
    idxs = top_k_similar(draft, docs, backend, k=k)
    examples = [docs[i] for i in idxs]
    system = """
    Ты опытный редактор. На основе примеров выдели голос и стиль канала в 6-10 кратких пунктах.
    """
    user = "Примеры постов:\n\n" + "\n\n---\n\n".join(examples)
    note = backend.chat_text(system, user)
    return RetrievedStyle(examples=examples, note=note)


def chat_json(self, system_prompt: str, user_prompt: str, schema_hint: str | None = None) -> dict[str, Any]:
    """Запрос к LLM с требованием вернуть JSON (устойчивый парсер + ретрай)."""
    sp = system_prompt.strip()
    up = textwrap.dedent(
        f"""
        {user_prompt.strip()}

        ВАЖНО: верни СТРОГО корректный JSON без комментариев, преамбул и пояснений.
        {schema_hint or ''}
        """
    ).strip()

    messages = [
        {"role": "system", "content": sp},
        {"role": "user", "content": up},
    ]

    # Первая попытка: просим json-формат
    resp = ollama.chat(
        model=self.cfg.chat_model,
        messages=messages,
        options={"temperature": self.cfg.temperature, "format": "json"},
    )
    c1 = resp["message"]["content"]
    parsed = _parse_json_loose(c1)
    if parsed is not None:
        return parsed

    # Ретрай: максимально жёстко
    retry_messages = messages + [
        {"role": "assistant", "content": c1 if isinstance(c1, str) else json.dumps(c1)},
        {"role": "user", "content": "Повтори ответ СТРОГО как корректный JSON. Только JSON, без пояснений."},
    ]
    resp2 = ollama.chat(
        model=self.cfg.chat_model,
        messages=retry_messages,
        options={"temperature": 0, "format": "json"},
    )
    c2 = resp2["message"]["content"]
    parsed2 = _parse_json_loose(c2)
    if parsed2 is not None:
        return parsed2

    snippet = str(c2 if c2 is not None else c1)
    snippet = _strip_code_fences(snippet)
    snippet = snippet[:800].replace("\n", "\\n")
    raise ValueError(f"LLM не вернул корректный JSON. Фрагмент ответа: {snippet}")



def compose_post(draft: str, qa_pairs: list[tuple[str, str]], style: RetrievedStyle, images: list[str], backend: LLMBackend) -> str:
    examples_block = "\n\n".join([f"<пример>\n{ex}\n</пример>" for ex in style.examples]) if style.examples else ""
    imgs_block = "\n".join([f"- {os.path.basename(p)}" for p in images]) if images else "(без изображений)"

    sys_p = "Ты редактор телеграм-канала. Составь итоговый пост."
    qa_text = "\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs]) if qa_pairs else "(уточнений нет)"

    user_p = f"""
    Черновик:
    {draft.strip()}

    Вопросы и ответы:
    {qa_text}

    Изображения:
    {imgs_block}

    Стилевая записка:
    {style.note.strip()}

    Примеры:
    {examples_block}
    """
    text = backend.chat_text(sys_p, user_p)
    text = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text).strip()
    return text

def generate_questions(draft: str, style: RetrievedStyle, backend: LLMBackend, n: int = 3) -> list[str]:
    """Генерирует уточняющие вопросы и приводит их к чистому виду."""
    schema = 'Ожидаемый JSON: {"questions": ["..."]}'

    sys_p = f"""
    Ты телеграм-редактор. На основе черновика и стиля канала задай РОВНО {n} уточняющих вопросов,
    чтобы лучше понять контекст, целевую аудиторию, цель поста, конкретику (суммы, сроки, CTA) и желаемый тон.
    Каждый вопрос — отдельная короткая строка без пояснений и нумерации.
    """.strip()

    user_p = (
        "Черновик:\n" + draft.strip() +
        "\n\nКраткая заметка о стиле канала:\n" + style.note.strip() +
        "\n\nВажно: верни только JSON с ключом questions."
    )

    out = backend.chat_json(sys_p, user_p, schema)

    raw = out.get("questions", [])
    if not isinstance(raw, list):
        raw = []

    # Санитизация
    qs: list[str] = []
    for q in raw:
        if not isinstance(q, str):
            continue
        qq = q.strip()
        # убрать перевод строки/нумерацию/маркировку, если модель подсунула
        qq = re.sub(r"^\s*(?:[-*]|\d+\.)\s*", "", qq)  # "- " или "1. "
        qq = qq.replace("\n", " ").strip()
        if qq:
            qs.append(qq)

    # Обрезать/дополнить до n
    if len(qs) < n:
        qs += ["Уточните цель поста и желаемый результат?"] * (n - len(qs))
    qs = qs[:n]

    return qs


def write_post(cfg: ModelConfig) -> None:
    print("==== Написать пост ====")
    backend = LLMBackend(cfg)

    print("Вставьте черновой текст. Завершите ввод пустой строкой и строкой END.")

    lines: list[str] = []
    while True:
        line = input()
        if line.strip() == "END":
            break
        lines.append(line)
    draft = "\n".join(lines).strip()
    if not draft:
        print("Пустой черновик. Отмена.")
        return

    corpus = load_corpus()
    style = build_style_pack(draft, corpus, backend, k=5)

    questions = generate_questions(draft, style, backend, n=3)
    answers: list[str] = []
    for i, q in enumerate(questions, 1):
        print(f"[Вопрос {i}/3]: {q}")
        a = input("Ваш ответ: ")
        answers.append(a)

    qa_pairs = list(zip(questions, answers))

    print("Укажите пути к изображениям через запятую (или просто Enter)")
    img_line = input().strip()
    images: list[str] = []
    if img_line:
        images = [p.strip() for p in img_line.split(",") if p.strip()]

    final_post = compose_post(draft, qa_pairs, style, images, backend)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    title = slugify(draft.splitlines()[0] if draft else "post")
    out_path = OUT_DIR / f"{ts}_{title}.txt"
    ensure_dirs()
    out_path.write_text(final_post, encoding="utf-8")

    print("\nГотово! Итоговый текст сохранён:", str(out_path))
    print("\n==== Предпросмотр ====\n")
    print(final_post)


if __name__ == "__main__":
    write_post(ModelConfig())
