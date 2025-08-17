#!/usr/bin/env bash
set -euo pipefail

# === Конфиг ===
PROJECT_NAME_TOML="ai-post-writter"                 # из [project].name
PACKAGE_NAME="${PROJECT_NAME_TOML//-/_}"            # имя пакета в src/
REQUIRES_PY="3.12"                                  # мажор-минор из requires-python
PY_PIN_CANDIDATES=(
  "cpython-${REQUIRES_PY}.11-macos-aarch64-none"
  "cpython-${REQUIRES_PY}.10-macos-aarch64-none"
  "cpython-${REQUIRES_PY}.9-macos-aarch64-none"
  "cpython-${REQUIRES_PY}.8-macos-aarch64-none"
  "cpython-${REQUIRES_PY}.7-macos-aarch64-none"
  "cpython-${REQUIRES_PY}.6-macos-aarch64-none"
  "cpython-${REQUIRES_PY}.5-macos-aarch64-none"
  "cpython-${REQUIRES_PY}.4-macos-aarch64-none"
)

# === Цвета для логов ===
bold() { printf "\033[1m%s\033[0m\n" "$*"; }
info() { printf "👉 %s\n" "$*"; }
ok()   { printf "✅ %s\n" "$*"; }
warn() { printf "⚠️  %s\n" "$*"; }

# === 0) Проверка наличия pyproject.toml ===
if [[ ! -f "pyproject.toml" ]]; then
  echo "pyproject.toml не найден в $(pwd). Помести скрипт в корень проекта и запусти."
  exit 1
fi

# === 1) Установка uv при отсутствии ===
if ! command -v uv >/dev/null 2>&1; then
  bold "Устанавливаю uv…"
  # macOS install
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ok "uv установлен."
  export PATH="$HOME/.local/bin:$PATH"
fi

# === 2) Фикс параметров ruff/mypy под имя пакета ===
#   - в твоём pyproject сейчас встречается "myproject" — заменим на имя пакета.
if grep -q 'myproject' pyproject.toml; then
  info "Привожу конфиги ruff/mypy к модулю ${PACKAGE_NAME}…"
  # BSD/macOS sed requires backup suffix; потом удалим бэкапы
  sed -i '' "s/myproject/${PACKAGE_NAME}/g" pyproject.toml || true
  find . -name "pyproject.toml''" -delete 2>/dev/null || true
  ok "Конфиги обновлены."
fi

# === 3) Создание структуры проекта ===
if [[ ! -d "src/${PACKAGE_NAME}" ]]; then
  info "Создаю src/${PACKAGE_NAME}…"
  mkdir -p "src/${PACKAGE_NAME}"
  printf "__all__ = []\n" > "src/${PACKAGE_NAME}/__init__.py"
  cat > "src/${PACKAGE_NAME}/cli.py" <<'PY'
from __future__ import annotations

import typer
from rich.console import Console
from rich.panel import Panel

app = typer.Typer(add_completion=False)
console = Console()

@app.command()
def hello(name: str = "world") -> None:
    console.print(Panel.fit(f"Hello, [bold]{name}[/]!", title="ai-post-writter"))

if __name__ == "__main__":
    app()
PY
  ok "Базовый пакет и CLI-скелет созданы."
fi

# === 4) Базовые файлы ===
if [[ ! -f ".gitignore" ]]; then
  cat > .gitignore <<'GIT'
.uv/
.venv/
dist/
build/
__pycache__/
*.pyc
*.pyo
*.pyd
*.egg-info/
.ipynb_checkpoints/
.mypy_cache/
.ruff_cache/
.coverage
GIT
  ok ".gitignore создан."
fi

if [[ ! -f "README.md" ]]; then
  cat > README.md <<EOF
# ${PROJECT_NAME_TOML}

Проект на Python, управляется **uv**.

## Быстрый старт

\`\`\`bash
uv sync --all-groups
uv run python -m ${PACKAGE_NAME}.cli hello
\`\`\`
EOF
  ok "README.md создан."
fi

# === 5) Пин версии Python 3.12.x ===
bold "Пин Python ${REQUIRES_PY}.x…"
PINNED=false

# Попробуем pin конкретный установленный интерпретатор 3.12, если есть
if command -v python3.12 >/dev/null 2>&1; then
  if uv python pin "$(command -v python3.12)"; then
    PINNED=true
    ok "Pinned $(uv run python -V)"
  fi
fi

# Иначе — перебираем кандидатов из uv python list
if [[ "${PINNED}" = false ]]; then
  # Убедимся, что список актуален
  uv python list >/dev/null || true
  for build in "${PY_PIN_CANDIDATES[@]}"; do
    if uv python pin "${build}" >/dev/null 2>&1; then
      ok "Pinned ${build}"
      PINNED=true
      break
    fi
  done
fi

# Если не получилось — попросим uv сам подтянуть 3.12.latest
if [[ "${PINNED}" = false ]]; then
  warn "Не удалось напрямую закрепить конкретный билд. Ставлю свежий патч 3.12…"
  uv python install "${REQUIRES_PY}"
  uv python pin "${REQUIRES_PY}"
  ok "Pinned $(uv run python -V)"
fi

# === 6) Синхронизация окружения (prod + dev) ===
bold "Устанавливаю зависимости (включая dev)…"
uv sync --all-groups
ok "Зависимости установлены."

# === 7) Форматирование, линт и тип-чек ===
bold "Форматирование Ruff…"
uv run ruff format .

bold "Линт Ruff…"
# не падаем на предупреждениях — собираем всё, что можно
if ! uv run ruff check .; then
  warn "Ruff нашёл замечания. Продолжаю."
fi

bold "Проверка типов mypy…"
if ! uv run mypy; then
  warn "Mypy нашёл замечания. Продолжаю."
fi

# === 8) Сборка пакета ===
bold "Сборка wheel и sdist…"
uv build
ok "Сборка завершена. Артефакты в ./dist"

# === 9) Подсказка по запуску CLI ===
bold "Проверка запуска:"
echo "  uv run python -m ${PACKAGE_NAME}.cli hello"
ok "Готово!"
