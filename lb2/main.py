"""ЛР №2: агент «Науковий Дайджест» на LlamaIndex (FunctionAgent + ArxivToolSpec)."""
import argparse
import asyncio
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

# Windows-консоль стартує у cp1251 і падає на кирилиці. Робимо це ПЕРЕД print().
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from llama_index.core.agent.workflow import FunctionAgent, ToolCall, ToolCallResult
from llama_index.core.tools import FunctionTool
from llama_index.llms.ollama import Ollama

# ── Налаштування (можна міняти через змінні середовища, без правки коду) ──
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3:1.7b")   # легка модель для ноутбука без GPU
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))
NUM_THREAD = int(os.getenv("NUM_THREAD", "0"))          # 0 = за замовчуванням
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))

TOPIC = "Multimodal Large Language Models"
TASK = (
    f"Знайди 2 найновіші статті на arXiv за темою '{TOPIC}'. "
    "Склади звіт за шаблоном і збережи його у файл 'digest.md'."
)


def check_environment() -> list:
    """Повертає список проблем (порожній — усе гаразд). Без трейсбеків."""
    try:
        with urllib.request.urlopen(f"{OLLAMA_BASE_URL}/api/tags", timeout=5) as r:
            available = {m["name"] for m in json.load(r).get("models", [])}
    except Exception as exc:
        return [f"Ollama не відповідає на {OLLAMA_BASE_URL} ({type(exc).__name__}).",
                "  Windows/macOS: відкрийте застосунок Ollama; Linux: `ollama serve`"]
    if AGENT_MODEL not in available and f"{AGENT_MODEL}:latest" not in available:
        return [f"Модель '{AGENT_MODEL}' не знайдена в Ollama.",
                f"  Виправити: ollama pull {AGENT_MODEL}"]
    return []


# ── Що насправді віддає ToolSpec з каталогу ──────────────────────────────
def show_toolspec() -> None:
    from llama_index.tools.arxiv import ArxivToolSpec
    spec = ArxivToolSpec()
    tools = spec.to_tool_list()
    print(f"ArxivToolSpec віддає {len(tools)} інструмент(ів):")
    for t in tools:
        print(f"  • {t.metadata.name}")
    result = tools[0](query=TOPIC, sort_by="recent")
    print("\nЩо повертає перший інструмент (перші 300 символів):")
    print(" ", str(result)[:300].replace("\n", " "), "…")
    print("\n  ↑ Авторів тут немає, тому нижче пишемо власний search_arxiv.")


# ── Власні інструменти ──────────────────────────────────────────────────
def search_arxiv(query: str, max_results: int = 2, recent: bool = True) -> str:
    """Searches arXiv: title, authors, date, URL, abstract.

    Args:
        query: Search phrase in English
        max_results: How many papers to return (1-5)
        recent: True for newest first, False for best relevance
    """
    import arxiv

    n = max(1, min(int(max_results), 5))
    raw = query.strip()
    # Пастка arXiv: з сортуванням за датою запит БЕЗ префікса поля ігнорується.
    expr = raw if ":" in raw else f'all:"{raw}"'
    order = arxiv.SortCriterion.SubmittedDate if recent else arxiv.SortCriterion.Relevance
    client = arxiv.Client(page_size=n, delay_seconds=3.0, num_retries=3)
    try:
        results = list(client.results(arxiv.Search(query=expr, max_results=n, sort_by=order)))
    except Exception as exc:  # мережа / ліміти arXiv
        return f"ERROR: arXiv request failed ({type(exc).__name__}). Try once more."
    if not results:
        return "No papers found. Try a broader query, once."
    out = []
    for i, r in enumerate(results, 1):
        authors = ", ".join(a.name for a in r.authors) or "n/a"
        out.append(
            f"[{i}]\nTITLE: {r.title.strip()}\nAUTHORS: {authors}\n"
            f"PUBLISHED: {r.published.date()}\nURL: {r.entry_id}\n"
            f"ABSTRACT: {' '.join(r.summary.split())[:600]}"
        )
    return "\n\n".join(out)


def write_report(filename: str, content: str) -> str:
    """Saves the finished Markdown report. Call this LAST, exactly once.

    Args:
        filename: File name ending in .md
        content: Full report in Markdown
    """
    # Ім'я файлу приходить ВІД МОДЕЛІ — це недовірений ввід.
    # Зрізаємо теки й літери диска, лишаємо тільки ім'я всередині робочої теки.
    base = REPORTS_DIR.resolve()
    base.mkdir(parents=True, exist_ok=True)
    name = str(filename).replace("\\", "/").rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        name = "report.md"
    if not name.endswith(".md"):
        name += ".md"
    target = (base / name).resolve()
    if not target.is_relative_to(base):
        return "Refused to write: pass a plain file name like 'digest.md'."
    target.write_text(content, encoding="utf-8")
    return f"Saved to {name} ({len(content.split())} words). Task complete."


SYSTEM_PROMPT = """You are "Науковий Дайджест", an agent that builds a report
on recent arXiv papers.

WORKFLOW — follow it exactly, once:
1. Call search_arxiv ONCE with the user's topic and the requested count.
2. Use ONLY the fields the tool returned. Never invent authors, dates or URLs.
   If a field is missing, write "n/a".
3. Build a Markdown report. Use this template for EVERY paper, verbatim.
   Do NOT add your own numbering, headings or intro — the report starts
   straight with the first "---":
---
### **<TITLE>**
**Автори:** *<AUTHORS>*
**Опубліковано:** <PUBLISHED>
**Посилання:** [Читати на arXiv](<URL>)
**Анотація:**
> <ABSTRACT>
---
4. Do NOT print the report in your reply. Saving is a TOOL CALL: call the
   tool write_report with filename="digest.md" and put the whole report into
   its content argument. Never write "write_report" as plain text.
   The task is NOT done until the tool has returned "Saved".
5. Only after that, give a one-sentence final answer in Ukrainian.

Your only tools are search_arxiv and write_report."""


def build_agent() -> FunctionAgent:
    extra = {"num_thread": NUM_THREAD} if NUM_THREAD > 0 else {}
    llm = Ollama(
        model=AGENT_MODEL,
        base_url=OLLAMA_BASE_URL,
        thinking=False,            # qwen3 інакше пише <think> перед кожним кроком
        context_window=NUM_CTX,    # іде в Ollama як num_ctx: менше RAM
        request_timeout=900.0,     # на CPU один крок може йти хвилину+
        additional_kwargs=extra,
    )
    tools = [FunctionTool.from_defaults(fn=search_arxiv),
             FunctionTool.from_defaults(fn=write_report)]
    return FunctionAgent(
        tools=tools,
        llm=llm,
        system_prompt=SYSTEM_PROMPT,
        # Без цього агент, що заплутався, падає з WorkflowRuntimeError.
        early_stopping_method="generate",
    )


async def run_agent(agent: FunctionAgent, task: str) -> str:
    """Запускає агента і друкує КОЖЕН виклик інструмента (стрімінг подій)."""
    handler = agent.run(user_msg=task, max_iterations=10)
    async for ev in handler.stream_events():
        if isinstance(ev, ToolCall):
            args = {k: str(v)[:60] for k, v in ev.tool_kwargs.items()}
            print(f"  [tool] {ev.tool_name}({args})")
        elif isinstance(ev, ToolCallResult):
            print(f"  [out ] {str(ev.tool_output).replace(chr(10), ' ')[:150]}…")
    return str(await handler)


def extract_report(text: str) -> str:
    """Лишає з відповіді тільки блоки шаблону (`---` + `### <назва>` ...)."""
    parts = re.split(r"(?m)^---\s*$", text)
    blocks = [p.strip() for p in parts if p.strip().startswith("### ")]
    return "\n\n".join(f"---\n{b}\n---" for b in blocks)


async def main() -> int:
    print("=" * 56)
    print("  ЛР №2 — «Науковий Дайджест» на LlamaIndex")
    print("=" * 56)
    print(f"  [agent] {AGENT_MODEL}, num_ctx={NUM_CTX}, think=off")
    print(f"\n> Завдання: {TASK}\n")
    started = time.time()
    try:
        answer = await run_agent(build_agent(), TASK)
    except Exception as exc:
        print(f"\n[!] Помилка виконання: {type(exc).__name__}: {exc}")
        return 1
    print(f"\nOK: Готово за {time.time() - started:.0f} c")
    print(f"  Відповідь агента: {answer[:300]}{'…' if len(answer) > 300 else ''}")
    report = REPORTS_DIR / "digest.md"
    if not report.exists():
        # Мала модель інколи друкує звіт текстом замість виклику інструмента.
        # Страховка: витягуємо з відповіді лише блоки за шаблоном і зберігаємо.
        recovered = extract_report(answer)
        if recovered:
            print("\n[fallback] модель не викликала write_report: зберігаю її звіт сам.")
            print(f"  [out ] {write_report('digest.md', recovered)}")
    if report.exists():
        text = report.read_text(encoding="utf-8")
        print(f"\n[file] {report} — {report.stat().st_size} байт, статей: {text.count('### ')}")
        print(text[:1200])
    else:
        print("\n[!] Звіт не створено: модель не викликала write_report.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="main.py", description="ЛР №2: Науковий Дайджест")
    parser.add_argument("--show-toolspec", action="store_true",
                        help="показати, що віддає ArxivToolSpec, і вийти")
    args = parser.parse_args()
    if args.show_toolspec:
        show_toolspec()
        sys.exit(0)
    problems = check_environment()
    if problems:
        print("\nЗапуск неможливий:")
        for line in problems:
            print(line if line.startswith(" ") else f"[!] {line}")
        sys.exit(0)
    sys.exit(asyncio.run(main()))
