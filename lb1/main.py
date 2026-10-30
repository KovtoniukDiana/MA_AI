import argparse
import os
import re
import sys
import time

import faiss
import numpy as np
import requests
from smolagents import tool, CodeAgent, LiteLLMModel

try:  # необов'язково: читає налаштування з файла .env
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def setup_console() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


setup_console()

OLLAMA_BASE_URL = "http://localhost:11434"
# Усе, що впливає на швидкість, можна міняти через .env без правки коду.
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3:1.7b")   # легка модель для слабкого ноутбука
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3:latest")
NUM_CTX = int(os.getenv("NUM_CTX", "4096"))            # менше контексту = менше RAM і швидше
MAX_OUT_TOKENS = int(os.getenv("MAX_OUT_TOKENS", "400"))  # відповідь агента невелика
NUM_THREAD = int(os.getenv("NUM_THREAD", "0"))         # 0 = за замовчуванням; інакше К-сть потоків CPU
MEMORY_MIN_SCORE = 0.40


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    """Після нормування скалярний добуток дорівнює косинусній близькості."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (vectors / norms).astype("float32")


class OllamaEmbedder:
    """Рахує L2-нормовані вектори текстів локальним сервером Ollama."""

    def __init__(self, model: str = EMBED_MODEL, base_url: str = OLLAMA_BASE_URL):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._dimension = None

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._dimension = len(self.embed(["dimension probe"])[0])
        return self._dimension

    def embed(self, texts: list) -> np.ndarray:
        resp = requests.post(
            f"{self.base_url}/api/embed",
            json={"model": self.model, "input": texts},
            timeout=120,
        )
        resp.raise_for_status()
        return l2_normalize(np.asarray(resp.json()["embeddings"], dtype="float32"))


class MemoryManager:
    def __init__(self, embedder, min_score: float = MEMORY_MIN_SCORE):
        print("Ініціалізація Менеджера Пам'яті...")
        self.embedder = embedder
        self.min_score = min_score
        self.dimension = embedder.dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.memories = []
        print(f"Менеджер Пам'яті готовий: {self.dimension} вимірів.")

    def __len__(self) -> int:
        return len(self.memories)

    def reset(self) -> None:
        self.index = faiss.IndexFlatIP(self.dimension)
        self.memories = []

    def add_memory(self, text: str) -> None:
        text = (text or "").strip()
        if not text:
            raise ValueError("Порожній спогад зберігати немає сенсу")
        if text in self.memories:  # той самий факт двічі не зберігаємо
            print("  [memory] такий спогад уже є, пропускаю")
            return
        self.index.add(self.embedder.embed([text]))
        self.memories.append(text)
        print(f"  [memory] збережено #{len(self.memories)}: {text[:70]}")

    def search(self, query: str, k: int = 3) -> list:
        if not self.memories or k <= 0:
            return []
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(self.embedder.embed([query]), k)
        hits = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0 or float(score) < self.min_score:
                continue
            hits.append((self.memories[int(idx)], float(score)))
        return hits

    def retrieve_relevant_memory(self, query: str, k: int = 3) -> str:
        hits = self.search(query, k)
        if not hits:
            print("  [memory] релевантних спогадів немає")
            return ""
        for text, score in hits:
            print(f"  [memory] знайдено (cos={score:.2f}): {text[:70]}")
        return "\n".join(text for text, _ in hits)


# ---------------------------------------------------------------- інструменти
_MEMORY = None


def attach_memory(memory) -> None:
    global _MEMORY
    _MEMORY = memory


def get_memory():
    if _MEMORY is None:
        raise RuntimeError("Пам'ять не ініціалізована: див. attach_memory()")
    return _MEMORY


@tool
def count_words(text: str) -> int:
    """
    Counts the number of words in a text.
    Args:
        text: The text to analyze.
    """
    print(f"  [tool] count_words({text!r})")
    return len(text.split())


@tool
def reverse_text(text: str) -> str:
    """
    Reverses a text character by character.
    Args:
        text: The text to reverse.
    """
    print(f"  [tool] reverse_text({text!r})")
    return text[::-1]


@tool
def save_fact_to_memory(fact_description: str, fact_content: str) -> str:
    """
    Saves a fact to long-term memory so it can be recalled in later turns.
    Call this whenever the user asks you to remember something, or after you
    compute a result the user may ask about again.
    Args:
        fact_description: Short description, e.g. "user's favorite color".
        fact_content: The value of the fact, e.g. "blue".
    """
    get_memory().add_memory(f"Fact about '{fact_description}': '{fact_content}'")
    return "Fact saved to long-term memory."


_CURRENT_TASK = ""


def _looks_like_the_question(answer: str) -> bool:
    a = answer.strip().lower().rstrip("?!.")
    q = _CURRENT_TASK.strip().lower().rstrip("?!.")
    if not a or a == q:
        return True
    if answer.strip().endswith("?") and not any(ch.isdigit() for ch in answer):
        return True
    aw, qw = set(a.split()), set(q.split())
    return bool(qw) and len(aw & qw) / max(len(aw), 1) > 0.8


@tool
def answer_from_context(answer_text: str) -> str:
    """
    Use this when the answer is already present in the Memory block and no
    computation is needed. Pass the ANSWER itself, never the user's question.
    Args:
        answer_text: The final answer for the user, in Ukrainian, stating the fact.
    """
    print(f"  [tool] answer_from_context({answer_text!r})")
    if _looks_like_the_question(answer_text):
        return ("ERROR: you passed the question, not the answer. Read the Memory "
                "block above, find the stored fact, and call this tool again with "
                "the fact itself (in Ukrainian).")
    return answer_text


# --- інструменти варіанта 1 ---
@tool
def count_sentences(text: str) -> int:
    """
    Counts the number of sentences in a text. A sentence ends with ".", "!",
    "?" or "…" followed by whitespace or the end of the text.
    Args:
        text: The text to analyze.
    """
    print(f"  [tool] count_sentences({text!r})")
    parts = re.split(r"(?<=[.!?…])\s+", text.strip())
    return sum(1 for p in parts if re.search(r"\w", p))


@tool
def find_longest_word(text: str) -> str:
    """
    Returns the longest word in a text (the first one if several are equally
    long). Punctuation is ignored; hyphenated words and words with an
    apostrophe count as one word. Returns an ERROR string if there are no words.
    Args:
        text: The text to analyze.
    """
    print(f"  [tool] find_longest_word({text!r})")
    words = re.findall(r"[^\W_]+(?:[-'’][^\W_]+)*", text)
    if not words:
        return "ERROR: no words found in the text."
    return max(words, key=len)


TOOLS = [count_words, reverse_text, save_fact_to_memory, answer_from_context,
         count_sentences, find_longest_word]

# ---------------------------------------------------------------- агент
_model_kwargs = {}
if NUM_THREAD > 0:
    _model_kwargs["num_thread"] = NUM_THREAD

model = LiteLLMModel(
    model_id=f"ollama_chat/{AGENT_MODEL}",
    api_base=OLLAMA_BASE_URL,
    num_ctx=NUM_CTX,
    max_tokens=MAX_OUT_TOKENS,
    reasoning_effort="none",
    **_model_kwargs,
)

INSTRUCTIONS = r"""You are a Python assistant with a long-term vector memory.
Always start your answer with a plan inside <thought> and </thought> tags.
Rules:
1. A "## Memory" block may be prepended to the task. If the task asks about a
   text analyzed EARLIER (the task itself does not contain that text) and the
   Memory block holds "sentences: N; longest word: W" for it, DO NOT call
   count_sentences, find_longest_word or save_fact_to_memory. Read N and W from
   the Memory block, call answer_from_context(answer_text=...) with the ANSWER
   and pass its result to final_answer(). If the task contains a NEW text in
   quotes, ignore the Memory block and analyze that text (rule 3).
2. Otherwise solve the task with the tools. Never count sentences or find the
   longest word in your head: use count_sentences and find_longest_word.
3. Whenever you analyze a text, FIRST put it into a variable: text = "...".
   Then call BOTH count_sentences(text=text) and find_longest_word(text=text),
   then call save_fact_to_memory BEFORE final_answer(), using the variable
   instead of retyping the text:
   save_fact_to_memory(fact_description=f"analysis of text '{text}'",
   fact_content=f"sentences: {s}; longest word: {w}")
   Never type the analyzed text twice.
4. Always call tools with keyword arguments.
5. The last line of your code must always be final_answer(...).
6. EVERY final answer must be exactly this Markdown table, in Ukrainian,
   with nothing before or after it:
   | Показник | Значення |
   |---|---|
   | Кількість речень | <number> |
   | Найдовше слово | <word> |
   This holds for answers taken from memory too. Never answer with plain text.
Example A - analyze a new text, save, answer with the table:
## Current task
Analyze the text 'Hello world. How are you?' and remember the result.
<thought>
I will count sentences, find the longest word, save both, answer with a table.
</thought>
<code>
text = "Hello world. How are you?"
s = count_sentences(text=text)
w = find_longest_word(text=text)
save_fact_to_memory(fact_description=f"analysis of text '{text}'", fact_content=f"sentences: {s}; longest word: {w}")
final_answer(f"| Показник | Значення |\n|---|---|\n| Кількість речень | {s} |\n| Найдовше слово | {w} |")
</code>
Example B - the Memory block already holds the answer:
## Memory (facts you saved in earlier turns)
Fact about 'analysis of text 'Good morning. Nice day.'': 'sentences: 2; longest word: morning'

NOTE: the task below contains no new text, so it asks about something you
already analyzed. The answer is in the facts above. Do NOT call
count_sentences, find_longest_word or save_fact_to_memory: read the numbers
from the facts and call answer_from_context.

## Current task
What was the longest word and how many sentences were in the text about the morning?
<thought>
The memory block answers this. I will put it into the table.
</thought>
<code>
answer = answer_from_context(answer_text="| Показник | Значення |\n|---|---|\n| Кількість речень | 2 |\n| Найдовше слово | morning |")
final_answer(answer)
</code>
"""

agent = CodeAgent(
    tools=TOOLS,
    model=model,
    instructions=INSTRUCTIONS,
    max_steps=4,
    verbosity_level=0,
)

# ---------------------------------------------------------------- оркестрація
DEMO_STEPS = [
    "Проаналізуй текст 'Сьогодні гарна погода. Ми йдемо гуляти! Чи підеш ти з нами?' і запам'ятай результат.",
    "Проаналізуй текст 'Програмування вимагає терпіння. Результат вартий зусиль.' і запам'ятай результат.",
    "Скільки речень було в тексті про погоду і яке там найдовше слово?",
    "Скільки речень було в тексті про програмування і яке там найдовше слово?",
]

memory_manager = None


def check_environment() -> list:
    """Повертає список проблем. Порожній список означає, що все гаразд."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        available = {m["name"] for m in resp.json().get("models", [])}
    except Exception as exc:
        return [f"Ollama не відповідає на {OLLAMA_BASE_URL} ({type(exc).__name__}).",
                "  Windows/macOS: відкрийте застосунок Ollama; Linux: `ollama serve`"]
    problems = []
    for name in (AGENT_MODEL, EMBED_MODEL):
        if name not in available and f"{name}:latest" not in available:
            problems.append(f"Модель '{name}' не знайдена в Ollama.")
            problems.append(f"  Виправити: ollama pull {name.split(':')[0]}"
                            + (f":{name.split(':')[1]}" if ":" in name and not name.endswith(":latest") else ""))
    return problems


RECALL_HINT = (
    "NOTE: the task below contains no new text, so it asks about something you\n"
    "already analyzed. The answer is in the facts above. Do NOT call\n"
    "count_sentences, find_longest_word or save_fact_to_memory: read the numbers\n"
    "from the facts and call answer_from_context."
)


def has_quoted_text(task: str) -> bool:
    """Чи містить запит новий текст у лапках (тоді його треба аналізувати)."""
    return re.search(r"['\"«“‘].{8,}?['\"»”’]", task) is not None


def build_task_with_memory(task: str, context: str) -> str:
    if not context:
        return task
    hint = "" if has_quoted_text(task) else RECALL_HINT + "\n\n"
    return (
        "## Memory (facts you saved in earlier turns)\n"
        f"{context}\n\n"
        f"{hint}"
        "## Current task\n"
        f"{task}"
    )


def run_turn(task: str, k: int = 3) -> str:
    """Один хід: пошук у пам'яті, промпт з контекстом, запуск агента."""
    global _CURRENT_TASK
    _CURRENT_TASK = task
    context = memory_manager.retrieve_relevant_memory(task, k=k)
    started = time.time()
    try:
        result = agent.run(build_task_with_memory(task, context), reset=True)
    except Exception as exc:
        return f"[!] Помилка виконання: {type(exc).__name__}: {exc}"
    print(f"  [time] {time.time() - started:.1f} c")
    return str(result)


def run_demo() -> int:
    print("\n--- Демо-сценарій: перевіряємо довготривалу пам'ять ---")
    for number, task in enumerate(DEMO_STEPS, 1):
        print(f"\n[{number}/{len(DEMO_STEPS)}] > {task}")
        print(f"  <- {run_turn(task)}")
    print(f"\n--- Готово. У пам'яті {len(memory_manager)} фактів. ---")
    return 0


def run_chat() -> int:
    print("\n--- Інтерактивний режим з RAG-пам'яттю ---")
    print("'/new' - очистити пам'ять, 'exit' або Ctrl+D - вихід.")
    while True:
        try:
            task = input("\n> Ваше завдання: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nЗавершено.")
            return 0
        if not task:
            continue
        if task.lower() in {"exit", "quit", "вихід"}:
            return 0
        if task.lower() == "/new":
            memory_manager.reset()
            print("Пам'ять очищено, починаємо новий чат.")
            continue
        print(f"  <- {run_turn(task)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ЛР №1: AI-агент на smolagents з довготривалою RAG-пам'яттю.")
    parser.add_argument("--chat", action="store_true",
                        help="інтерактивний діалог замість демо-сценарію")
    args = parser.parse_args()

    problems = check_environment()
    if problems:
        print("\nЗапуск неможливий:")
        for line in problems:
            print(line if line.startswith(" ") else f"[!] {line}")
        sys.exit(0)

    memory_manager = MemoryManager(OllamaEmbedder())
    attach_memory(memory_manager)
    sys.exit(run_chat() if args.chat else run_demo())
