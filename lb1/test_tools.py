import hashlib
import re

import numpy as np
import pytest

import main
from main import MemoryManager, l2_normalize, attach_memory


class FakeEmbedder:
    """Підставний ембедер: мішок слів у хешованому просторі. Без Ollama."""
    dimension = 1024

    def embed(self, texts):
        out = np.zeros((len(texts), self.dimension), dtype="float32")
        for i, t in enumerate(texts):
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                out[i, h % self.dimension] += 1.0
        return l2_normalize(out)


@pytest.fixture
def memory():
    m = MemoryManager(FakeEmbedder())
    attach_memory(m)
    return m


@pytest.mark.parametrize("text,expected", [
    ("Привіт. Як справи? Добре!", 3),
    ("Одне речення без крапки", 1),
    ("Число 3.14 не ділить речення.", 1),
    ("Що?! Справді... Так.", 3),
    ("", 0),
    ("   ", 0),
])
def test_count_sentences(text, expected):
    assert main.count_sentences(text) == expected


def test_longest_word():
    assert main.find_longest_word("Кіт і програмування") == "програмування"


def test_longest_word_ignores_punctuation():
    assert main.find_longest_word("Hello, wonderful!!!") == "wonderful"


def test_longest_word_tie_returns_first():
    assert main.find_longest_word("кіт пес") == "кіт"


def test_longest_word_hyphen_and_apostrophe():
    assert main.find_longest_word("а що-небудь б") == "що-небудь"
    assert main.find_longest_word("м'яч і ліс") == "м'яч"


def test_longest_word_empty():
    assert main.find_longest_word("... !!!").startswith("ERROR")


def test_demo_texts():
    assert main.count_sentences("Сьогодні гарна погода. Ми йдемо гуляти! Чи підеш ти з нами?") == 3
    assert main.find_longest_word("Сьогодні гарна погода. Ми йдемо гуляти! Чи підеш ти з нами?") == "Сьогодні"


def test_save_and_search(memory):
    main.save_fact_to_memory("analysis of text 'weather'", "sentences: 3; longest word: Сьогодні")
    assert len(memory) == 1
    assert memory.search("analysis of text weather")


def test_threshold_drops_irrelevant(memory):
    memory.add_memory("кіт сидить на дивані")
    assert memory.search("квантова хромодинаміка") == []


def test_empty_memory_rejected(memory):
    with pytest.raises(ValueError):
        memory.add_memory("   ")


def test_reset(memory):
    memory.add_memory("якийсь факт")
    memory.reset()
    assert len(memory) == 0
    assert memory.search("якийсь факт") == []


def test_answer_from_context_rejects_question():
    main._CURRENT_TASK = "Скільки речень було в тексті?"
    assert main.answer_from_context("Скільки речень було в тексті?").startswith("ERROR")


def test_answer_from_context_accepts_answer():
    main._CURRENT_TASK = "Скільки речень було в тексті?"
    assert main.answer_from_context("У тексті було 3 речення.") == "У тексті було 3 речення."


def test_duplicate_memory_not_stored(memory):
    memory.add_memory("факт про кота")
    memory.add_memory("факт про кота")
    assert len(memory) == 1


def test_recall_hint_added_when_no_new_text():
    out = main.build_task_with_memory("Скільки речень було в тексті про погоду?", "факт")
    assert "answer_from_context" in out
    assert out.rstrip().endswith("Скільки речень було в тексті про погоду?")


def test_no_hint_when_task_has_new_text():
    task = "Проаналізуй текст 'Програмування вимагає терпіння. Результат вартий зусиль.'"
    out = main.build_task_with_memory(task, "факт")
    assert "NOTE:" not in out


def test_no_memory_no_changes():
    assert main.build_task_with_memory("щось", "") == "щось"
