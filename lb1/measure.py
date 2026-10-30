"""Вимірює, скільки прогонів сценарію дали правильну Markdown-таблицю в усіх 4 відповідях."""
import re
import sys

import main
from main import MemoryManager, OllamaEmbedder, attach_memory, run_turn, DEMO_STEPS

# очікувані (речення, найдовше слово) для кожного з 4 ходів
EXPECTED = [(3, "Сьогодні"), (2, "Програмування"), (3, "Сьогодні"), (2, "Програмування")]


def table_ok(answer: str, sentences: int, word: str) -> bool:
    flat = answer.replace(" ", "")
    return (
        "|---" in flat
        and re.search(rf"\|\s*Кількість речень\s*\|\s*{sentences}\s*\|", answer) is not None
        and re.search(rf"\|\s*Найдовше слово\s*\|\s*{re.escape(word)}\s*\|", answer, re.IGNORECASE) is not None
    )


if __name__ == "__main__":
    problems = main.check_environment()
    if problems:
        print("\n".join(problems))
        sys.exit(1)
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    main.memory_manager = MemoryManager(OllamaEmbedder())
    attach_memory(main.memory_manager)
    ok = 0
    for r in range(1, runs + 1):
        main.memory_manager.reset()
        results = [table_ok(run_turn(t), *EXPECTED[i]) for i, t in enumerate(DEMO_STEPS)]
        print(f"\n>>> Прогін {r}: {results}")
        ok += all(results)
    print(f"\n=== Правильна таблиця в усіх 4 відповідях: {ok}/{runs} прогонів ===")
