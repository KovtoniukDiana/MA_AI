"""ЛР №3: агент «AI-Конвертер» на LangChain (create_agent + Tavily + власні інструменти).

Запуск:  python lab_main.py
Потрібні ключі в .env: GOOGLE_API_KEY, TAVILY_API_KEY.
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path

# Консоль Windows стартує у cp1251 і падає на кирилиці. Робимо це ПЕРЕД print().
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

from langchain_core.tools import tool
from pydantic import BaseModel, Field

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
# Єдина тека, у яку агенту дозволено писати файли.
OUTPUT_DIR = Path(__file__).resolve().parent / "artifacts"

TASK = (
    "Мені потрібно замовити 2500 футів арматури для об'єкта в Європі. "
    "Переведи цю довжину в метри. Також знайди в інтернеті стандартну вагу "
    "одного метра арматури діаметром 12 мм і порахуй загальну вагу замовлення. "
    "Збережи результати у файл conversion_report.txt і вкажи джерело."
)

# ── Таблиці перерахунку: кожна категорія має базову одиницю ──────────────
UNITS = {
    "length": {"m": 1.0, "km": 1000.0, "cm": 0.01, "mm": 0.001,
               "ft": 0.3048, "in": 0.0254, "yd": 0.9144, "mi": 1609.344},
    "mass": {"kg": 1.0, "g": 0.001, "t": 1000.0,
             "lb": 0.45359237, "oz": 0.028349523125},
    "area": {"m2": 1.0, "cm2": 1e-4, "ha": 10000.0,
             "ft2": 0.09290304, "in2": 6.4516e-4},
    "pressure": {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "bar": 1e5,
                 "psi": 6894.757293168, "atm": 101325.0},
}
ALIASES = {
    "foot": "ft", "feet": "ft", "meter": "m", "metre": "m", "meters": "m",
    "metres": "m", "inch": "in", "inches": "in", "yard": "yd", "yards": "yd",
    "mile": "mi", "miles": "mi", "kilometer": "km", "kilometre": "km",
    "pound": "lb", "pounds": "lb", "lbs": "lb", "ounce": "oz", "tonne": "t",
    "ton": "t", "gram": "g", "kilogram": "kg", "sqft": "ft2", "sqm": "m2",
}


def _norm_unit(unit: str) -> str:
    u = str(unit).strip().lower().replace("²", "2").replace(".", "").replace(" ", "")
    return ALIASES.get(u, u)


def convert_value(value: float, from_unit: str, to_unit: str) -> float:
    """Чиста функція перерахунку (її й тестуємо). Кидає ValueError."""
    f, t = _norm_unit(from_unit), _norm_unit(to_unit)
    for table in UNITS.values():
        if f in table and t in table:
            return value * table[f] / table[t]
    known = sorted({u for tbl in UNITS.values() for u in tbl})
    if not any(f in tbl for tbl in UNITS.values()) or not any(t in tbl for tbl in UNITS.values()):
        raise ValueError(f"Unknown unit in '{from_unit}' -> '{to_unit}'. Known: {', '.join(known)}")
    raise ValueError(f"Cannot convert '{from_unit}' to '{to_unit}': different quantities")


def safe_output_path(filename: str) -> Path:
    """Ім'я файлу приходить від моделі = недовірений ввід. Лишаємо тільки
    базове ім'я всередині OUTPUT_DIR; усе інше зрізаємо або відхиляємо."""
    raw = str(filename).strip()
    if not raw or "\x00" in raw:
        raise ValueError("Порожнє або некоректне ім'я файлу")
    name = raw.replace("\\", "/").rstrip("/").rsplit("/", 1)[-1].strip()
    if not name or name in {".", ".."}:
        raise ValueError(f"Некоректне ім'я файлу: {filename!r}")
    reserved = {"con", "prn", "aux", "nul",
                *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}
    if name.split(".")[0].lower() in reserved:
        raise ValueError(f"Зарезервоване ім'я файлу Windows: {name!r}")
    base = Path(OUTPUT_DIR).resolve()
    base.mkdir(parents=True, exist_ok=True)
    target = (base / name).resolve()
    if not target.is_relative_to(base):  # страхувальний пояс
        raise ValueError(f"Шлях виходить за межі робочої теки: {filename!r}")
    return target


# ── Інструмент 1: пакетна конвертація (результат УСІЄЇ дії) ──────────────
class ConversionItem(BaseModel):
    value: float = Field(description="Number to convert")
    from_unit: str = Field(description="Source unit, e.g. ft, m, lb, kg, psi, MPa")
    to_unit: str = Field(description="Target unit, e.g. m, ft, kg, MPa")


class ConvertArgs(BaseModel):
    items: list[ConversionItem] = Field(description="ALL conversions at once")


ConvertArgs.model_rebuild()


@tool(args_schema=ConvertArgs)
def convert_units(items: list[ConversionItem]) -> str:
    """Converts values between units (length, mass, area, pressure).
    Pass ALL conversions in ONE call. Returns every result; never convert yourself."""
    results = []
    for raw in items:
        it = raw if isinstance(raw, dict) else raw.model_dump()
        try:
            out = convert_value(float(it["value"]), it["from_unit"], it["to_unit"])
            results.append({"input": f"{it['value']} {it['from_unit']}",
                            "result": f"{out:.6g} {it['to_unit']}"})
        except ValueError as exc:
            results.append({"input": f"{it['value']} {it['from_unit']}", "error": str(exc)})
    return json.dumps({"conversions": results}, ensure_ascii=False, indent=2)


# ── Інструмент 2: вага замовлення (перерахунок + підсумок в одному виклику) ──
@tool
def order_weight(length: float, length_unit: str, kg_per_meter: float) -> str:
    """Computes the FULL result for a rebar order: converts the length to meters
    and returns total weight in kg and tonnes. Use after you know kg per meter.

    Args:
        length: Ordered length, e.g. 2500
        length_unit: Unit of the length, e.g. ft or m
        kg_per_meter: Weight of one meter of the product in kg
    """
    if length <= 0 or kg_per_meter <= 0:
        return "ERROR: length and kg_per_meter must be positive numbers."
    try:
        meters = convert_value(length, length_unit, "m")
    except ValueError as exc:
        return f"ERROR: {exc}"
    total_kg = meters * kg_per_meter
    return json.dumps({
        "length_m": f"{meters:.2f}",
        "kg_per_meter": f"{kg_per_meter:g}",
        "total_weight_kg": f"{total_kg:.2f}",
        "total_weight_tonnes": f"{total_kg / 1000:.3f}",
        "note": "Use these numbers as is. Do not recalculate.",
    }, ensure_ascii=False, indent=2)


# ── Інструмент 3: запис файлу (ім'я від моделі - недовірений ввід) ──────
@tool
def write_report(filename: str, content: str) -> str:
    """Saves the final report to a text file. Call this ONCE, at the very end.

    Args:
        filename: Plain file name, e.g. conversion_report.txt
        content: Full report text
    """
    try:
        target = safe_output_path(filename)
    except ValueError as exc:
        return f"Refused: {exc}. Pass a plain file name."
    target.write_text(content, encoding="utf-8")
    return f"Saved: {target.name}. Task complete."


SYSTEM_PROMPT = """Ти «AI-Конвертер», асистент інженера-будівельника.
1. НІКОЛИ не переводь одиниці й не рахуй сам: використовуй convert_units
   (усі перетворення одним викликом) або order_weight.
2. Якщо потрібне довідкове значення (наприклад, вага 1 м арматури), знайди його
   через tavily_search ОДИН раз (запит англійською, напр. "rebar 12 mm weight
   per meter kg/m"), візьми число з результатів і запам'ятай URL
   джерела. Якщо числа в результатах немає - так і скажи, не вгадуй.
3. Називай числа так, як їх повернули інструменти.
4. Якщо просять зберегти - НЕ друкуй звіт у відповіді, а виклич write_report
   один раз наприкінці: у content мають бути всі результати і URL джерела.
5. Фінальна відповідь - коротка, українською."""


def build_agent():
    from dotenv import load_dotenv
    from langchain.agents import create_agent  # НЕ langgraph.prebuilt.create_react_agent
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_tavily import TavilySearch

    load_dotenv()
    llm = ChatGoogleGenerativeAI(model=GEMINI_MODEL, max_retries=2)
    tools = [TavilySearch(max_results=3), convert_units, order_weight, write_report]
    return create_agent(model=llm, tools=tools, system_prompt=SYSTEM_PROMPT)


def _text(msg) -> str:
    t = getattr(msg, "text", "")
    # У новому langchain .text - властивість (str); у старому - метод.
    if not isinstance(t, str) and callable(t):
        t = t()
    return (t or "").strip()


async def run_agent(agent, task: str) -> str:
    """Стрімить кроки агента: що він вирішив викликати і що йому повернули."""
    from langchain_core.messages import AIMessage, ToolMessage

    final = ""
    async for chunk in agent.astream({"messages": [("user", task)]}, stream_mode="updates"):
        for update in chunk.values():
            for msg in (update or {}).get("messages", []):
                if isinstance(msg, AIMessage):
                    if _text(msg):
                        print(f"  [model] {_text(msg)[:300]}")
                        final = _text(msg)
                    for call in msg.tool_calls:
                        args = {k: str(v)[:70] for k, v in call["args"].items()}
                        print(f"  [tool ] {call['name']}({args})")
                elif isinstance(msg, ToolMessage):
                    print(f"  [out  ] {str(msg.content).replace(chr(10), ' ')[:200]}…")
    return final


def _quiet_ssl_shutdown(loop, context) -> None:
    """Ігнорує безпечний галас Windows при закритті SSL-з'єднань наприкінці."""
    text = f"{context.get('message', '')} {context.get('exception', '')}"
    if "SSL" in text or "Event loop is closed" in text:
        return
    loop.default_exception_handler(context)


async def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    asyncio.get_running_loop().set_exception_handler(_quiet_ssl_shutdown)
    missing = [k for k in ("GOOGLE_API_KEY", "TAVILY_API_KEY") if not os.getenv(k)]
    if missing:
        print(f"[!] У .env немає: {', '.join(missing)}")
        print("    GOOGLE_API_KEY - https://aistudio.google.com/, TAVILY_API_KEY - https://www.tavily.com/")
        return 0
    print("=" * 56)
    print("  ЛР №3 - «AI-Конвертер» на LangChain")
    print("=" * 56)
    print(f"  [agent] {GEMINI_MODEL}\n\n> Завдання: {TASK}\n")
    started = time.time()
    try:
        answer = await run_agent(build_agent(), TASK)
    except Exception as exc:
        print(f"\n[!] Помилка виконання: {type(exc).__name__}: {exc}")
        return 1
    print(f"\nOK: Готово за {time.time() - started:.0f} c\n  Відповідь агента: {answer}")
    report = OUTPUT_DIR / "conversion_report.txt"
    if report.exists():
        print(f"\n[file] {report} - {report.stat().st_size} байт\n")
        print(report.read_text(encoding="utf-8")[:1200])
    else:
        print("\n[!] Звіт не створено: модель не викликала write_report.")
    await asyncio.sleep(0.5)  # даємо HTTP-з'єднанням закритися до зупинки циклу
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        # Proactor-цикл Windows друкує «Fatal error on SSL transport» при виході.
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    sys.exit(asyncio.run(main()))