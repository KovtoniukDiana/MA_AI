"""MCP-сервер «Library Manager»: бібліотека (книги, читачі, видачі) на SQLite.

Запуск:  python server.py          (stdio-транспорт, його запускає клієнт)
Інспектор: npx @modelcontextprotocol/inspector@latest python server.py
"""
import json
import sqlite3
import sys
from contextlib import closing, contextmanager
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import psutil

# SDK 2.x: FastMCP перейменовано на MCPServer, модуль mcp.server.fastmcp
# прибрано. Шим дає працювати на обох версіях.
try:
    from mcp.server.mcpserver import MCPServer as ServerClass  # SDK 2.x
except ModuleNotFoundError:
    from mcp.server.fastmcp import FastMCP as ServerClass  # SDK 1.x
from mcp.types import ToolAnnotations

from setup_db import DB_FILE, MAX_ACTIVE_LOANS

MAX_ROWS = 200            # скільки рядків віддаємо моделі за один запит
LOAN_DAYS = 14            # стандартний строк видачі
PROGRESS_STEP = 1000      # як часто SQLite кличе обробник (у VM-інструкціях)
MAX_PROGRESS_CALLS = 2000  # бюджет: ~2 млн інструкцій на один запит

mcp = ServerClass(
    "Library Manager",
    instructions=(
        "Бібліотека: книги, читачі, видачі. Для довільних питань до даних "
        "читай схему з ресурсу library://schema й пиши ОДИН SELECT через "
        "read_library_query. Для видачі й повернення книг користуйся "
        "issue_book та return_book, а не SQL."
    ),
)


# ---------------------------------------------------------------- допоміжне

class LibraryError(Exception):
    """Очікувана помилка предметної області: текст піде моделі як є."""


def _fold(value: Any) -> Any:
    """Регістронезалежне порівняння. Власний LIKE у SQLite знає регістр лише
    для ASCII, тож «фантастика» не знайшла б «Фантастика»."""
    return value.casefold() if isinstance(value, str) else value


def _like(text: str) -> str:
    """Шаблон для LIKE ... ESCAPE '\\': спецсимволи % і _ шукаємо буквально."""
    esc = _fold(text.strip()).replace("\\", "\\\\")
    esc = esc.replace("%", "\\%").replace("_", "\\_")
    return f"%{esc}%"


def _today() -> date:
    return date.today()


def _prepare(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.row_factory = sqlite3.Row
    conn.create_function("fold", 1, _fold, deterministic=True)
    return conn


# Операції, потрібні звичайному SELECT. Решту авторизатор відкине.
ALLOWED = frozenset({
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    sqlite3.SQLITE_RECURSIVE,
})
DENIED_FUNCTIONS = frozenset({"load_extension"})


def _authorizer(action, arg1, arg2, db_name, source):
    if action not in ALLOWED:
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in DENIED_FUNCTIONS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def connect_ro() -> sqlite3.Connection:
    """З'єднання ТІЛЬКИ ДЛЯ ЧИТАННЯ: для SQL, який прислала модель.

    Три незалежні шари захисту (перевірка рядка query.startswith("SELECT")
    не захищає ні від одного з цих випадків):
      1. mode=ro: рушій БД не відкриває файл на запис;
      2. авторизатор: дозволені лише SELECT/READ/FUNCTION/RECURSIVE, тож
         ATTACH, PRAGMA, DDL та DML відхиляються ще на етапі компіляції.
         mode=ro стосується лише головної бази: ATTACH приєднав би
         новий файл із правом запису, цю дірку закриває саме авторизатор;
      3. бюджет кроків + ліміти довжини: рекурсивний запит, що починається
         зі слова SELECT, зупиняється, а не крутить мільйони ітерацій.
    """
    # as_uri() сам екранує пробіли, кирилицю та # ? % у шляху.
    uri = Path(DB_FILE).resolve().as_uri() + "?mode=ro"
    conn = _prepare(sqlite3.connect(uri, uri=True))
    conn.set_authorizer(_authorizer)

    budget = {"calls": 0}

    def guard() -> int:
        budget["calls"] += 1
        return 1 if budget["calls"] > MAX_PROGRESS_CALLS else 0  # 1 = скасувати

    conn.set_progress_handler(guard, PROGRESS_STEP)
    # Ліміти рушія: один гігантський рядок (zeroblob(1e9)) не з'їсть пам'ять.
   
    if hasattr(conn, "setlimit"):
        conn.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        conn.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 10_000)
    return conn


@contextmanager
def transaction():
    """Транзакція на запис: або виконано все, або нічого.

    Коректність наявності гарантує умова всередині UPDATE ... WHERE. Транзакція
    додає інше: списання примірника й запис видачі виконуються разом або не
    виконуються взагалі (будь-який виняток відкочує всі зміни). BEGIN IMMEDIATE
    бере блокування запису на старті, тож паралельні виклики стають у чергу,
    а не конфліктують під час підвищення блокування.
    """
    conn = _prepare(sqlite3.connect(DB_FILE, timeout=10.0, isolation_level=None))
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        conn.execute("BEGIN IMMEDIATE")
        yield conn
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def _find_one(conn, rows_exact, rows_like, what: str, text: str, label) -> sqlite3.Row:
    """Точний збіг має пріоритет; інакше частковий збіг має бути єдиним."""
    exact = conn.execute(*rows_exact).fetchall()
    if len(exact) == 1:
        return exact[0]
    found = exact or conn.execute(*rows_like).fetchall()
    if not found:
        raise LibraryError(f"{what} «{text}» не знайдено.")
    if len(found) > 1:
        names = "; ".join(label(r) for r in found[:6])
        raise LibraryError(f"«{text}» підходить кільком записам: {names}. Уточніть.")
    return found[0]


def _find_book(conn, text: str) -> sqlite3.Row:
    cols = "SELECT id, title, author, copies_total, copies_available FROM books"
    return _find_one(
        conn,
        (cols + " WHERE fold(title) = fold(?)", (text.strip(),)),
        (cols + " WHERE fold(title) LIKE ? ESCAPE '\\' ORDER BY title LIMIT 7",
         (_like(text),)),
        "Книгу", text, lambda r: f"«{r['title']}» ({r['author']})",
    )


def _find_reader(conn, text: str) -> sqlite3.Row:
    cols = "SELECT id, full_name, email FROM readers"
    return _find_one(
        conn,
        (cols + " WHERE fold(email) = fold(?) OR fold(full_name) = fold(?)",
         (text.strip(), text.strip())),
        (cols + " WHERE fold(full_name) LIKE ? ESCAPE '\\' OR fold(email) LIKE ? "
         "ESCAPE '\\' ORDER BY full_name LIMIT 7", (_like(text), _like(text))),
        "Читача", text, lambda r: f"{r['full_name']} <{r['email']}>",
    )


def _as_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------ ресурси

@mcp.resource("library://books")
def list_books() -> str:
    """Повний каталог книг (JSON): жанр, рік, примірники всього й вільних."""
    with closing(connect_ro()) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, title, author, genre, year, copies_total, copies_available "
            "FROM books ORDER BY title")]
    return _as_json(rows)


@mcp.resource("library://readers")
def list_readers() -> str:
    """Список читачів (JSON)."""
    with closing(connect_ro()) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT id, full_name, email FROM readers ORDER BY full_name")]
    return _as_json(rows)


@mcp.resource("library://loans/active")
def list_active_loans() -> str:
    """Поточні видачі: хто що тримає й до якої дати (JSON)."""
    with closing(connect_ro()) as conn:
        rows = [dict(r) for r in conn.execute(
            "SELECT l.id, b.title, r.full_name AS reader, l.loan_date, l.due_date "
            "FROM loans l JOIN books b ON b.id = l.book_id "
            "JOIN readers r ON r.id = l.reader_id "
            "WHERE l.return_date IS NULL ORDER BY l.due_date")]
    return _as_json(rows)


@mcp.resource("library://schema")
def db_schema() -> str:
    """Схема бази (DDL): щоб модель бачила, з чим працює."""
    with closing(connect_ro()) as conn:
        rows = conn.execute(
            "SELECT sql FROM sqlite_master WHERE sql IS NOT NULL "
            "AND name NOT LIKE 'sqlite_%'").fetchall()
    return "\n\n".join(r[0] for r in rows)


# --------------------------------------------------------------- інструменти

READ_ONLY = ToolAnnotations(
    read_only_hint=True, idempotent_hint=True, open_world_hint=False)


@mcp.tool(annotations=READ_ONLY)
def read_library_query(query: str) -> str:
    """Виконує SQL-запит НА ЧИТАННЯ й повертає рядки як JSON-РЯДОК.

    Формат: {"row_count": int, "rows": [{...}]}. Хай роботу робить SQL:
    GROUP BY, COUNT, SUM, JOIN доступні. З'єднання відкрите лише для читання
    й захищене авторизатором SQLite, тому запис, ATTACH і PRAGMA неможливі.
    Максимум 200 рядків, а на довгі запити діє бюджет кроків: додавайте
    WHERE або LIMIT. Для пошуку без урахування регістру (кирилиця) є
    функція fold(): fold(genre) = fold('фантастика'). Активна видача це
    рядок loans із return_date IS NULL. Схема: ресурс library://schema.
    Змінювати дані треба через issue_book, return_book і add_book.

    Args:
        query: Один SQL-оператор, який лише читає дані
    """
    try:
        with closing(connect_ro()) as conn:
            cur = conn.execute(query)
            rows = [dict(r) for r in cur.fetchmany(MAX_ROWS)]
            truncated = cur.fetchone() is not None
    except sqlite3.Warning as exc:
        # У старіших Python складений запит дає Warning, який НЕ підклас
        # Error: без окремої гілки він вилітав би необробленим винятком.
        return f"Error: {exc} Один оператор за виклик."
    except sqlite3.Error as exc:
        text = str(exc).lower()
        if "not authorized" in text:
            return ("Error: інструмент лише читає. Запис, ATTACH, PRAGMA і "
                    "load_extension заблоковані авторизатором SQLite.")
        if "interrupted" in text:
            return ("Error: запит вичерпав бюджет кроків. Додайте WHERE або "
                    "LIMIT. Саме так зупиняються рекурсивні запити.")
        if "one statement" in text:
            return "Error: один оператор за виклик."
        return f"SQL error: {exc}"

    out: dict[str, Any] = {"row_count": len(rows), "rows": rows}
    if truncated:
        out["note"] = f"Показано перші {MAX_ROWS} рядків."
    return _as_json(out)


@mcp.tool(annotations=READ_ONLY)
def search_books(genre: str = "", author: str = "", title: str = "",
                 only_available: bool = False) -> str:
    """Шукає книги за жанром, автором і/або назвою (частковий збіг, без
    урахування регістру). Порожні параметри ігноруються.

    Args:
        genre: Жанр або його частина, наприклад «фантастика»
        author: Автор або його частина
        title: Назва або її частина
        only_available: Лише ті книги, що зараз є на полиці
    """
    where, params = [], []
    for column, value in (("genre", genre), ("author", author), ("title", title)):
        if value.strip():
            where.append(f"fold({column}) LIKE ? ESCAPE '\\'")
            params.append(_like(value))
    if only_available:
        where.append("copies_available > 0")
    sql = ("SELECT id, title, author, genre, year, copies_available, copies_total "
           "FROM books" + (" WHERE " + " AND ".join(where) if where else "")
           + " ORDER BY genre, title")
    with closing(connect_ro()) as conn:
        rows = [dict(r) for r in conn.execute(sql, params).fetchmany(MAX_ROWS)]
    return _as_json({"row_count": len(rows), "books": rows})


@mcp.tool(annotations=ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=False,
    open_world_hint=False))
def issue_book(book: str, reader: str, days: int = LOAN_DAYS) -> str:
    """Видає книгу читачу: створює видачу й АТОМАРНО забирає один примірник.

    Args:
        book: Повна або часткова назва книги
        reader: Ім'я (повне або часткове) чи email читача
        days: На скільки днів видати, 1..60
    """
    if not 1 <= days <= 60:
        return "Error: строк видачі має бути від 1 до 60 днів."
    try:
        with transaction() as conn:
            b = _find_book(conn, book)
            r = _find_reader(conn, reader)
            # ОДИН атомарний UPDATE: умова наявності живе в самому запиті,
            # тож між перевіркою і списанням ніхто не влізе (TOCTOU).
            cur = conn.execute(
                "UPDATE books SET copies_available = copies_available - 1 "
                "WHERE id = ? AND copies_available > 0", (b["id"],))
            if cur.rowcount == 0:
                raise LibraryError(
                    f"усі {b['copies_total']} примірник(ів) «{b['title']}» "
                    "зараз видано.")
            loan_date = _today()
            due = loan_date + timedelta(days=days)
            try:
                conn.execute(
                    "INSERT INTO loans (book_id, reader_id, loan_date, due_date) "
                    "VALUES (?, ?, ?, ?)",
                    (b["id"], r["id"], loan_date.isoformat(), due.isoformat()))
            except sqlite3.IntegrityError as exc:
                # Виняток відкотить і списання примірника вище.
                if "loan limit" in str(exc):
                    raise LibraryError(
                        f"{r['full_name']} уже тримає {MAX_ACTIVE_LOANS} книги "
                        "(ліміт). Спершу треба повернути одну.") from exc
                if "UNIQUE" in str(exc):
                    raise LibraryError(
                        f"{r['full_name']} уже тримає «{b['title']}».") from exc
                raise LibraryError(f"база відхилила видачу: {exc}") from exc
            left = b["copies_available"] - 1
    except LibraryError as exc:
        return f"Error: {exc}"
    return (f"OK: «{b['title']}» видано {r['full_name']} до {due.isoformat()}. "
            f"Вільних примірників: {left}.")


@mcp.tool(annotations=ToolAnnotations(
    read_only_hint=False, destructive_hint=True, idempotent_hint=False,
    open_world_hint=False))
def return_book(book: str, reader: str) -> str:
    """Приймає книгу назад: закриває активну видачу й повертає примірник.

    Args:
        book: Повна або часткова назва книги
        reader: Ім'я (повне або часткове) чи email читача
    """
    try:
        with transaction() as conn:
            b = _find_book(conn, book)
            r = _find_reader(conn, reader)
            loan = conn.execute(
                "SELECT id, due_date FROM loans "
                "WHERE book_id = ? AND reader_id = ? AND return_date IS NULL",
                (b["id"], r["id"])).fetchone()
            # Закриваємо видачу умовним UPDATE: подвійне повернення не
            # зарахує примірник двічі.
            cur = conn.execute(
                "UPDATE loans SET return_date = ? "
                "WHERE book_id = ? AND reader_id = ? AND return_date IS NULL",
                (_today().isoformat(), b["id"], r["id"]))
            if cur.rowcount == 0 or loan is None:
                raise LibraryError(f"{r['full_name']} зараз не тримає «{b['title']}».")
            conn.execute(
                "UPDATE books SET copies_available = copies_available + 1 "
                "WHERE id = ?", (b["id"],))
    except LibraryError as exc:
        return f"Error: {exc}"
    except sqlite3.IntegrityError as exc:
        return f"Error: база відхилила повернення: {exc}"
    late = (_today() - date.fromisoformat(loan["due_date"])).days
    note = f" Прострочено на {late} дн." if late > 0 else ""
    return f"OK: {r['full_name']} повернув(ла) «{b['title']}».{note}"


@mcp.tool(annotations=ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False,
    open_world_hint=False))
def add_book(title: str, author: str, genre: str, year: int, copies: int = 1) -> str:
    """Додає нову книгу в каталог або збільшує кількість примірників наявної
    (збіг за назвою й автором без урахування регістру).

    Args:
        title: Назва книги
        author: Автор
        genre: Жанр (для нової книги)
        year: Рік видання (для нової книги)
        copies: Скільки примірників додати, >= 1
    """
    title, author, genre = title.strip(), author.strip(), genre.strip()
    if not (title and author and genre):
        return "Error: назва, автор і жанр не можуть бути порожніми."
    if copies < 1:
        return "Error: кількість примірників має бути не меншою за 1."
    try:
        with transaction() as conn:
            row = conn.execute(
                "SELECT id, title FROM books "
                "WHERE fold(title) = fold(?) AND fold(author) = fold(?)",
                (title, author)).fetchone()
            if row:
                conn.execute(
                    "UPDATE books SET copies_total = copies_total + ?, "
                    "copies_available = copies_available + ? WHERE id = ?",
                    (copies, copies, row["id"]))
                return f"OK: у «{row['title']}» додано {copies} примірник(ів)."
            conn.execute(
                "INSERT INTO books (title, author, genre, year, copies_total, "
                "copies_available) VALUES (?, ?, ?, ?, ?, ?)",
                (title, author, genre, year, copies, copies))
    except sqlite3.IntegrityError as exc:
        return f"Error: база відхилила запис: {exc}"
    return f"OK: книгу «{title}» додано, примірників: {copies}."


@mcp.tool(annotations=ToolAnnotations(
    read_only_hint=True, idempotent_hint=False, open_world_hint=False))
def get_system_metrics() -> dict[str, Any]:
    """Поточні системні метрики машини, де працює сервер: CPU, пам'ять і диск
    (диск, на якому лежить база). Обсяги в байтах."""
    mem = psutil.virtual_memory()
    disk = psutil.disk_usage(Path(DB_FILE).resolve().anchor or "/")
    return {
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "memory_total": mem.total,
        "memory_used": mem.used,
        "memory_available": mem.available,
        "memory_percent": mem.percent,
        "disk_total": disk.total,
        "disk_used": disk.used,
        "disk_free": disk.free,
        "disk_percent": disk.percent,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


# -------------------------------------------------------------------- промпт

@mcp.prompt()
def library_analyst(question: str) -> str:
    """Готує промпт для відповіді на питання про бібліотеку через SQL.

    Args:
        question: Питання, на яке треба відповісти
    """
    return f"""Ти аналітик бібліотеки й працюєш із цією базою SQLite.

Схема:
{db_schema()}

Правила предметної області:
- активна видача: рядок loans, де return_date IS NULL;
- вільні примірники це books.copies_available, а не copies_total;
- прострочена видача: активна, і due_date < date('now') (сьогоднішня дата);
- для пошуку без урахування регістру (кирилиця) використовуй fold().

Питання: {question}

Напиши ОДИН SQL-запит, виконай його через read_library_query і назви числа
й назви, які він повернув. Не оцінюй «на око» й не додавай даних, яких у
результаті немає. Якщо запит повернув помилку, виправ його й повтори.
"""


if __name__ == "__main__":
    # У stdio-режимі stdout належить JSON-RPC: друкувати можна лише в stderr,
    # інакше протокол ламається.
    if sys.stderr.isatty():
        print("Library Manager · stdio", file=sys.stderr)
    mcp.run()
