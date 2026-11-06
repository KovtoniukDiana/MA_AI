"""Автотести сервера: схема, захист SQL, видача/повернення, атомарність.

Кожен тест працює з власною копією бази у тимчасовій теці, а сервер
перемикається на неї через monkeypatch: справжня library.db не чіпається.
Запуск: pytest -q
"""
import asyncio
import json
import sqlite3
import threading
from datetime import date

import pytest

import server
import setup_db


@pytest.fixture()
def db(tmp_path, monkeypatch):
    path = tmp_path / "test.db"
    setup_db.setup_database(path)
    monkeypatch.setattr(server, "DB_FILE", path)
    monkeypatch.setattr(server, "_today", lambda: date(2026, 11, 1))
    return path


def raw(db_path):
    """Пряме з'єднання з увімкненими FK: для перевірки схеми в обхід інструментів."""
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def available(db_path, title):
    with raw(db_path) as c:
        return c.execute("SELECT copies_available FROM books WHERE title = ?",
                         (title,)).fetchone()[0]


# --------------------------------------------------------------- схема БД

def test_check_available_not_above_total(db):
    with pytest.raises(sqlite3.IntegrityError):
        raw(db).execute("UPDATE books SET copies_available = copies_total + 1 "
                        "WHERE id = 1")


def test_check_available_not_negative(db):
    with pytest.raises(sqlite3.IntegrityError):
        raw(db).execute("UPDATE books SET copies_available = -1 WHERE id = 1")


def test_unique_title_author(db):
    with pytest.raises(sqlite3.IntegrityError):
        raw(db).execute("INSERT INTO books (title, author, genre, year, "
                        "copies_total, copies_available) "
                        "VALUES ('Дюна', 'Френк Герберт', 'x', 2000, 1, 1)")


def test_unique_email_and_format(db):
    c = raw(db)
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO readers (full_name, email) VALUES "
                  "('X', 'olena.shevchenko@example.com')")
    with pytest.raises(sqlite3.IntegrityError):
        c.execute("INSERT INTO readers (full_name, email) VALUES ('Y', 'not-an-email')")


def test_foreign_key_on_loans(db):
    with pytest.raises(sqlite3.IntegrityError):
        raw(db).execute("INSERT INTO loans (book_id, reader_id, loan_date, due_date) "
                        "VALUES (999, 1, '2026-11-01', '2026-11-15')")


def test_due_date_check(db):
    with pytest.raises(sqlite3.IntegrityError):
        raw(db).execute("INSERT INTO loans (book_id, reader_id, loan_date, due_date) "
                        "VALUES (7, 5, '2026-11-10', '2026-11-01')")


def test_seed_counters_consistent(db):
    """copies_available = copies_total мінус активні видачі, для кожної книги."""
    bad = raw(db).execute(
        "SELECT COUNT(*) FROM books b WHERE b.copies_available <> b.copies_total - "
        "(SELECT COUNT(*) FROM loans l WHERE l.book_id = b.id "
        "AND l.return_date IS NULL)").fetchone()[0]
    assert bad == 0


# ----------------------------------------------- захист read_library_query

def test_select_works(db):
    out = json.loads(server.read_library_query(
        "SELECT COUNT(*) AS n FROM loans WHERE return_date IS NULL"))
    assert out["rows"] == [{"n": 5}]


def test_join_aggregation(db):
    out = json.loads(server.read_library_query(
        "SELECT b.genre, COUNT(*) AS n FROM loans l JOIN books b ON b.id = l.book_id "
        "GROUP BY b.genre ORDER BY n DESC, b.genre"))
    assert out["row_count"] >= 1 and "genre" in out["rows"][0]


@pytest.mark.parametrize("sql", [
    "DELETE FROM books",
    "UPDATE books SET copies_available = 99",
    "INSERT INTO readers (full_name, email) VALUES ('a', 'a@b.cc')",
    "DROP TABLE loans",
    "ATTACH DATABASE 'evil.db' AS evil",
    "PRAGMA writable_schema = ON",
    "CREATE TABLE t (x)",
])
def test_writes_and_dangerous_statements_rejected(db, sql):
    before = raw(db).execute("SELECT COUNT(*) FROM books").fetchone()[0]
    out = server.read_library_query(sql)
    assert out.startswith("Error") and "лише читає" in out
    assert raw(db).execute("SELECT COUNT(*) FROM books").fetchone()[0] == before


def test_mode_ro_blocks_writes_even_without_authorizer(db):
    """Перший шар незалежний від авторизатора: рушій не відкриває файл на запис."""
    conn = server.connect_ro()
    conn.set_authorizer(None)
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        conn.execute("DELETE FROM books")
    conn.close()


def test_attach_is_denied_by_authorizer_not_by_mode(db, tmp_path):
    """mode=ro не стосується ATTACH: цю дірку закриває саме авторизатор."""
    conn = server.connect_ro()
    with pytest.raises(sqlite3.DatabaseError, match="not authorized"):
        conn.execute(f"ATTACH DATABASE '{tmp_path / 'new.db'}' AS n")
    conn.close()
    assert not (tmp_path / "new.db").exists()


def test_load_extension_function_denied(db):
    assert "лише читає" in server.read_library_query("SELECT load_extension('x')")


def test_recursive_query_starting_with_select_is_stopped(db):
    """startswith('SELECT') його б пропустив: зупиняє бюджет кроків."""
    sql = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "
           "SELECT COUNT(*) FROM c")
    assert "бюджет кроків" in server.read_library_query(sql)


def test_cross_join_bomb_is_stopped(db):
    sql = ("SELECT COUNT(*) FROM books a, books b, books c, books d, books e, "
           "books f, books g, books h")
    assert "бюджет кроків" in server.read_library_query(sql)


def test_result_is_truncated_to_max_rows(db):
    sql = ("WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c "
           "WHERE x < 500) SELECT x FROM c")
    out = json.loads(server.read_library_query(sql))
    assert out["row_count"] == server.MAX_ROWS and "note" in out


def test_huge_value_is_limited(db):
    out = server.read_library_query("SELECT zeroblob(100000000)")
    assert out.startswith("SQL error")


def test_multiple_statements_rejected(db):
    assert "один оператор" in server.read_library_query("SELECT 1; SELECT 2").lower()


def test_syntax_error_is_reported_not_raised(db):
    assert server.read_library_query("SELEKT *").startswith("SQL error")


def test_connect_ro_works_with_cyrillic_and_spaces_in_path(tmp_path, monkeypatch):
    folder = tmp_path / "Лекція 2 #lb4"
    folder.mkdir()
    path = folder / "library.db"
    setup_db.setup_database(path)
    monkeypatch.setattr(server, "DB_FILE", path)
    assert json.loads(server.read_library_query("SELECT 1 AS x"))["rows"] == [{"x": 1}]


# ----------------------------------------------------------- search_books

def test_search_by_genre_is_case_insensitive_for_cyrillic(db):
    for variant in ("фантастика", "ФАНТАСТИКА", "Фантастика", "фант"):
        out = json.loads(server.search_books(genre=variant))
        assert {b["title"] for b in out["books"]} == {"Дюна", "Основа", "Соляріс"}


def test_search_only_available_and_combined_filters(db):
    out = json.loads(server.search_books(genre="фантастика", only_available=True))
    assert "Соляріс" not in {b["title"] for b in out["books"]}
    out = json.loads(server.search_books(genre="класика", author="коцюбинський"))
    assert out["row_count"] == 2


def test_search_wildcards_are_literal(db):
    assert json.loads(server.search_books(title="%"))["row_count"] == 0


def test_search_without_filters_returns_all(db):
    assert json.loads(server.search_books())["row_count"] == 12


# -------------------------------------------------------------- issue_book

def test_issue_book_ok(db):
    msg = server.issue_book("чистий код", "Шевченко")
    assert msg.startswith("OK") and "2026-11-15" in msg
    assert available(db, "Чистий код") == 3 - 1 - 1  # 3 всього, 1 вже видано


def test_issue_book_custom_days(db):
    assert "2026-11-08" in server.issue_book("Sapiens", "ткачук", days=7)


def test_issue_book_invalid_days(db):
    assert server.issue_book("Дюна", "Шевченко", days=0).startswith("Error")
    assert server.issue_book("Дюна", "Шевченко", days=61).startswith("Error")


def test_issue_no_copies_left(db):
    msg = server.issue_book("Соляріс", "Мельник")
    assert msg.startswith("Error") and "видано" in msg
    assert available(db, "Соляріс") == 0


def test_issue_unknown_book_and_reader(db):
    assert "не знайдено" in server.issue_book("Неіснуюча", "Шевченко")
    assert "не знайдено" in server.issue_book("Дюна", "Нікому Невідомий")


def test_issue_ambiguous_book_asks_to_clarify(db):
    msg = server.issue_book("Коцюбинський", "Шевченко")  # у назвах такого немає
    assert "не знайдено" in msg
    msg = server.issue_book("о", "Шевченко")  # підходить багатьом
    assert "Уточніть" in msg


def test_issue_same_book_twice_to_same_reader(db):
    assert server.issue_book("Дюна", "Шевченко").startswith("OK")
    msg = server.issue_book("Дюна", "Шевченко")
    assert msg.startswith("Error") and "уже тримає" in msg
    assert available(db, "Дюна") == 3  # другий виклик нічого не списав


def test_issue_limit_rolls_back_stock_change(db):
    """Ліміт спрацьовує вже ПІСЛЯ списання: відкат мусить повернути примірник."""
    for title in ("Дюна", "Основа", "Intermezzo"):  # Шевченко вже тримає «Місто»
        server.issue_book(title, "Шевченко")
    before = available(db, "Sapiens")
    msg = server.issue_book("Sapiens", "Шевченко")
    assert msg.startswith("Error") and "ліміт" in msg
    assert available(db, "Sapiens") == before


def test_ambiguous_prefers_exact_match(db):
    assert server.issue_book("Основа", "Шевченко").startswith("OK")


def test_atomic_stock_under_concurrency(db):
    """20 читачів хапають останній примірник: рівно одна видача, жодного мінуса."""
    with raw(db) as c:
        c.execute("INSERT INTO books (title, author, genre, year, copies_total, "
                  "copies_available) VALUES ('Рідкісна', 'Автор', 'Тест', 2000, 1, 1)")
        for i in range(20):
            c.execute("INSERT INTO readers (full_name, email) VALUES (?, ?)",
                      (f"Читач {i}", f"reader{i}@example.com"))
    results, gate = [], threading.Barrier(20)

    def worker(i):
        gate.wait()
        results.append(server.issue_book("Рідкісна", f"reader{i}@example.com"))

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert sum(r.startswith("OK") for r in results) == 1
    assert available(db, "Рідкісна") == 0
    with raw(db) as c:
        assert c.execute("SELECT COUNT(*) FROM loans WHERE book_id = "
                         "(SELECT id FROM books WHERE title = 'Рідкісна')"
                         ).fetchone()[0] == 1


# ------------------------------------------------------------- return_book

def test_return_book_ok_and_overdue_note(db):
    msg = server.return_book("Соляріс", "Коваленко")  # строк був 2026-10-29
    assert msg.startswith("OK") and "Прострочено на 3 дн." in msg
    assert available(db, "Соляріс") == 1


def test_return_not_overdue_has_no_note(db):
    assert "Прострочено" not in server.return_book("Місто", "Шевченко")


def test_double_return_does_not_add_copy_twice(db):
    server.return_book("Соляріс", "Коваленко")
    msg = server.return_book("Соляріс", "Коваленко")
    assert msg.startswith("Error") and "не тримає" in msg
    assert available(db, "Соляріс") == 1


def test_return_book_not_borrowed(db):
    assert server.return_book("Дюна", "Мельник").startswith("Error")


def test_issue_return_issue_cycle(db):
    server.return_book("Соляріс", "Коваленко")
    assert server.issue_book("Соляріс", "Мельник").startswith("OK")
    assert available(db, "Соляріс") == 0


# ---------------------------------------------------------------- add_book

def test_add_new_book(db):
    assert server.add_book("Нова книга", "Автор", "Тест", 2024, 2).startswith("OK")
    with raw(db) as c:
        assert c.execute("SELECT copies_total, copies_available FROM books "
                         "WHERE title = 'Нова книга'").fetchone() == (2, 2)


def test_add_existing_book_adds_copies_case_insensitive(db):
    assert "додано 2" in server.add_book("дюна", "ФРЕНК ГЕРБЕРТ", "x", 1965, 2)
    with raw(db) as c:
        assert c.execute("SELECT copies_total, copies_available FROM books "
                         "WHERE title = 'Дюна'").fetchone() == (6, 6)
        assert c.execute("SELECT COUNT(*) FROM books").fetchone()[0] == 12


def test_add_book_validation(db):
    assert server.add_book("", "A", "g", 2000).startswith("Error")
    assert server.add_book("T", "A", "g", 2000, copies=0).startswith("Error")
    assert server.add_book("T", "A", "g", 5, copies=1).startswith("Error")  # CHECK year


# ------------------------------------------------- ресурси, промпт, метрики

def test_resources_return_json(db):
    assert len(json.loads(server.list_books())) == 12
    assert len(json.loads(server.list_readers())) == 5
    assert len(json.loads(server.list_active_loans())) == 5


def test_schema_resource_contains_constraints(db):
    ddl = server.db_schema()
    for word in ("CHECK", "UNIQUE", "REFERENCES", "TRIGGER"):
        assert word in ddl
    assert "sqlite_sequence" not in ddl


def test_prompt_embeds_schema_and_question(db):
    text = server.library_analyst("Скільки книг видано?")
    assert "CREATE TABLE books" in text and "Скільки книг видано?" in text
    assert "read_library_query" in text


def test_system_metrics_shape(db):
    m = server.get_system_metrics()
    assert 0 <= m["cpu_percent"] <= 100 and m["memory_total"] > 0
    assert m["disk_free"] <= m["disk_total"] and m["timestamp"].endswith("+00:00")


# ---------------------------------------- реєстрація в MCP (через протокол)

def test_mcp_registration_and_annotations():
    tools = {t.name: t for t in asyncio.run(server.mcp.list_tools())}
    assert set(tools) == {"read_library_query", "search_books", "issue_book",
                          "return_book", "add_book", "get_system_metrics"}
    assert tools["read_library_query"].annotations.read_only_hint is True
    assert tools["issue_book"].annotations.destructive_hint is True
    assert tools["add_book"].annotations.destructive_hint is False
    uris = {str(r.uri) for r in asyncio.run(server.mcp.list_resources())}
    assert uris == {"library://books", "library://readers",
                    "library://loans/active", "library://schema"}
    prompts = asyncio.run(server.mcp.list_prompts())
    assert [p.name for p in prompts] == ["library_analyst"]
