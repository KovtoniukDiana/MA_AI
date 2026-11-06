"""Створює базу library.db: схема, обмеження цілісності та початкові дані.

Запуск: python setup_db.py
"""
import os
import sqlite3
from pathlib import Path

# Шлях можна перевизначити змінною LIBRARY_DB (так користуються тести).
DB_FILE = Path(os.environ.get("LIBRARY_DB", Path(__file__).with_name("library.db")))

# Скільки книжок читач може тримати одночасно. Правило живе в схемі (тригер),
# а не лише в Python: інструменти викликає модель, і вона помиляється.
MAX_ACTIVE_LOANS = 3

# Усі обмеження задано В СХЕМІ. Правило в схемі діє завжди: для будь-якого
# інструмента, для скрипта, для ручного SQL.
SCHEMA = f"""
CREATE TABLE books (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    title            TEXT    NOT NULL,
    author           TEXT    NOT NULL,
    genre            TEXT    NOT NULL,
    year             INTEGER NOT NULL CHECK (year BETWEEN 1000 AND 2100),
    copies_total     INTEGER NOT NULL CHECK (copies_total >= 1),
    copies_available INTEGER NOT NULL CHECK (copies_available >= 0),
    UNIQUE (title, author),
    CHECK (copies_available <= copies_total)
);

CREATE TABLE readers (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT NOT NULL,
    email     TEXT NOT NULL UNIQUE CHECK (email LIKE '%_@_%._%')
);

CREATE TABLE loans (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    book_id     INTEGER NOT NULL REFERENCES books (id),
    reader_id   INTEGER NOT NULL REFERENCES readers (id),
    loan_date   TEXT    NOT NULL,
    due_date    TEXT    NOT NULL,
    return_date TEXT,
    CHECK (due_date >= loan_date),
    CHECK (return_date IS NULL OR return_date >= loan_date)
);

-- Одна й та сама книга не може бути двічі активно видана одному читачу.
CREATE UNIQUE INDEX uq_active_loan
    ON loans (book_id, reader_id) WHERE return_date IS NULL;

CREATE INDEX idx_books_genre ON books (genre);
CREATE INDEX idx_loans_reader ON loans (reader_id);

-- Ліміт активних видач на читача.
CREATE TRIGGER trg_loan_limit
BEFORE INSERT ON loans
WHEN (SELECT COUNT(*) FROM loans
      WHERE reader_id = NEW.reader_id AND return_date IS NULL) >= {MAX_ACTIVE_LOANS}
BEGIN
    SELECT RAISE(ABORT, 'loan limit reached');
END;
"""

# (title, author, genre, year, copies_total)
BOOKS = [
    ("Тіні забутих предків", "Михайло Коцюбинський", "Класика", 1911, 3),
    ("Місто", "Валер'ян Підмогильний", "Класика", 1928, 2),
    ("Intermezzo", "Михайло Коцюбинський", "Класика", 1909, 1),
    ("Дюна", "Френк Герберт", "Фантастика", 1965, 4),
    ("Основа", "Айзек Азімов", "Фантастика", 1951, 2),
    ("Соляріс", "Станіслав Лем", "Фантастика", 1961, 1),
    ("Чистий код", "Роберт Мартін", "Програмування", 2008, 3),
    ("Прагматичний програміст", "Ендрю Хант", "Програмування", 1999, 2),
    ("Дизайн програм, що вражають даними", "Мартін Клеппман", "Програмування", 2017, 2),
    ("Sapiens", "Юваль Ной Харарі", "Нон-фікшн", 2011, 3),
    ("Шлях Дракона", "Олена Тесленко", "Фентезі", 2019, 2),
    ("Гаррі Поттер і філософський камінь", "Джоан Роулінг", "Фентезі", 1997, 5),
]

READERS = [
    ("Олена Шевченко", "olena.shevchenko@example.com"),
    ("Андрій Мельник", "andriy.melnyk@example.com"),
    ("Марія Коваленко", "maria.kovalenko@example.com"),
    ("Ігор Бондаренко", "igor.bondarenko@example.com"),
    ("Софія Ткачук", "sofia.tkachuk@example.com"),
]

# (title, reader_email, loan_date, due_date, return_date | None)
LOANS = [
    ("Дюна", "olena.shevchenko@example.com", "2026-10-01", "2026-10-15", "2026-10-12"),
    ("Чистий код", "andriy.melnyk@example.com", "2026-10-10", "2026-10-24", None),
    ("Соляріс", "maria.kovalenko@example.com", "2026-10-15", "2026-10-29", None),
    ("Місто", "olena.shevchenko@example.com", "2026-10-20", "2026-11-03", None),
    ("Sapiens", "igor.bondarenko@example.com", "2026-10-22", "2026-11-05", None),
    ("Основа", "andriy.melnyk@example.com", "2026-10-25", "2026-11-08", "2026-11-01"),
    ("Гаррі Поттер і філософський камінь", "sofia.tkachuk@example.com",
     "2026-10-28", "2026-11-11", None),
]


def setup_database(db_file: Path = DB_FILE) -> None:
    """Перестворює базу з нуля й наповнює її початковими даними."""
    db_file = Path(db_file)
    db_file.unlink(missing_ok=True)
    conn = sqlite3.connect(db_file)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA)
    conn.executemany(
        "INSERT INTO books (title, author, genre, year, copies_total, "
        "copies_available) VALUES (?, ?, ?, ?, ?, ?)",
        [(t, a, g, y, n, n) for t, a, g, y, n in BOOKS],
    )
    conn.executemany("INSERT INTO readers (full_name, email) VALUES (?, ?)", READERS)

    book_ids = {t: i for i, t in conn.execute("SELECT id, title FROM books")}
    reader_ids = {e: i for i, e in conn.execute("SELECT id, email FROM readers")}
    for title, email, loan, due, ret in LOANS:
        conn.execute(
            "INSERT INTO loans (book_id, reader_id, loan_date, due_date, return_date) "
            "VALUES (?, ?, ?, ?, ?)",
            (book_ids[title], reader_ids[email], loan, due, ret),
        )
        # Активна видача займає один примірник. Лічильник узгоджуємо тут же,
        # щоб початкові дані не суперечили схемі.
        if ret is None:
            conn.execute(
                "UPDATE books SET copies_available = copies_available - 1 WHERE id = ?",
                (book_ids[title],),
            )
    conn.commit()
    conn.close()
    print(f"База {db_file.name} створена.")


if __name__ == "__main__":
    setup_database()
