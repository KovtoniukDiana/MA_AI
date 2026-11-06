"""Наскрізна перевірка: запускає server.py як окремий процес (stdio) і говорить
із ним протоколом MCP так само, як Claude Desktop.

Працює на тимчасовій копії бази, тож library.db не змінюється.
Запуск: python verify_server.py
"""
import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

import setup_db

ROOT = Path(__file__).parent
EXPECTED_TOOLS = {"read_library_query", "search_books", "issue_book",
                  "return_book", "add_book", "get_system_metrics"}
EXPECTED_RESOURCES = {"library://books", "library://readers",
                      "library://loans/active", "library://schema"}
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    global failed
    failed += not ok
    print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f": {detail}" if detail else ""))


def text_of(result) -> str:
    return "".join(c.text for c in result.content if getattr(c, "text", None))


async def main(db_file: Path) -> None:
    params = StdioServerParameters(
        command=sys.executable, args=[str(ROOT / "server.py")],
        env={**os.environ, "LIBRARY_DB": str(db_file)})
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as s:
            await s.initialize()

            tools = {t.name: t for t in (await s.list_tools()).tools}
            check("tools/list", set(tools) == EXPECTED_TOOLS,
                  f"{len(tools)} інструментів")
            check("анотації read_only_hint",
                  tools["read_library_query"].annotations.read_only_hint is True
                  and tools["issue_book"].annotations.read_only_hint is False)

            uris = {str(r.uri) for r in (await s.list_resources()).resources}
            check("resources/list", uris == EXPECTED_RESOURCES, f"{len(uris)} ресурсів")
            prompts = (await s.list_prompts()).prompts
            check("prompts/list", [p.name for p in prompts] == ["library_analyst"])

            res = await s.read_resource("library://books")
            books = json.loads(res.contents[0].text)
            check("resources/read library://books", len(books) == 12,
                  f"{len(books)} книг")
            res = await s.read_resource("library://schema")
            check("resources/read library://schema",
                  "CREATE TABLE loans" in res.contents[0].text)

            p = await s.get_prompt("library_analyst",
                                   {"question": "Хто має прострочення?"})
            body = p.messages[0].content.text
            check("prompts/get library_analyst",
                  "Хто має прострочення?" in body and "CREATE TABLE books" in body)

            r = await s.call_tool("search_books", {"genre": "фантастика",
                                                   "only_available": True})
            found = json.loads(text_of(r))
            check("search_books (жанр, кирилиця)", found["row_count"] == 2,
                  ", ".join(b["title"] for b in found["books"]))

            sql = "SELECT COUNT(*) AS active FROM loans WHERE return_date IS NULL"
            r = await s.call_tool("read_library_query", {"query": sql})
            rows = json.loads(text_of(r))["rows"]
            check("read_library_query SELECT", rows == [{"active": 5}], str(rows))
            for label, sql in (("DELETE заблоковано", "DELETE FROM books"),
                               ("ATTACH заблоковано", "ATTACH DATABASE 'x.db' AS x")):
                r = await s.call_tool("read_library_query", {"query": sql})
                check(label, "лише читає" in text_of(r))
            r = await s.call_tool("read_library_query", {"query":
                "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM c) "
                "SELECT COUNT(*) FROM c"})
            check("рекурсивний SELECT зупинено бюджетом", "бюджет кроків" in text_of(r))

            r = await s.call_tool("issue_book", {"book": "Дюна", "reader": "Мельник"})
            check("issue_book", text_of(r).startswith("OK"), text_of(r))
            r = await s.call_tool("issue_book",
                                  {"book": "Соляріс", "reader": "Мельник"})
            check("issue_book без вільних примірників",
                  text_of(r).startswith("Error"), text_of(r))
            r = await s.call_tool("return_book",
                                  {"book": "Дюна", "reader": "Мельник"})
            check("return_book", text_of(r).startswith("OK"), text_of(r))
            r = await s.call_tool("add_book", {
                "title": "Нова книга", "author": "Автор", "genre": "Тест",
                "year": 2026, "copies": 2})
            check("add_book", text_of(r).startswith("OK"), text_of(r))

            r = await s.call_tool("get_system_metrics", {})
            check("get_system_metrics", "cpu_percent" in text_of(r))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "verify.db"
        setup_db.setup_database(db)
        asyncio.run(main(db))
    print("\nУсе гаразд." if not failed else f"\nПровалів: {failed}")
    sys.exit(1 if failed else 0)
