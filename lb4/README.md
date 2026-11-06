# Library Manager: MCP-сервер (ЛР №4)

Бібліотека (books, readers, loans) на SQLite: 4 ресурси, 6 інструментів, 1 промпт.

```
python -m venv venv
venv\Scripts\activate            # Mac/Linux: source venv/bin/activate
pip install -r requirements.txt
python setup_db.py               # створює library.db (скидає дані до початкових)
python verify_server.py          # наскрізна перевірка через stdio
pytest -q                        # 53 тести
npx @modelcontextprotocol/inspector@latest python server.py
```

Claude Desktop (File → Settings → Developer → Edit Config), шляхи абсолютні,
python береться з venv:

```json
{
  "mcpServers": {
    "Library Manager": {
      "command": "C:\\Шлях\\До\\lb4\\venv\\Scripts\\python.exe",
      "args": ["C:\\Шлях\\До\\lb4\\server.py"]
    }
  }
}
```

Після змін у конфігу Claude Desktop треба перезапустити.
Перед скріншотами виконайте `python setup_db.py`, щоб дані збігалися зі звітом.
