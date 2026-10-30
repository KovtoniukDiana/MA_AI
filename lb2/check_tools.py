import sys

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass


def show(title, fn):
    print(f"\n=== {title} ===")
    try:
        result = fn()
        text = str(result).replace("\n", " ")
        print(f"OK, отримано {len(text)} символів. Початок:")
        print("  ", text[:300], "…")
    except Exception as exc:
        print(f"НЕ ПРАЦЮЄ: {type(exc).__name__}: {str(exc)[:200]}")


def wikipedia():
    # Wikimedia блокує дефолтний User-Agent, без цього рядка буде JSONDecodeError.
    import wikipedia as wp
    wp.set_user_agent("StudentLab2/1.0 (student@example.com)")
    from llama_index.tools.wikipedia import WikipediaToolSpec
    spec = WikipediaToolSpec()
    return spec.load_data(page="Alan Turing", lang="en")


def yahoo_finance():
    from llama_index.tools.yahoo_finance import YahooFinanceToolSpec
    spec = YahooFinanceToolSpec()
    return spec.stock_basic_info("AAPL")


def arxiv():
    from llama_index.tools.arxiv import ArxivToolSpec
    spec = ArxivToolSpec()
    return spec.to_tool_list()[0](query="Multimodal Large Language Models")


if __name__ == "__main__":
    show("WikipediaToolSpec: текст сторінки 'Alan Turing'", wikipedia)
    show("YahooFinanceToolSpec: основні дані акції AAPL", yahoo_finance)
    show("ArxivToolSpec: пошук 'Multimodal Large Language Models'", arxiv)
