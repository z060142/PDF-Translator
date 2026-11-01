"""PDF translator: layout-preserving PDF translation with LLM check-and-fix loop."""


def main() -> None:
    from .app import main as app_main
    app_main()
