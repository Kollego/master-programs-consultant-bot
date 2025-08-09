from __future__ import annotations

import argparse

from src.bot.handlers import build_app


def main() -> None:
    parser = argparse.ArgumentParser(description="ITMO Programs Bot")
    parser.add_argument("--no-llm", action="store_true", help="Disable local LLM generation")
    args = parser.parse_args()

    app = build_app(enable_local_llm=not args.no_llm)
    app.run_polling(allowed_updates=None)


if __name__ == "__main__":
    main() 