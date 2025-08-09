from __future__ import annotations

from src.bot.handlers import build_app


def main() -> None:
    app = build_app()
    app.run_polling(allowed_updates=None)


if __name__ == "__main__":
    main() 