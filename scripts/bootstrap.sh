#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

# 1) Python venv
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt

# 2) Env file
if [[ ! -f .env ]]; then
  cp config.sample.env .env
fi

# 3) Load env
set -a
source .env || true
set +a

# 4) Scrape live site -> programs.json
python -m src.scraping.parse_itmo \
  --urls https://abit.itmo.ru/program/master/ai https://abit.itmo.ru/program/master/ai_product \
  --out data/processed/programs.json

# 5) Build corpus and FAISS index
python -m src.pipeline.build_corpus --in data/processed/programs.json --out data/processed/corpus.jsonl
python -m src.pipeline.index --in data/processed/corpus.jsonl --index_dir data/vector_store

# 6) Run bot if token available
if [[ -n "${TELEGRAM_BOT_TOKEN:-${TELEGRAM_TOKEN:-}}" ]]; then
  echo "Starting bot..."
  python -m src.bot.main
else
  echo "Done. Set TELEGRAM_BOT_TOKEN in .env and run: source .venv/bin/activate && python -m src.bot.main"
fi 