from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

from telegram import Update, ReplyKeyboardMarkup, ReplyKeyboardRemove, KeyboardButton, BotCommand
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes, filters,
    ConversationHandler, CallbackContext,
)
from dotenv import load_dotenv

from src.utils.logging_config import setup_logging
from src.utils.storage import read_json
from src.bot.rag import Retriever, is_relevant, build_answer
from src.bot.recommender import recommend_electives, score_program, Elective
from src.bot.generator import TinyGenerator

logger = logging.getLogger(__name__)

SELECT_PROGRAM, ASK_BACKGROUND = range(2)


@dataclass
class AppState:
    retriever: Retriever
    programs: Dict[str, Dict]


def load_programs(path: str) -> Dict[str, Dict]:
    arr = read_json(path)
    by_title = {p["title"]: p for p in arr}
    return by_title


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    titles = list(context.bot_data["programs"].keys())
    msg = (
        "Здравствуйте! Я помогу выбрать между магистратурами и спланировать учёбу.\n"
        "Выберите программу кнопкой ниже."
    )
    keyboard = [[KeyboardButton(t)] for t in titles] if titles else []
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True, one_time_keyboard=True)
    await update.message.reply_text(msg, reply_markup=reply_markup)
    return SELECT_PROGRAM


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "/start — выбрать программу кнопками\n"
        "/programs — краткая информация\n"
        "/plan — учебный план\n"
        "/background — заполнить бэкграунд (1–2 предложения)\n"
        "/recommend — рекомендации по курсам под ваш бэкграунд\n"
        "/compare — сравнить программы под ваш бэкграунд\n\n"
        "После выбора программы и бэкграунда можно задавать вопросы в свободной форме."
    )


async def list_programs(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    items = []
    for title, prog in context.bot_data["programs"].items():
        items.append(f"- {title} | {prog.get('url')}")
    await update.message.reply_text("\n".join(items))


async def select_program(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    choice = (update.message.text or "").strip()
    progs = context.bot_data["programs"]
    if choice not in progs:
        await update.message.reply_text("Не нашёл такую программу. Нажмите кнопку с названием программы.")
        return SELECT_PROGRAM
    context.user_data["program"] = choice
    await update.message.reply_text(
        f"Вы выбрали: {choice}. Доступно: /plan, /background, /recommend, /compare. Можете также задать вопрос.",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END


async def plan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    prog_name = context.user_data.get("program")
    if not prog_name:
        await update.message.reply_text("Сначала выберите программу через /start.")
        return
    prog = context.bot_data["programs"][prog_name]
    lines = [f"Учебный план — {prog_name}"]
    for c in prog.get("curriculum", [])[:30]:
        nm = c.get("name")
        tp = c.get("type")
        if not nm:
            continue
        if tp:
            lines.append(f"• {nm} ({tp})")
        else:
            lines.append(f"• {nm}")
    if len(prog.get("curriculum", [])) > 30:
        lines.append("… (сокращено)")
    lines.append(f"Источник: {prog.get('url')}")
    await update.message.reply_text("\n".join(lines))


async def background_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Кратко опишите ваш бэкграунд, интересы и цели (1–2 предложения).")
    context.user_data["awaiting_background"] = True


async def save_background(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    if context.user_data.get("awaiting_background"):
        txt = (update.message.text or "").strip()
        if not txt:
            await update.message.reply_text("Пустой бэкграунд не сохранён. Опишите в 1–2 предложениях.")
            return True
        context.user_data["background"] = txt
        context.user_data["awaiting_background"] = False
        await update.message.reply_text("Бэкграунд сохранён. Теперь можно использовать /recommend или /compare.")
        return True
    return False


async def recommend_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    prog_name = context.user_data.get("program")
    if not prog_name:
        await update.message.reply_text("Сначала выберите программу через /start.")
        return ConversationHandler.END
    bg = (context.user_data.get("background") or "").strip()
    if not bg:
        await update.message.reply_text("Опишите ваш бэкграунд, интересы и карьерные цели (1–2 предложения).")
        return ASK_BACKGROUND
    # If background already present, produce recommendations immediately
    prog = context.bot_data["programs"][prog_name]
    recs = recommend_electives(bg, prog, top_k=7)
    if not recs:
        await update.message.reply_text("Не удалось подобрать подходящие дисциплины. Уточните ваш бэкграунд через /background.")
        return ConversationHandler.END
    show = [f"• {r.name}" for r in recs[:5]]
    more = max(0, len(recs) - 5)
    intro = f"Рекомендации для «{prog_name}» по вашему профилю:"\
        if show else f"Для программы «{prog_name}» нашлись подходящие дисциплины."
    tail = f"\nЕщё {more} — в учебном плане." if more > 0 else ""
    url = prog.get("url")
    cta = f"\nУчебный план: {url}" if url else ""
    await update.message.reply_text("\n".join([intro, *show]) + tail + cta)
    return ConversationHandler.END


async def receive_background(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    background = (update.message.text or "").strip()
    context.user_data["background"] = background
    prog_name = context.user_data.get("program")
    prog = context.bot_data["programs"][prog_name]
    recs = recommend_electives(background, prog, top_k=7)
    if not recs:
        await update.message.reply_text("Не удалось подобрать выборные дисциплины. Уточните ваш бэкграунд.")
    else:
        show = [f"• {r.name}" for r in recs[:5]]
        more = max(0, len(recs) - 5)
        intro = f"С учётом вашего профиля я бы предложил в программе «{prog_name}»:" if show else f"Для программы «{prog_name}» нашлись подходящие элективы."
        if more > 0:
            intro += f"\nЕщё {more} подходящих варианта(ов) есть в учебном плане."
        url = prog.get("url")
        cta = f"\nПолный список дисциплин — {url}." if url else ""
        await update.message.reply_text("\n".join([intro, *show]) + cta)
    return ConversationHandler.END


async def compare_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    programs: Dict[str, Dict] = context.bot_data["programs"]
    bg = (context.user_data.get("background") or "").strip()
    if not bg:
        await update.message.reply_text("Сначала заполните бэкграунд через /background (1–2 предложения).")
        return
    # Score each program
    scored: List[Tuple[str, float, List[Elective]]] = []
    for title, prog in programs.items():
        agg, recs = score_program(bg, prog, top_k=5)
        scored.append((title, agg, recs))
    scored.sort(key=lambda t: t[1], reverse=True)
    best = scored[:2]
    lines: List[str] = []
    for title, agg, recs in best:
        bullets = "\n".join([f"    • {e.name}" for e in recs[:3]]) if recs else "    —"
        lines.append(f"{title} — совпадение {round(agg*100):d}%\n{bullets}")
    await update.message.reply_text(
        "Сравнение программ по вашему профилю:\n" + "\n\n".join(lines) + "\n\nИспользуйте /start, чтобы выбрать и изучить одну программу подробнее."
    )


_COST_WORDS = ("сколько стоит", "стоимост", "цена", "сколько в год", "платн", "руб", "₽")

_PREF_WORDS = (
    "лучшие предметы", "самые лучшие", "самые интересные", "что выбрать", "какие предметы лучше",
)


async def message_router(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()

    # Intercept background save inline if awaiting
    if await save_background(update, context):
        return

    prog_name = context.user_data.get("program")
    if not prog_name:
        await update.message.reply_text("Сначала выберите программу через /start.")
        return
    retr: Retriever = context.bot_data["retriever"]
    pairs = retr.search(text, k=8)
    def _prog_score(p):
        d, s = p
        mt = getattr(d, "meta", {}) or {}
        return 1 if mt.get("program_title") == prog_name else 0
    pairs.sort(key=lambda p: (_prog_score(p), p[1]), reverse=True)

    if any(w in text.lower() for w in _COST_WORDS):
        try:
            fact_doc = next(
                (d for d in retr.docs if (d.meta or {}).get("program_title") == prog_name and (d.meta or {}).get("section") == "facts"),
                None,
            )
            if fact_doc is not None and all(fact_doc is not d for d, _ in pairs):
                pairs = [(fact_doc, 1.0)] + pairs
        except Exception:
            pass

    pairs = pairs[:5]
    if not is_relevant(text, pairs):
        await update.message.reply_text("Я отвечаю только на вопросы по магистерским программам ИТМО (ИИ и AI Product). Уточните вопрос.")
        return

    # For vague "best subjects" style queries, prefer deterministic heuristic
    if any(kw in text.lower() for kw in _PREF_WORDS):
        answer = build_answer(text, pairs)
        await update.message.reply_text(answer)
        return

    # Try local LLM if configured, with typing indicator
    generator: Optional[TinyGenerator] = context.bot_data.get("generator")
    if generator is not None:
        try:
            await update.message.reply_text("Генерирую ответ…")
            contexts = [d.text for d, _ in pairs if d and d.text][:5]
            llm_answer = generator.generate(text, contexts)
            if llm_answer and llm_answer.strip():
                await update.message.reply_text(llm_answer)
                return
        except Exception as e:
            logger.warning("Local LLM generation failed: %s; falling back to heuristic", e)

    answer = build_answer(text, pairs)
    await update.message.reply_text(answer)


async def _post_init(app: Application) -> None:
    try:
        await app.bot.set_my_commands([
            BotCommand("start", "Начать и выбрать программу"),
            BotCommand("programs", "Краткая информация о программах"),
            BotCommand("plan", "Показать учебный план выбранной программы"),
            BotCommand("background", "Сохранить ваш бэкграунд (1–2 предложения)"),
            BotCommand("recommend", "Рекомендации курсов под ваш бэкграунд"),
            BotCommand("compare", "Сравнить программы под ваш бэкграунд"),
            BotCommand("help", "Подсказки по использованию"),
        ])
    except Exception as e:
        logger.warning("Failed to set bot commands: %s", e)


def build_app(enable_local_llm: bool = False) -> Application:
    setup_logging()
    load_dotenv()
    token = os.environ.get("TELEGRAM_BOT_TOKEN") or os.environ.get("TELEGRAM_TOKEN")
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

    retriever = Retriever(index_dir="data/vector_store")
    programs = load_programs("data/processed/programs.json")

    # Local LLM is enabled only if requested by caller
    generator: Optional[TinyGenerator] = None
    if enable_local_llm:
        try:
            generator = TinyGenerator(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            logger.info("Local LLM enabled: %s", generator.model_name)
        except Exception as e:
            logger.warning("Failed to initialize local LLM: %s", e)

    app = Application.builder().token(token).build()

    app.bot_data["retriever"] = retriever
    app.bot_data["programs"] = programs
    app.bot_data["generator"] = generator

    conv = ConversationHandler(
        entry_points=[
            CommandHandler("start", start),
            CommandHandler("recommend", recommend_cmd),
        ],
        states={
            SELECT_PROGRAM: [MessageHandler(filters.TEXT & ~filters.COMMAND, select_program)],
            ASK_BACKGROUND: [MessageHandler(filters.TEXT & ~filters.COMMAND, receive_background)],
        },
        fallbacks=[CommandHandler("start", start)],
    )

    app.add_handler(conv)
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("programs", list_programs))
    app.add_handler(CommandHandler("plan", plan_cmd))
    app.add_handler(CommandHandler("background", background_cmd))
    app.add_handler(CommandHandler("compare", compare_cmd))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, message_router))

    # Set slash-command hints in clients
    app.post_init = _post_init

    return app 