from __future__ import annotations

import logging
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class TinyGenerator:
    def __init__(self, model_name: str = "roneneldan/TinyStories-8M") -> None:
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.device = (
            "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
            ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model.to(self.device)
        self.model.eval()
        logger.info("TinyGenerator loaded: %s on %s", model_name, self.device)

    def generate(self, question: str, contexts: List[str], max_new_tokens: int = 120) -> str:
        # Keep prompt simple and compact for a tiny model
        ctx = "\n\n".join(f"- {c.strip()}" for c in contexts if c and c.strip())
        prompt = (
            "Задача: Ответь на русском кратко (1-2 предложения). Используй только информацию из Контекста. "
            "Если в контексте нет ответа — напиши: 'В материалах нет точного ответа.'\n\n"
            f"Контекст:\n{ctx}\n\nВопрос: {question}\nОтвет:"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Extract after 'Ответ:' if present
        ans = text.split("Ответ:")[-1].strip()
        # Post-trim to 2 sentences max
        parts = ans.split(".")
        if len(parts) > 2:
            ans = ".".join(parts[:2]).strip()
            if not ans.endswith("."):
                ans += "."
        return ans or "В материалах нет точного ответа." 