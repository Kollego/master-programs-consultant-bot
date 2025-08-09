from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse
from urllib import robotparser

import requests

logger = logging.getLogger(__name__)


@dataclass
class FetchConfig:
    timeout_sec: int = 20
    max_retries: int = 3
    min_delay_sec: float = 0.8
    max_delay_sec: float = 1.8
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )


class RobotsCache:
    def __init__(self) -> None:
        self._cache: dict[str, robotparser.RobotFileParser] = {}

    def is_allowed(self, url: str, user_agent: str) -> bool:
        parsed = urlparse(url)
        base = f"{parsed.scheme}://{parsed.netloc}"
        robots_url = base + "/robots.txt"
        if robots_url not in self._cache:
            rp = robotparser.RobotFileParser()
            try:
                rp.set_url(robots_url)
                rp.read()
            except Exception:
                # If robots can't be read, be conservative
                logger.warning("Could not read robots.txt from %s", robots_url)
                rp = robotparser.RobotFileParser()
                rp.parse("")
            self._cache[robots_url] = rp
        return self._cache[robots_url].can_fetch(user_agent, url)


robots_cache = RobotsCache()


def fetch_html(url: str, cfg: Optional[FetchConfig] = None) -> str:
    cfg = cfg or FetchConfig()
    if not robots_cache.is_allowed(url, cfg.user_agent):
        raise PermissionError(f"robots.txt disallows fetching: {url}")

    headers = {"User-Agent": cfg.user_agent, "Accept-Language": "ru,en;q=0.8"}

    for attempt in range(1, cfg.max_retries + 1):
        delay = random.uniform(cfg.min_delay_sec, cfg.max_delay_sec)
        time.sleep(delay)
        try:
            resp = requests.get(url, headers=headers, timeout=cfg.timeout_sec)
            if resp.status_code >= 500:
                raise requests.RequestException(f"Server error {resp.status_code}")
            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding or "utf-8"
            return resp.text
        except Exception as e:
            logger.warning("Fetch attempt %d failed for %s: %s", attempt, url, e)
            if attempt == cfg.max_retries:
                raise
    raise RuntimeError(f"Failed to fetch {url}") 