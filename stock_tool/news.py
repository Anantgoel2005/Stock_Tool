import logging
from typing import List, Optional
from urllib.parse import quote_plus

import feedparser

from .config import GOOGLE_NEWS_RSS_URL, YAHOO_RSS_URL


def fetch_news_articles(
    symbol: str,
    query: Optional[str] = None,
    days_back: int = 1,
) -> List[str]:
    """
    Fetch news headline snippets using RSS sources only (Yahoo Finance, Google News).
    """
    query = (query or symbol).strip()

    articles = _fetch_from_yahoo(symbol)
    if articles:
        logging.info(
            "Yahoo Finance RSS provided %s articles for '%s'.", len(articles), symbol
        )
        return articles

    logging.info("Yahoo RSS empty; falling back to Google News for '%s'.", query)
    articles = _fetch_from_google_news(query)
    if articles:
        logging.info(
            "Google News RSS provided %s articles for query '%s'.",
            len(articles),
            query,
        )
    else:
        logging.warning(
            "No articles available from Yahoo or Google RSS sources for '%s'.", query
        )
    return articles


def _fetch_from_yahoo(symbol: str) -> List[str]:
    encoded_symbol = quote_plus(symbol)
    url = YAHOO_RSS_URL.format(symbol=encoded_symbol)
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            logging.warning("Yahoo RSS returned no entries for %s.", symbol)
            return []
        logging.info("Yahoo RSS returned %s entries for %s.", len(feed.entries), symbol)
        return [
            f"{entry.get('title', '')}. {entry.get('summary', '')}".strip()
            for entry in feed.entries
        ]
    except Exception as exc:
        logging.error("Error parsing Yahoo RSS for %s: %s", symbol, exc)
        return []


def _fetch_from_google_news(query: str) -> List[str]:
    encoded_query = quote_plus(query)
    url = GOOGLE_NEWS_RSS_URL.format(query=encoded_query)
    try:
        feed = feedparser.parse(url)
        if not feed.entries:
            logging.warning("Google News RSS returned no entries for '%s'.", query)
            return []
        logging.info(
            "Google News RSS returned %s entries for query '%s'.",
            len(feed.entries),
            query,
        )
        return [
            f"{entry.get('title', '')}. {entry.get('summary', '')}".strip()
            for entry in feed.entries
        ]
    except Exception as exc:
        logging.error("Error parsing Google News RSS for '%s': %s", query, exc)
        return []

