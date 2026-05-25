"""
News Service — Fetches live news articles from NewsAPI.org.

Provides search and curated supply-chain headline fetching
for the disruption monitoring pipeline.
"""

import httpx
from loguru import logger
from backend.config import settings


# NewsAPI base URL
_BASE_URL = "https://newsapi.org/v2"

# Pre-built query for supply chain disruption headlines
_SUPPLY_CHAIN_QUERY = (
    "(supply chain OR semiconductor OR manufacturing) AND "
    "(disruption OR shortage OR earthquake OR sanctions OR strike OR flood OR shutdown OR pandemic)"
)


class NewsService:
    """Fetches news articles from NewsAPI.org."""

    def __init__(self):
        self.api_key = settings.news_api_key
        logger.info("📰 News Service initialized (NewsAPI.org)")

    async def search_news(self, query: str, page_size: int = 10, sort_by: str = "publishedAt") -> dict:
        """
        Search for news articles matching a query.

        Parameters
        ----------
        query : str
            Keywords to search for.
        page_size : int
            Number of results (max 100, default 10).
        sort_by : str
            Sort order: 'publishedAt', 'relevancy', or 'popularity'.

        Returns
        -------
        dict with keys: status, totalResults, articles[]
        """
        params = {
            "q": query,
            "pageSize": min(page_size, 100),
            "sortBy": sort_by,
            "language": "en",
            "apiKey": self.api_key,
        }

        logger.info("📰 Searching news for: '{}' (pageSize={})", query, page_size)

        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.get(f"{_BASE_URL}/everything", params=params)
            response.raise_for_status()
            data = response.json()

        if data.get("status") != "ok":
            error_msg = data.get("message", "Unknown NewsAPI error")
            logger.error("NewsAPI error: {}", error_msg)
            raise RuntimeError(f"NewsAPI error: {error_msg}")

        articles = self._normalize_articles(data.get("articles", []))
        logger.info("📰 Found {} articles", len(articles))

        return {
            "status": "ok",
            "totalResults": data.get("totalResults", 0),
            "articles": articles,
        }

    async def get_supply_chain_headlines(self, page_size: int = 10) -> dict:
        """
        Fetch curated supply chain disruption news.

        Uses a pre-built query combining supply chain keywords
        with common disruption triggers.
        """
        return await self.search_news(
            query=_SUPPLY_CHAIN_QUERY,
            page_size=page_size,
            sort_by="publishedAt",
        )

    def _normalize_articles(self, raw_articles: list) -> list:
        """
        Normalize NewsAPI article objects into a clean format.

        Combines title + description + content for the full
        analysis text that gets sent to Groq.
        """
        articles = []
        for article in raw_articles:
            # Skip removed articles
            if article.get("title") == "[Removed]":
                continue

            title = article.get("title", "").strip()
            description = article.get("description", "").strip()
            content = article.get("content", "").strip()

            # Build the full text for analysis (title + description + content)
            parts = [p for p in [title, description, content] if p]
            full_text = ". ".join(parts)

            articles.append({
                "title": title,
                "description": description,
                "content": content,
                "full_text": full_text,
                "source": article.get("source", {}).get("name", "Unknown"),
                "author": article.get("author", "Unknown"),
                "url": article.get("url", ""),
                "image_url": article.get("urlToImage", ""),
                "published_at": article.get("publishedAt", ""),
            })

        return articles