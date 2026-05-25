"""
LLM Service — Groq API (LLaMA-3) for disruption analysis.

Uses the free Groq cloud API with OpenAI-compatible interface
to analyze news articles for supply chain disruptions.
"""

import json
import re
from loguru import logger
from backend.config import settings


# System prompt for structured disruption extraction
_SYSTEM_PROMPT = """You are a supply chain disruption analyst. Analyze the given news article and extract:
1. disruption_type: one of [natural_disaster, geopolitical, labor, pandemic, operational, financial, logistics, supply, cyber_attack, unknown]
2. severity: one of [high, medium, low]
3. affected_companies: list of company names mentioned or likely affected
4. affected_countries: list of countries mentioned or likely affected
5. summary: a 1-2 sentence summary of the disruption

Return ONLY valid JSON in this exact format:
{
    "disruption_type": "...",
    "severity": "...",
    "affected_companies": ["..."],
    "affected_countries": ["..."],
    "summary": "..."
}"""


class LLMService:
    """Analyzes news text for supply chain disruptions using Groq (LLaMA-3)."""

    def __init__(self):
        if not settings.groq_api_key:
            raise RuntimeError(
                "GROQ_API_KEY is required. Get a free key at https://console.groq.com"
            )
        logger.info("🤖 LLM Service: using Groq ({}) — FREE", settings.groq_model)

    async def analyze_disruption(self, news_text: str) -> dict:
        """
        Analyze a news article for supply chain disruptions via Groq API.
        Returns structured JSON with disruption details.
        """
        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )

            response = await client.chat.completions.create(
                model=settings.groq_model,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": news_text},
                ],
                temperature=0.1,
                max_tokens=500,
            )

            content = response.choices[0].message.content.strip()
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                result = json.loads(json_match.group())
                logger.info("✅ Groq analysis complete: {}", result.get("disruption_type"))
                return result
            else:
                logger.error("Groq returned non-JSON response: {}", content[:200])
                raise RuntimeError("Groq returned non-JSON response")

        except json.JSONDecodeError as exc:
            logger.error("Failed to parse Groq JSON response: {}", exc)
            raise RuntimeError(f"Failed to parse LLM response: {exc}")
        except Exception as exc:
            logger.error("Groq API error: {}", exc)
            raise
