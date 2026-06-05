"""Gemini research client for Servo-Skull Phase 0 source gathering.

Uses Gemini 2.5 Flash/Pro with Google Search grounding for web-anchored
research reports. The deep-research-preview models require the Interactions
API (not yet fully exposed in google-genai SDK v2.6.x); this implementation
uses the standard generateContent endpoint with search grounding which
produces equivalent quality for rules-engine seeding purposes.
"""
import logging
import time
from typing import Any

from servo_skull.models import DeepResearchReport

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None
    types = None

logger = logging.getLogger(__name__)

_MODEL_STANDARD = "gemini-2.5-flash"
_MODEL_MAX = "gemini-2.5-pro"
_MAX_WAIT_SECONDS = 300  # 5 minutes for standard generateContent


class DeepResearchClient:
    """Gemini research client using Google Search grounding for web-anchored reports.

    Requires GEMINI_API_KEY in environment or explicit api_key argument.
    """

    def __init__(self, api_key: str, use_max: bool = False, timeout: float = _MAX_WAIT_SECONDS):
        if not api_key:
            raise ValueError("api_key must not be empty")
        if genai is None:
            raise ImportError("google-genai package is required")
        self.api_key = api_key
        self.model = _MODEL_MAX if use_max else _MODEL_STANDARD
        self.timeout = timeout
        self.client = genai.Client(api_key=self.api_key)

    def research(self, query: str, domain: str = "general") -> DeepResearchReport:
        """Submit a research query with Google Search grounding and return the report.

        Args:
            query: Natural language research question
            domain: Logical domain tag for output metadata

        Returns:
            DeepResearchReport with report text and extracted sources
        """
        logger.info(f"Starting research: model={self.model} domain={domain}")
        start = time.time()

        config = types.GenerateContentConfig(
            tools=[types.Tool(google_search=types.GoogleSearch())],
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=query,
            config=config,
        )

        duration = time.time() - start
        report_text = response.text or ""
        sources = _extract_sources(response)

        logger.info(f"Research complete in {duration:.1f}s, {len(sources)} sources")

        return DeepResearchReport(
            query=query,
            model=self.model,
            domain=domain,
            report_text=report_text,
            sources=sources,
            duration_seconds=round(duration, 2),
        )


def _extract_sources(response: Any) -> list[dict[str, Any]]:
    """Pull grounding source metadata from a Gemini response if present."""
    sources: list[dict[str, Any]] = []
    try:
        candidates = getattr(response, "candidates", []) or []
        for candidate in candidates:
            grounding = getattr(candidate, "grounding_metadata", None)
            if grounding is None:
                continue
            chunks = getattr(grounding, "grounding_chunks", []) or []
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    sources.append({
                        "title": getattr(web, "title", ""),
                        "url": getattr(web, "uri", ""),
                    })
    except Exception as exc:
        logger.warning(f"Could not extract grounding sources: {exc}")
    return sources
