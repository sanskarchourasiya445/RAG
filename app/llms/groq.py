"""
app/llms/groq.py

Concrete BaseLLM implementation for xAI's Groq models.

Note: xAI does not ship a bespoke first-party Python SDK as ubiquitous as
Mistral's or Google's - their documented integration path is an
OpenAI-compatible REST API. We use the `openai` package pointed at xAI's
base_url, which is the officially documented approach.

Loads its own Groq_API_KEY directly from the environment - no central
settings object involved.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from openai import OpenAI

from app.config import GROQ_MODEL
from app.llms.base import BaseLLM, LLMGenerationError

load_dotenv()

logger = logging.getLogger(__name__)

XAI_BASE_URL = "https://api.groq.com/openai/v1"


class GroqLLM(BaseLLM):
    provider_name = "groq"

    def __init__(self, api_key: str | None = None, model: str = GROQ_MODEL) -> None:
        api_key = api_key or os.getenv("GROQ_API_KEY")
        if not api_key:
            raise LLMGenerationError(
                "GROQ_API_KEY is not set. Add it to your .env file to use "
                "the Groq provider, or switch LLM_PROVIDER in app/config.py."
            )
        self.model = model
        self._client = OpenAI(api_key=api_key, base_url=XAI_BASE_URL)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Groq generation failed: %s", exc)
            raise LLMGenerationError(f"Groq generation failed: {exc}") from exc

        if not response.choices:
            raise LLMGenerationError("Groq returned no choices in the response.")

        return response.choices[0].message.content