"""
app/llms/mistral.py

Concrete BaseLLM implementation for Mistral AI.

This file loads its OWN environment variable (MISTRAL_API_KEY) directly
via python-dotenv + os.getenv, right here where it's used. There is no
central settings object - each provider file is responsible for its own
credentials, which keeps things simple: to see everything Mistral needs,
you only ever have to look in this one file.
"""

from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from mistralai import Mistral

from app.config import MISTRAL_MODEL
from app.llms.base import BaseLLM, LLMGenerationError

load_dotenv()

logger = logging.getLogger(__name__)


class MistralLLM(BaseLLM):
    provider_name = "mistral"

    def __init__(self, api_key: str | None = None, model: str = MISTRAL_MODEL) -> None:
        api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise LLMGenerationError(
                "MISTRAL_API_KEY is not set. Add it to your .env file to use "
                "the Mistral provider, or switch LLM_PROVIDER in app/config.py."
            )
        self.model = model
        self._client = Mistral(api_key=api_key)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.chat.complete(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # low temperature: we want grounded,
                # deterministic-ish answers for a RAG system, not creative
                # variety.
            )
        except Exception as exc:  # noqa: BLE001 - normalize every SDK-level
            # failure (network timeout, auth error, rate limit) into our
            # own domain exception so the pipeline only ever needs to
            # catch LLMGenerationError, regardless of provider.
            logger.error("Mistral generation failed: %s", exc)
            raise LLMGenerationError(f"Mistral generation failed: {exc}") from exc

        if not response.choices:
            raise LLMGenerationError("Mistral returned no choices in the response.")

        return response.choices[0].message.content