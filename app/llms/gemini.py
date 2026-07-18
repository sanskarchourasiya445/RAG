"""
app/llms/gemini.py

Concrete BaseLLM implementation for Google Gemini, using the
google-generativeai SDK.

Loads its own GEMINI_API_KEY directly from the environment - no central
settings object involved.
"""

from __future__ import annotations

import logging
import os

import google.generativeai as genai
from dotenv import load_dotenv

from app.config import GEMINI_MODEL
from app.llms.base import BaseLLM, LLMGenerationError

load_dotenv()

logger = logging.getLogger(__name__)


class GeminiLLM(BaseLLM):
    provider_name = "gemini"

    def __init__(self, api_key: str | None = None, model: str = GEMINI_MODEL) -> None:
        api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise LLMGenerationError(
                "GEMINI_API_KEY is not set. Add it to your .env file to use "
                "the Gemini provider, or switch LLM_PROVIDER in app/config.py."
            )
        genai.configure(api_key=api_key)
        self.model_name = model
        self._model = genai.GenerativeModel(model_name=model)

    def generate(self, prompt: str) -> str:
        try:
            response = self._model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=0.2),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Gemini generation failed: %s", exc)
            raise LLMGenerationError(f"Gemini generation failed: {exc}") from exc

        # Gemini can return an empty candidate set if the prompt tripped a
        # safety filter. response.text raises in that case - we check
        # explicitly so we can give a clearer error than a raw SDK traceback.
        if not response.candidates:
            raise LLMGenerationError(
                "Gemini returned no candidates (possibly blocked by safety filters)."
            )

        return response.text
    