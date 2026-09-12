"""
Environment-driven OpenAI-compatible LLM client builder.

Reads credentials from environment variables following emotional-memory patterns:
- EMOTIONAL_MEMORY_LLM_API_KEY (or OPENAI_API_KEY as fallback)
- EMOTIONAL_MEMORY_LLM_BASE_URL (default: https://api.openai.com/v1)
- EMOTIONAL_MEMORY_LLM_MODEL (e.g., gpt-5-mini, gpt-4o-mini)
- HF_TOKEN (optional, for HuggingFace model downloads)

Returns None if required keys are missing (skip-clean for demos/tests).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any


def build_llm_client() -> Callable[[str, dict[str, Any]], str] | None:
    """
    Build an OpenAI-compatible callable for emotional-memory LLMAppraisalEngine.

    Returns:
        A callable(prompt: str, schema: dict) -> str compatible with
        LLMAppraisalEngine, or None if API key is missing.

    Environment Variables:
        EMOTIONAL_MEMORY_LLM_API_KEY or OPENAI_API_KEY: Required API key
        EMOTIONAL_MEMORY_LLM_BASE_URL: Base URL (default: https://api.openai.com/v1)
        EMOTIONAL_MEMORY_LLM_MODEL: Model name (e.g., gpt-5-mini)
        HF_TOKEN: Optional HuggingFace token for model downloads
    """
    api_key = os.getenv("EMOTIONAL_MEMORY_LLM_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None

    base_url = os.getenv("EMOTIONAL_MEMORY_LLM_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("EMOTIONAL_MEMORY_LLM_MODEL", "gpt-4o-mini")

    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI(api_key=api_key, base_url=base_url)

    def llm_callable(prompt: str, schema: dict[str, Any]) -> str:
        """
        Call OpenAI-compatible API with structured output.

        Args:
            prompt: The prompt text
            schema: JSON schema for structured output

        Returns:
            JSON string matching the schema
        """
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "appraisal_response",
                    "strict": True,
                    "schema": schema,
                },
            },
        )
        content: str | None = response.choices[0].message.content
        if content is None:
            raise ValueError("Empty response from LLM")
        return content

    return llm_callable


def get_hf_token() -> str | None:
    """
    Get HuggingFace token from environment.

    Returns:
        HF_TOKEN value or None if not set
    """
    return os.getenv("HF_TOKEN")
