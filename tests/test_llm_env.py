"""Tests for environment-driven LLM integration."""

import os
from unittest.mock import patch

import pytest


def test_build_llm_client_returns_none_without_api_key():
    """build_llm_client returns None when API key is missing."""
    from affective_fly.llm_env import build_llm_client

    with patch.dict(os.environ, {}, clear=True):
        client = build_llm_client()
        assert client is None


def test_build_llm_client_accepts_emotional_memory_api_key():
    """build_llm_client reads EMOTIONAL_MEMORY_LLM_API_KEY."""
    from affective_fly.llm_env import build_llm_client

    with patch.dict(os.environ, {"EMOTIONAL_MEMORY_LLM_API_KEY": "test-key"}, clear=True):
        pytest.importorskip("openai")
        client = build_llm_client()
        assert client is not None


def test_build_llm_client_falls_back_to_openai_api_key():
    """build_llm_client falls back to OPENAI_API_KEY."""
    from affective_fly.llm_env import build_llm_client

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        pytest.importorskip("openai")
        client = build_llm_client()
        assert client is not None


def test_build_llm_client_returns_none_without_openai_package():
    """build_llm_client returns None if openai package is not installed."""
    from affective_fly.llm_env import build_llm_client

    with patch.dict(os.environ, {"OPENAI_API_KEY": "test-key"}, clear=True):
        with patch.dict("sys.modules", {"openai": None}):
            client = build_llm_client()
            assert client is None


def test_get_hf_token_returns_none_when_unset():
    """get_hf_token returns None when HF_TOKEN is not set."""
    from affective_fly.llm_env import get_hf_token

    with patch.dict(os.environ, {}, clear=True):
        token = get_hf_token()
        assert token is None


def test_get_hf_token_returns_value_when_set():
    """get_hf_token returns HF_TOKEN value."""
    from affective_fly.llm_env import get_hf_token

    with patch.dict(os.environ, {"HF_TOKEN": "test-hf-token"}, clear=True):
        token = get_hf_token()
        assert token == "test-hf-token"


def test_demo_llm_appraisal_exits_cleanly_without_key(capsys):
    """demo_llm_appraisal.py prints skip message and exits cleanly."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "examples/demo_llm_appraisal.py"],
        env={},
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0
    assert "skipping" in result.stdout.lower()


def test_demo_embedder_exits_cleanly_without_extra(capsys):
    """demo_embedder.py prints skip message and exits cleanly when embed extra missing."""
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "examples/demo_embedder.py"],
        env={},
        capture_output=True,
        text=True,
        cwd=os.getcwd(),
    )
    assert result.returncode == 0
    # May either skip due to import or succeed if embed is installed
    assert result.returncode == 0


@pytest.mark.integration
@pytest.mark.skipif(
    os.getenv("EMOTIONAL_MEMORY_LLM_API_KEY") is None and os.getenv("OPENAI_API_KEY") is None,
    reason="No LLM API key configured (integration test)",
)
def test_live_llm_appraisal_with_key():
    """Integration test: live LLM call when key is present (CI skips)."""
    from emotional_memory import EmotionalMemory, InMemoryStore

    from affective_fly import DualPathEncoder, FakeEmbedder, LIFCircuit, SensoryFrame
    from affective_fly.llm_env import build_llm_client

    llm_client = build_llm_client()
    assert llm_client is not None, "API key present but client build failed"

    encoder = DualPathEncoder.from_llm(llm_client, fallback_on_error=True)
    em = EmotionalMemory(store=InMemoryStore(), embedder=FakeEmbedder())

    frame = SensoryFrame.from_dict({
        "context": "journal",
        "note_id": "integration-test-001",
        "query": "test query",
        "sentiment": 0.5,
    })

    em.set_affect(LIFCircuit(n_kc=100, n_dan=10, n_mbon=20, seed=0).core_affect(frame.sensory))
    mem = em.encode(frame.event_text(), metadata=frame.context)

    appraisal = encoder.appraise(frame.event_text(), frame.context)
    updated = encoder.attach(em, mem, appraisal)

    assert updated.tag.appraisal is not None
    assert -1.0 <= appraisal.novelty <= 1.0
    assert -1.0 <= appraisal.goal_relevance <= 1.0
    assert 0.0 <= appraisal.coping_potential <= 1.0
