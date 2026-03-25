"""Tests for config/settings.py"""

import os

import pytest

from config.settings import Settings


def test_defaults():
    s = Settings()
    assert s.ollama_model == "qwen2.5:7b"
    assert s.qdrant_port == 6333
    assert s.chunk_size == 512
    assert s.chunk_overlap == 50
    assert s.top_k == 5


def test_local_dev_patches_urls():
    s = Settings(local_dev=True)
    assert "localhost" in s.ollama_base_url
    assert s.qdrant_host == "localhost"


def test_local_dev_false_keeps_docker_urls(monkeypatch):
    # Bypass the .env file and clear any env overrides to exercise code defaults
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    monkeypatch.delenv("QDRANT_HOST", raising=False)
    monkeypatch.delenv("LOCAL_DEV", raising=False)
    s = Settings(_env_file=None, local_dev=False)
    assert "host.docker.internal" in s.ollama_base_url
    assert s.qdrant_host == "qdrant"


def test_qdrant_url_property():
    s = Settings(local_dev=True)
    assert s.qdrant_url == "http://localhost:6333"


def test_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3")
    monkeypatch.setenv("TOP_K", "10")
    s = Settings()
    assert s.ollama_model == "llama3"
    assert s.top_k == 10
