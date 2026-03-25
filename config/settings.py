"""
Central configuration via Pydantic BaseSettings.

Priority (highest → lowest):
  1. Actual environment variables
  2. .env file
  3. Defaults defined here

LOCAL_DEV=true  →  switches Ollama URL to localhost and Qdrant host to localhost
                   so you can run the Python code directly on the host without Docker.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ------------------------------------------------------------------ #
    # Developer flag
    # ------------------------------------------------------------------ #
    local_dev: bool = Field(
        default=False, description="True when running outside Docker"
    )

    # ------------------------------------------------------------------ #
    # Ollama
    # ------------------------------------------------------------------ #
    ollama_base_url: str = Field(
        default="http://host.docker.internal:11434",
        description="Ollama endpoint. Overridden to localhost when LOCAL_DEV=true.",
    )
    ollama_model: str = Field(default="qwen2.5:7b")
    ollama_temperature: float = Field(default=0.1)
    ollama_timeout: int = Field(
        default=120, description="Seconds before LLM request times out"
    )

    # ------------------------------------------------------------------ #
    # Qdrant
    # ------------------------------------------------------------------ #
    qdrant_host: str = Field(
        default="qdrant",
        description="Qdrant hostname. Overridden to localhost when LOCAL_DEV=true.",
    )
    qdrant_port: int = Field(default=6333)
    collection_name: str = Field(default="securematagent_docs")

    # ------------------------------------------------------------------ #
    # Embeddings
    # ------------------------------------------------------------------ #
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: str = Field(default="cpu", description="'cpu' or 'cuda'")

    # ------------------------------------------------------------------ #
    # Chunking / retrieval
    # ------------------------------------------------------------------ #
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    top_k: int = Field(default=5, description="Number of retrieved chunks per query")
    memory_window: int = Field(
        default=5, description="Conversational memory turns to keep"
    )

    # ------------------------------------------------------------------ #
    # FastAPI
    # ------------------------------------------------------------------ #
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_reload: bool = Field(default=False)

    # ------------------------------------------------------------------ #
    # Streamlit
    # ------------------------------------------------------------------ #
    streamlit_port: int = Field(default=8501)

    # ------------------------------------------------------------------ #
    # Data paths
    # ------------------------------------------------------------------ #
    data_dir: str = Field(default="data/documents")

    # ------------------------------------------------------------------ #
    # Post-init: auto-patch URLs for local dev
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _apply_local_dev_overrides(self) -> "Settings":
        if self.local_dev:
            if self.ollama_base_url == "http://host.docker.internal:11434":
                self.ollama_base_url = "http://localhost:11434"
            if self.qdrant_host == "qdrant":
                self.qdrant_host = "localhost"
        return self

    # ------------------------------------------------------------------ #
    # Convenience properties
    # ------------------------------------------------------------------ #
    @property
    def qdrant_url(self) -> str:
        return f"http://{self.qdrant_host}:{self.qdrant_port}"

    @property
    def ollama_api_tags_url(self) -> str:
        return f"{self.ollama_base_url}/api/tags"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
