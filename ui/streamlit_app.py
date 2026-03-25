"""
SecureMatAgent — Streamlit chat UI.

Connects to the FastAPI backend at API_BASE_URL (default: http://localhost:8000).
Supports multi-turn conversation with memory-clear and ingest-trigger buttons.
"""

from __future__ import annotations

import os
import time
from typing import Optional

import httpx
import streamlit as st

# ------------------------------------------------------------------ #
# Config
# ------------------------------------------------------------------ #
API_BASE = os.getenv("API_BASE_URL", "http://localhost:8000/api/v1")
APP_TITLE = "SecureMatAgent"
APP_ICON = "🛡️"

# ------------------------------------------------------------------ #
# Page setup
# ------------------------------------------------------------------ #
st.set_page_config(
    page_title=APP_TITLE,
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# Session state
# ------------------------------------------------------------------ #
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []
if "session_id" not in st.session_state:
    st.session_state.session_id: Optional[str] = None


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #


def _api_ready() -> dict:
    try:
        r = httpx.get(f"{API_BASE}/ready", timeout=5)
        return r.json()
    except Exception as exc:
        return {"status": "error", "ollama": str(exc), "qdrant": str(exc)}


def _chat(message: str) -> str:
    payload = {"message": message, "session_id": st.session_state.session_id}
    try:
        r = httpx.post(f"{API_BASE}/chat", json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["answer"]
    except httpx.HTTPStatusError as exc:
        return f"API error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return f"Connection error: {exc}"


def _clear_memory() -> str:
    try:
        r = httpx.delete(f"{API_BASE}/memory", timeout=10)
        r.raise_for_status()
        return r.json().get("message", "Memory cleared.")
    except Exception as exc:
        return f"Error: {exc}"


def _trigger_ingest(data_dir: str = "") -> str:
    payload = {"data_dir": data_dir or None}
    try:
        r = httpx.post(f"{API_BASE}/ingest", json=payload, timeout=300)
        r.raise_for_status()
        d = r.json()
        return f"Ingested {d['chunks_upserted']} chunks. {d['message']}"
    except httpx.HTTPStatusError as exc:
        return f"Ingest error {exc.response.status_code}: {exc.response.text}"
    except Exception as exc:
        return f"Connection error: {exc}"


# ------------------------------------------------------------------ #
# Sidebar
# ------------------------------------------------------------------ #
with st.sidebar:
    st.title(f"{APP_ICON} {APP_TITLE}")
    st.caption("Cybersecurity-Aware Research Intelligence")
    st.divider()

    # Service status
    st.subheader("Service Status")
    if st.button("Refresh Status", use_container_width=True):
        st.session_state["ready_info"] = _api_ready()

    ready_info = st.session_state.get("ready_info", None)
    if ready_info is None:
        ready_info = _api_ready()
        st.session_state["ready_info"] = ready_info

    status_color = "green" if ready_info.get("status") == "ok" else "red"
    st.markdown(f"**API:** :{status_color}[{ready_info.get('status', 'unknown')}]")
    st.markdown(f"**Ollama:** {ready_info.get('ollama', 'N/A')}")
    st.markdown(f"**Qdrant:** {ready_info.get('qdrant', 'N/A')}")
    st.divider()

    # Memory controls
    st.subheader("Memory")
    if st.button(
        "Clear Conversation Memory", use_container_width=True, type="secondary"
    ):
        msg = _clear_memory()
        st.session_state.messages = []
        st.success(msg)
    st.divider()

    # Ingestion
    st.subheader("Ingest Documents")
    ingest_dir = st.text_input("Data directory (leave blank for default):", value="")
    if st.button("Run Ingest", use_container_width=True, type="primary"):
        with st.spinner("Ingesting documents…"):
            result = _trigger_ingest(ingest_dir)
        st.info(result)
    st.divider()

    st.caption(f"API endpoint: `{API_BASE}`")


# ------------------------------------------------------------------ #
# Main chat area
# ------------------------------------------------------------------ #
st.title(f"{APP_ICON} {APP_TITLE}")
st.caption(
    "Ask questions about cybersecurity, CVEs, or your ingested research documents."
)

# Render history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Input
if prompt := st.chat_input("Ask SecureMatAgent a question…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            t0 = time.time()
            answer = _chat(prompt)
            elapsed = time.time() - t0
        st.markdown(answer)
        st.caption(f"Response time: {elapsed:.1f}s")

    st.session_state.messages.append({"role": "assistant", "content": answer})
