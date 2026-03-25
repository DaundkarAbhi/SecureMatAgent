"""
SecureMatAgent — Streamlit Demo UI

Calls the agent directly (no API hop) for minimal latency in demo mode.
Run with:
    streamlit run src/frontend/app.py --server.port 8501
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any, Dict, List, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be the very first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SecureMatAgent",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so `src.*` imports work
# ---------------------------------------------------------------------------
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Custom CSS — clean, professional dark-accent theme
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* ── Global font ── */
    html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] * { color: #f1f5f9 !important; }
    [data-testid="stSidebar"] .stTextInput > div > div > input {
        background: #334155;
        border: 1px solid #475569;
        color: #f1f5f9;
    }
    [data-testid="stSidebar"] .stButton > button {
        background: #ef4444;
        color: white;
        border: none;
        border-radius: 6px;
        width: 100%;
    }
    [data-testid="stSidebar"] .stButton > button:hover { background: #dc2626; }

    /* ── Header banner ── */
    .agent-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #0f4c75 100%);
        border-radius: 12px;
        padding: 24px 32px;
        margin-bottom: 24px;
        border-left: 5px solid #3b82f6;
    }
    .agent-header h1 { color: #f8fafc; margin: 0; font-size: 2rem; font-weight: 700; }
    .agent-header p  { color: #94a3b8; margin: 6px 0 0; font-size: 0.95rem; }

    /* ── Example query buttons ── */
    .example-btn > button {
        background: #f8fafc;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        color: #334155;
        font-size: 0.82rem;
        padding: 6px 12px;
        text-align: left;
        width: 100%;
        margin-bottom: 4px;
        transition: background 0.15s;
    }
    .example-btn > button:hover {
        background: #e2e8f0;
        border-color: #3b82f6;
        color: #1e40af;
    }

    /* ── Chat messages ── */
    [data-testid="stChatMessage"] { border-radius: 10px; }

    /* ── Metric chips in sidebar ── */
    .metric-chip {
        background: #1e3a5f;
        border-radius: 20px;
        padding: 4px 12px;
        font-size: 0.75rem;
        color: #93c5fd;
        display: inline-block;
        margin: 2px;
    }

    /* ── Divider ── */
    hr { border-color: #e2e8f0; }

    /* ── Source tags ── */
    .source-tag {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        border-radius: 4px;
        color: #1d4ed8;
        display: inline-block;
        font-size: 0.78rem;
        margin: 2px 4px 2px 0;
        padding: 2px 8px;
    }

    /* ── Tool tags ── */
    .tool-tag {
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-radius: 4px;
        color: #15803d;
        display: inline-block;
        font-size: 0.78rem;
        margin: 2px 4px 2px 0;
        padding: 2px 8px;
    }

    /* ── Trace event pills ── */
    .trace-thought  { background:#eff6ff; border-left:3px solid #3b82f6; color:#1e40af; padding:4px 8px; border-radius:4px; margin:3px 0; font-size:0.82rem; }
    .trace-tool     { background:#f0fdf4; border-left:3px solid #22c55e; color:#15803d; padding:4px 8px; border-radius:4px; margin:3px 0; font-size:0.82rem; }
    .trace-obs      { background:#f8fafc; border-left:3px solid #94a3b8; color:#475569; padding:4px 8px; border-radius:4px; margin:3px 0; font-size:0.82rem; }
    .trace-error    { background:#fef2f2; border-left:3px solid #ef4444; color:#991b1b; padding:4px 8px; border-radius:4px; margin:3px 0; font-size:0.82rem; }
    .trace-finish   { background:#fefce8; border-left:3px solid #eab308; color:#713f12; padding:4px 8px; border-radius:4px; margin:3px 0; font-size:0.82rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict]: role, content, meta
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
if "total_queries" not in st.session_state:
    st.session_state.total_queries = 0
if "total_latency_ms" not in st.session_state:
    st.session_state.total_latency_ms = 0.0


# ---------------------------------------------------------------------------
# Helper: call the agent
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_ask():
    """Import ask() once; cached across reruns so the agent singleton persists."""
    # Set LOCAL_DEV so agent connects to localhost Ollama / Qdrant
    os.environ.setdefault("LOCAL_DEV", "true")
    from src.agent.run import ask  # noqa: PLC0415

    return ask


def run_agent(query: str, session_id: str) -> Dict[str, Any]:
    ask_fn = _load_ask()
    return ask_fn(query, session_id=session_id)


# ---------------------------------------------------------------------------
# Helper: render source / tool tags
# ---------------------------------------------------------------------------
def _render_tags(items: List[str], css_class: str) -> str:
    if not items:
        return "<em style='color:#94a3b8;font-size:0.82rem;'>None</em>"
    return "".join(f"<span class='{css_class}'>{item}</span>" for item in items)


# ---------------------------------------------------------------------------
# Helper: render trace expander
# ---------------------------------------------------------------------------
_TRACE_CSS: Dict[str, str] = {
    "agent_action": "trace-thought",
    "llm_start": "trace-thought",
    "llm_end": "trace-thought",
    "tool_start": "trace-tool",
    "tool_end": "trace-obs",
    "agent_finish": "trace-finish",
    "chain_error": "trace-error",
}

_TRACE_LABEL: Dict[str, str] = {
    "agent_action": "Thought",
    "llm_start": "LLM call",
    "llm_end": "LLM response",
    "tool_start": "Tool call",
    "tool_end": "Observation",
    "agent_finish": "Finish",
    "chain_error": "Error",
}


def _render_trace_expander(session_id: str, total_latency_ms: float) -> None:
    """
    Render an st.expander showing all trace events for the last agent run.

    Color-code: thoughts=blue, tool_calls=green, observations=gray, errors=red.
    """
    try:
        from src.observability.local_tracer import \
            get_trace_log  # noqa: PLC0415

        events = get_trace_log(session_id)
    except Exception:
        return

    if not events:
        return

    # Compute per-event latency breakdown for display
    total_tool_ms = sum(
        ev.latency_ms
        for ev in events
        if ev.latency_ms is not None and "tool" in ev.event_type
    )
    total_llm_ms = sum(
        ev.latency_ms
        for ev in events
        if ev.latency_ms is not None and "llm" in ev.event_type
    )

    with st.expander("🔬 Agent Trace", expanded=False):
        # Latency breakdown header
        col1, col2, col3 = st.columns(3)
        col1.metric("Total", f"{total_latency_ms/1000:.2f}s")
        col2.metric("LLM", f"{total_llm_ms/1000:.2f}s")
        col3.metric("Tools", f"{total_tool_ms/1000:.2f}s")

        st.markdown("---")

        for i, ev in enumerate(events):
            css = _TRACE_CSS.get(ev.event_type, "trace-obs")
            label = _TRACE_LABEL.get(ev.event_type, ev.event_type)
            lat_str = f" · {ev.latency_ms:.0f}ms" if ev.latency_ms is not None else ""
            tok_str = f" · {ev.tokens_used} tok" if ev.tokens_used is not None else ""

            content = ev.input or ev.output or ""
            content_preview = content[:200] + "…" if len(content) > 200 else content

            st.markdown(
                f"<div class='{css}'>"
                f"<strong>Step {i+1} · {label}</strong> "
                f"<code>{ev.name}</code>"
                f"<span style='color:#94a3b8;font-size:0.75rem;'>{lat_str}{tok_str}</span>"
                f"<br><span style='font-size:0.8rem;'>{content_preview}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🛡️ SecureMatAgent")
    st.markdown("---")

    session_id = st.text_input(
        "Session ID",
        value="default",
        help="Use the same session ID to share conversation history across queries.",
    )

    domain_filter = st.radio(
        "Domain focus",
        options=["All", "Materials Science", "Cybersecurity"],
        index=0,
        help="Prepend a domain hint to every query (UI only, agent handles both naturally).",
    )

    st.markdown("---")

    if st.button(
        "🗑️ Clear History", help="Wipe the in-memory conversation for this session"
    ):
        # Clear LangGraph memory
        try:
            from src.agent.memory import clear_session_memory  # noqa: PLC0415

            clear_session_memory(session_id)
        except Exception:
            pass
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.session_state.total_latency_ms = 0.0
        st.success("History cleared.")

    st.markdown("---")

    # Session stats
    st.markdown("**Session stats**")
    col_a, col_b = st.columns(2)
    col_a.metric("Queries", st.session_state.total_queries)
    avg_lat = (
        st.session_state.total_latency_ms / st.session_state.total_queries
        if st.session_state.total_queries > 0
        else 0.0
    )
    col_b.metric("Avg latency", f"{avg_lat/1000:.1f}s" if avg_lat else "—")

    st.markdown("---")

    with st.expander("ℹ️ About SecureMatAgent"):
        st.markdown("""
**SecureMatAgent** is a local, privacy-first RAG agent that combines:

- 📄 **Document search** — NIST standards, XRD protocols, SDS sheets, arXiv papers
- 🔬 **Materials calculator** — Bragg's law, d-spacing, density, unit conversion
- 🔍 **Anomaly detection** — statistical + domain-range checks on lattice parameters
- 🌐 **Web search** — DuckDuckGo for live CVE & research lookups
- 📊 **Data extractor** — structured material property extraction

**Stack:** LangGraph · Ollama (qwen2.5:7b) · Qdrant · sentence-transformers

All computation runs **100% locally** — no data leaves your machine.
            """)

    with st.expander("⚙️ Tools available"):
        tools = [
            ("🔎", "document_search", "Semantic search over the knowledge base"),
            (
                "🧮",
                "materials_calculator",
                "Bragg angles, d-spacing, density, conversions",
            ),
            ("🌐", "web_search", "Live DuckDuckGo search (CVEs, papers)"),
            ("⚠️", "data_anomaly_checker", "Detect statistical & domain anomalies"),
            ("📋", "data_extractor", "Extract structured material properties"),
        ]
        for icon, name, desc in tools:
            st.markdown(f"**{icon} `{name}`**  \n{desc}")


# ---------------------------------------------------------------------------
# Main area header
# ---------------------------------------------------------------------------
st.markdown(
    """
    <div class="agent-header">
      <h1>🛡️ SecureMatAgent</h1>
      <p>
        A local RAG agent for <strong>materials science</strong> research and
        <strong>laboratory cybersecurity</strong> — powered by Ollama + Qdrant.
        All data stays on your machine.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Example query buttons
# ---------------------------------------------------------------------------
EXAMPLE_QUERIES: List[str] = [
    "What are the NIST guidelines for protecting research data?",
    "Calculate the d-spacing for BaTiO3 (100) reflection, a=4.01 A",
    "Check if these lattice parameters look anomalous: a=3.9, b=3.9, c=45.2 for cubic perovskite",
    "Extract material properties for LiCoO2",
    "What recent CVEs affect laboratory information management systems?",
]

st.markdown("**Try an example query:**")
cols = st.columns(len(EXAMPLE_QUERIES))
for col, query in zip(cols, EXAMPLE_QUERIES):
    with col:
        # Truncate button label for display
        label = query[:42] + "…" if len(query) > 42 else query
        if st.button(label, key=f"ex_{hash(query)}", help=query):
            st.session_state.pending_query = query

st.markdown("---")

# ---------------------------------------------------------------------------
# Chat history display
# ---------------------------------------------------------------------------
chat_container = st.container()

with chat_container:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                tools_used: List[str] = meta.get("tools_used", [])
                sources: List[str] = meta.get("sources", [])
                latency: float = meta.get("latency_ms", 0.0)

                # Inline latency badge
                st.markdown(
                    f"<span style='color:#94a3b8;font-size:0.78rem;'>⏱ {latency/1000:.2f}s</span>",
                    unsafe_allow_html=True,
                )

                if tools_used or sources:
                    with st.expander("🔍 Agent Reasoning & Sources"):
                        st.markdown("**Tools used:**")
                        st.markdown(
                            _render_tags(tools_used, "tool-tag"),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Sources:**")
                        st.markdown(
                            _render_tags(sources, "source-tag"),
                            unsafe_allow_html=True,
                        )

                _render_trace_expander(session_id, latency)

# ---------------------------------------------------------------------------
# Chat input
# ---------------------------------------------------------------------------
user_input: Optional[str] = st.chat_input(
    "Ask about materials science or lab cybersecurity…",
    key="chat_input",
)

# Resolve query — typed input takes priority over button click
active_query: Optional[str] = user_input or st.session_state.get("pending_query")
if user_input:
    st.session_state.pending_query = None  # clear button-click on typed input

if active_query:
    st.session_state.pending_query = None  # always clear

    # Apply domain hint if a specific filter is chosen
    domain_prefix = ""
    if domain_filter == "Materials Science":
        domain_prefix = "[Focus: materials science] "
    elif domain_filter == "Cybersecurity":
        domain_prefix = "[Focus: cybersecurity] "

    full_query = domain_prefix + active_query

    # Append user message
    st.session_state.messages.append({"role": "user", "content": active_query})
    with st.chat_message("user"):
        st.markdown(active_query)

    # Run agent
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking…"):
            try:
                result = run_agent(full_query, session_id=session_id)
                answer: str = result.get("answer", "")
                tools_used_: List[str] = result.get("tools_used", [])
                sources_: List[str] = result.get("sources", [])
                latency_ms_: float = result.get("latency_ms", 0.0)

                st.markdown(answer)

                # Latency badge
                st.markdown(
                    f"<span style='color:#94a3b8;font-size:0.78rem;'>⏱ {latency_ms_/1000:.2f}s</span>",
                    unsafe_allow_html=True,
                )

                if tools_used_ or sources_:
                    with st.expander("🔍 Agent Reasoning & Sources"):
                        st.markdown("**Tools used:**")
                        st.markdown(
                            _render_tags(tools_used_, "tool-tag"),
                            unsafe_allow_html=True,
                        )
                        st.markdown("**Sources:**")
                        st.markdown(
                            _render_tags(sources_, "source-tag"),
                            unsafe_allow_html=True,
                        )

                _render_trace_expander(session_id, latency_ms_)

                # Persist to session state
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "meta": {
                            "tools_used": tools_used_,
                            "sources": sources_,
                            "latency_ms": latency_ms_,
                        },
                    }
                )
                st.session_state.total_queries += 1
                st.session_state.total_latency_ms += latency_ms_

            except Exception as exc:
                error_msg = f"**Error:** {exc}"
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )

    # Rerun to refresh sidebar stats without duplicating the message
    st.rerun()
