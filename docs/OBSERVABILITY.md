# SecureMatAgent — Observability & Tracing

SecureMatAgent ships with a **local-first tracing system** that requires no external
accounts or API keys.  LangSmith remote tracing is an optional upgrade.

---

## Quick start (local tracing — default)

No configuration needed.  Local tracing is active whenever `LANGCHAIN_API_KEY` is
**not** set.

Every agent invocation automatically:

1. Attaches a `LocalTracer` callback to the LangGraph run.
2. Captures every LLM call, tool call, agent thought, and error.
3. Stores events in-memory, keyed by `session_id`.
4. Optionally writes events to `traces/<session_id>.jsonl` (set `TRACE_TO_FILE=true`).

### View traces

**Streamlit UI** — click the "🔬 Agent Trace" expander below any assistant response.
It shows each step, its type, name, and latency.

**REST API** — `GET /api/traces/{session_id}` returns the full event log as JSON.

```bash
curl http://localhost:8000/api/traces/default | python -m json.tool
```

**Chat endpoint** — `POST /api/chat` now includes a `trace_summary` field in the
response body (condensed step list with event type, name, and latency).

---

## Trace event types

| Event type     | Meaning                                   | Colour in UI |
|----------------|-------------------------------------------|--------------|
| `llm_start`    | LLM call initiated with prompt            | Blue         |
| `llm_end`      | LLM returned a response (with token count)| Blue         |
| `tool_start`   | Tool invoked with input                   | Green        |
| `tool_end`     | Tool returned observation                 | Gray         |
| `agent_action` | Agent chose a tool (thought/reasoning)    | Blue         |
| `agent_finish` | Agent produced its final answer           | Yellow       |
| `chain_error`  | An exception occurred in the chain        | Red          |

### TraceEvent fields

| Field         | Type            | Description                              |
|---------------|-----------------|------------------------------------------|
| `timestamp`   | ISO-8601 string | UTC time the event was recorded          |
| `event_type`  | string          | One of the types above                   |
| `name`        | string          | Model name, tool name, or "agent"        |
| `input`       | string or null  | Prompt / tool input preview (≤500 chars) |
| `output`      | string or null  | Response / observation preview           |
| `latency_ms`  | float or null   | Wall-clock ms for this step              |
| `tokens_used` | int or null     | Token count (LLM only, if reported)      |

---

## File-based trace persistence

Set `TRACE_TO_FILE=true` in `.env` to append events as newline-delimited JSON
to `traces/<session_id>.jsonl`:

```env
TRACE_TO_FILE=true
```

Each line is one `TraceEvent` serialised with `json.dumps`.  The file grows
across sessions — rotate or archive as needed.

### Reading trace files

```python
import json

with open("traces/default.jsonl") as f:
    events = [json.loads(line) for line in f]

for ev in events:
    print(ev["event_type"], ev["name"], ev.get("latency_ms"))
```

---

## Enable LangSmith (optional)

[LangSmith](https://smith.langchain.com) provides a hosted trace UI with search,
filtering, and feedback collection.

### 1 — Create a LangSmith account

Sign up at <https://smith.langchain.com> and create a project called
`securematagent` (or any name).

### 2 — Get your API key

Settings → API Keys → Create API Key.

### 3 — Configure the project

Add to your `.env`:

```env
LANGCHAIN_API_KEY=ls__your_key_here
LANGCHAIN_PROJECT=securematagent
# optional — custom endpoint for self-hosted LangSmith:
# LANGCHAIN_ENDPOINT=https://your-langsmith.example.com
```

### 4 — Restart the application

```bash
# API server
uvicorn src.api.main:app --reload

# Streamlit UI
streamlit run src/frontend/app.py
```

`setup_tracing()` runs at agent creation time.  It detects `LANGCHAIN_API_KEY`
and sets `LANGCHAIN_TRACING_V2=true` automatically.  All subsequent LLM and
tool calls are forwarded to LangSmith in addition to the local store.

> **Note:** LangSmith and local tracing coexist.  The `/api/traces/{session_id}`
> endpoint and the Streamlit trace expander always show the local data, even when
> LangSmith is enabled.

---

## Structured logging

`src/observability/logger.py` provides a JSON-formatted logger:

```python
from src.observability.logger import get_logger

logger = get_logger(__name__)
logger.info("Query received", extra={"session_id": "abc", "query_len": 42})
```

Output (stdout, one line per record):

```json
{"ts": "2026-03-18T10:22:01", "level": "INFO", "logger": "src.api.main",
 "msg": "Query received", "session_id": "abc", "query_len": 42}
```

Log levels used across the codebase:

| Level     | Usage                                                   |
|-----------|---------------------------------------------------------|
| `DEBUG`   | Per-step trace events (tool I/O, LLM prompts)           |
| `INFO`    | Request start/end, session stats, tracing mode          |
| `WARNING` | Degraded services, slow responses, retries              |
| `ERROR`   | Agent failures, chain errors, ingestion failures        |

---

## Architecture overview

```
ask(query, session_id)          ← src/agent/run.py
  │
  ├── LocalTracer(session_id)   ← new per-call, attached via config["callbacks"]
  │
  └── agent.invoke(config={..., "callbacks": [tracer]})
        │
        ├── on_llm_start / on_llm_end     → TraceEvent stored in _trace_store[session_id]
        ├── on_tool_start / on_tool_end   → TraceEvent stored
        ├── on_agent_action / on_finish   → TraceEvent stored
        └── on_chain_error                → TraceEvent stored + logger.error()

GET /api/traces/{session_id}    → reads _trace_store[session_id]
POST /api/chat                  → includes trace_summary in response body
Streamlit expander              → calls get_trace_log(session_id) directly
```
