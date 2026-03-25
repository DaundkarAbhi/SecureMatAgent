"""
Pydantic request/response schemas for the SecureMatAgent API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str = Field(
        ..., min_length=1, description="The user's question or instruction"
    )
    session_id: str = Field(
        default="default", description="Conversation session identifier"
    )


class IngestRequest(BaseModel):
    data_dir: str = Field(
        ...,
        description="Absolute or relative path to the directory containing source documents",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class TraceSummaryItem(BaseModel):
    step: int
    event_type: str
    name: str
    latency_ms: Optional[float] = None


class TraceEventModel(BaseModel):
    timestamp: str
    event_type: str
    name: str
    input: Optional[str] = None
    output: Optional[str] = None
    latency_ms: Optional[float] = None
    tokens_used: Optional[int] = None


class TraceLogResponse(BaseModel):
    session_id: str
    events: List[TraceEventModel]
    event_count: int


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The agent's final answer")
    tools_used: List[str] = Field(
        default_factory=list, description="Names of tools invoked during reasoning"
    )
    sources: List[str] = Field(
        default_factory=list,
        description="Source filenames cited from the knowledge base",
    )
    latency_ms: float = Field(
        ..., description="Wall-clock time for the full agent run (milliseconds)"
    )
    session_id: str = Field(..., description="Session identifier used for this request")
    trace_summary: Optional[List[TraceSummaryItem]] = Field(
        default=None, description="Condensed per-step trace (event type, name, latency)"
    )


class HealthStatus(BaseModel):
    status: str  # "ok" | "degraded" | "error"
    message: str


class HealthResponse(BaseModel):
    status: str = Field(..., description="Overall status: ok | degraded | error")
    ollama: HealthStatus
    qdrant: HealthStatus


class IngestResponse(BaseModel):
    status: str = Field(..., description="ok | error")
    total_files: int = Field(default=0)
    total_chunks: int = Field(default=0)
    collection_size: int = Field(default=0)
    domain_breakdown: Dict[str, int] = Field(default_factory=dict)
    message: str = Field(default="")


class SessionHistoryMessage(BaseModel):
    role: str  # "human" | "ai" | "tool"
    content: str
    name: Optional[str] = None  # tool name for tool messages


class SessionHistoryResponse(BaseModel):
    session_id: str
    messages: List[SessionHistoryMessage]
    message_count: int


class DeleteSessionResponse(BaseModel):
    session_id: str
    cleared: bool
    message: str
