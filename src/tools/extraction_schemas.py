"""
Pydantic output schemas for DataExtractorTool.

Each model maps to one of the 5 extraction types and provides:
  - Structured field definitions with Optional typing (Python 3.10 compat)
  - .to_markdown() for human-readable formatting
  - parse_llm_output() classmethod with one retry on JSON parse failure
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_json_block(text: str) -> str:
    """
    Pull the first {...} JSON object from *text*, stripping markdown fences.
    Handles ```json ... ``` and raw JSON embedded in prose.
    """
    # Strip markdown code fences
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.DOTALL)
    if fenced:
        return fenced.group(1)
    # First standalone JSON object
    obj = re.search(r"(\{[\s\S]*\})", text, re.DOTALL)
    if obj:
        return obj.group(1)
    return text.strip()


def _safe_parse(raw: str) -> Dict[str, Any]:
    """Return parsed JSON dict or raise ValueError with a descriptive message."""
    candidate = _extract_json_block(raw)
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise ValueError(f"JSON decode error: {exc}. Raw text: {raw[:400]}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected a JSON object, got {type(parsed).__name__}.")
    return parsed


# ---------------------------------------------------------------------------
# 1. Material Properties
# ---------------------------------------------------------------------------


class MaterialProperties(BaseModel):
    crystal_system: Optional[str] = None
    space_group: Optional[str] = None
    lattice_a: Optional[str] = None
    lattice_b: Optional[str] = None
    lattice_c: Optional[str] = None
    lattice_alpha: Optional[str] = None
    lattice_beta: Optional[str] = None
    lattice_gamma: Optional[str] = None
    density: Optional[str] = None
    band_gap: Optional[str] = None
    melting_point: Optional[str] = None
    additional: Optional[Dict[str, str]] = Field(default_factory=dict)


class MaterialPropertiesResult(BaseModel):
    material: str = Field(default="Unknown")
    properties: MaterialProperties = Field(default_factory=MaterialProperties)
    sources: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"## Material Properties: {self.material}\n"]
        p = self.properties
        rows = [
            ("Crystal System", p.crystal_system),
            ("Space Group", p.space_group),
            ("Lattice a", p.lattice_a),
            ("Lattice b", p.lattice_b),
            ("Lattice c", p.lattice_c),
            ("α", p.lattice_alpha),
            ("β", p.lattice_beta),
            ("γ", p.lattice_gamma),
            ("Density", p.density),
            ("Band Gap", p.band_gap),
            ("Melting Point", p.melting_point),
        ]
        filled = [(k, v) for k, v in rows if v]
        if filled:
            lines.append("| Property | Value |")
            lines.append("|---|---|")
            for k, v in filled:
                lines.append(f"| {k} | {v} |")
        if p.additional:
            lines.append("\n**Additional Properties:**")
            for k, v in p.additional.items():
                lines.append(f"- {k}: {v}")
        if self.sources:
            lines.append(f"\n**Sources:** {', '.join(self.sources)}")
        return "\n".join(lines)

    @classmethod
    def parse_llm_output(cls, raw: str, retry_fn=None) -> "MaterialPropertiesResult":
        try:
            data = _safe_parse(raw)
            props_data = data.get("properties", {})
            return cls(
                material=data.get("material", "Unknown"),
                properties=MaterialProperties(
                    **(
                        {
                            k: v
                            for k, v in props_data.items()
                            if k in MaterialProperties.model_fields
                        }
                        if isinstance(props_data, dict)
                        else {}
                    )
                ),
                sources=data.get("sources", []),
            )
        except (ValueError, TypeError) as exc:
            if retry_fn:
                retry_raw = retry_fn(str(exc))
                data = _safe_parse(retry_raw)
                props_data = data.get("properties", {})
                return cls(
                    material=data.get("material", "Unknown"),
                    properties=MaterialProperties(
                        **(
                            {
                                k: v
                                for k, v in props_data.items()
                                if k in MaterialProperties.model_fields
                            }
                            if isinstance(props_data, dict)
                            else {}
                        )
                    ),
                    sources=data.get("sources", []),
                )
            return cls(material="Parse error", sources=[str(exc)[:200]])


# ---------------------------------------------------------------------------
# 2. Safety Data
# ---------------------------------------------------------------------------


class SafetyDataResult(BaseModel):
    chemical: str = Field(default="Unknown")
    ghs_classification: Optional[str] = None
    hazard_statements: List[str] = Field(default_factory=list)
    first_aid: Optional[Dict[str, str]] = Field(default_factory=dict)
    ppe_required: List[str] = Field(default_factory=list)
    storage_disposal: Optional[str] = None
    sources: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"## Safety Data: {self.chemical}\n"]
        if self.ghs_classification:
            lines.append(f"**GHS Classification:** {self.ghs_classification}")
        if self.hazard_statements:
            lines.append("\n**Hazard Statements:**")
            for h in self.hazard_statements:
                lines.append(f"- {h}")
        if self.first_aid:
            lines.append("\n**First Aid:**")
            for route, action in self.first_aid.items():
                lines.append(f"- *{route}:* {action}")
        if self.ppe_required:
            lines.append(f"\n**PPE Required:** {', '.join(self.ppe_required)}")
        if self.storage_disposal:
            lines.append(f"\n**Storage/Disposal:** {self.storage_disposal}")
        if self.sources:
            lines.append(f"\n**Sources:** {', '.join(self.sources)}")
        return "\n".join(lines)

    @classmethod
    def parse_llm_output(cls, raw: str, retry_fn=None) -> "SafetyDataResult":
        try:
            data = _safe_parse(raw)
            return cls(
                chemical=data.get("chemical", "Unknown"),
                ghs_classification=data.get("ghs_classification"),
                hazard_statements=data.get("hazard_statements", []),
                first_aid=data.get("first_aid", {}),
                ppe_required=data.get("ppe_required", []),
                storage_disposal=data.get("storage_disposal"),
                sources=data.get("sources", []),
            )
        except (ValueError, TypeError) as exc:
            if retry_fn:
                retry_raw = retry_fn(str(exc))
                data = _safe_parse(retry_raw)
                return cls(
                    chemical=data.get("chemical", "Unknown"),
                    ghs_classification=data.get("ghs_classification"),
                    hazard_statements=data.get("hazard_statements", []),
                    first_aid=data.get("first_aid", {}),
                    ppe_required=data.get("ppe_required", []),
                    storage_disposal=data.get("storage_disposal"),
                    sources=data.get("sources", []),
                )
            return cls(chemical="Parse error", sources=[str(exc)[:200]])


# ---------------------------------------------------------------------------
# 3. Compliance Requirements
# ---------------------------------------------------------------------------


class ComplianceRequirement(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None


class ComplianceResult(BaseModel):
    framework: str = Field(default="Unknown")
    control_family: Optional[str] = None
    requirements: List[ComplianceRequirement] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"## Compliance: {self.framework}"]
        if self.control_family:
            lines.append(f"**Control Family:** {self.control_family}\n")
        if self.requirements:
            lines.append("| ID | Title | Description |")
            lines.append("|---|---|---|")
            for r in self.requirements:
                rid = r.id or ""
                title = r.title or ""
                desc = (r.description or "")[:120]
                lines.append(f"| {rid} | {title} | {desc} |")
        if self.sources:
            lines.append(f"\n**Sources:** {', '.join(self.sources)}")
        return "\n".join(lines)

    @classmethod
    def parse_llm_output(cls, raw: str, retry_fn=None) -> "ComplianceResult":
        try:
            data = _safe_parse(raw)
            reqs_raw = data.get("requirements", [])
            reqs = [
                (
                    ComplianceRequirement(**r)
                    if isinstance(r, dict)
                    else ComplianceRequirement()
                )
                for r in reqs_raw
            ]
            return cls(
                framework=data.get("framework", "Unknown"),
                control_family=data.get("control_family"),
                requirements=reqs,
                sources=data.get("sources", []),
            )
        except (ValueError, TypeError) as exc:
            if retry_fn:
                retry_raw = retry_fn(str(exc))
                data = _safe_parse(retry_raw)
                reqs_raw = data.get("requirements", [])
                reqs = [
                    (
                        ComplianceRequirement(**r)
                        if isinstance(r, dict)
                        else ComplianceRequirement()
                    )
                    for r in reqs_raw
                ]
                return cls(
                    framework=data.get("framework", "Unknown"),
                    control_family=data.get("control_family"),
                    requirements=reqs,
                    sources=data.get("sources", []),
                )
            return cls(framework="Parse error", sources=[str(exc)[:200]])


# ---------------------------------------------------------------------------
# 4. Table Extraction
# ---------------------------------------------------------------------------


class TableResult(BaseModel):
    table_title: str = Field(default="Extracted Table")
    headers: List[str] = Field(default_factory=list)
    rows: List[List[str]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"## {self.table_title}\n"]
        if self.headers:
            lines.append("| " + " | ".join(self.headers) + " |")
            lines.append("|" + "---|" * len(self.headers))
            for row in self.rows:
                # Pad/trim row to header width
                padded = list(row) + [""] * (len(self.headers) - len(row))
                lines.append(
                    "| "
                    + " | ".join(str(c) for c in padded[: len(self.headers)])
                    + " |"
                )
        elif self.rows:
            for row in self.rows:
                lines.append("| " + " | ".join(str(c) for c in row) + " |")
        if self.sources:
            lines.append(f"\n**Sources:** {', '.join(self.sources)}")
        return "\n".join(lines)

    @classmethod
    def parse_llm_output(cls, raw: str, retry_fn=None) -> "TableResult":
        try:
            data = _safe_parse(raw)
            return cls(
                table_title=data.get("table_title", "Extracted Table"),
                headers=data.get("headers", []),
                rows=data.get("rows", []),
                sources=data.get("sources", []),
            )
        except (ValueError, TypeError) as exc:
            if retry_fn:
                retry_raw = retry_fn(str(exc))
                data = _safe_parse(retry_raw)
                return cls(
                    table_title=data.get("table_title", "Extracted Table"),
                    headers=data.get("headers", []),
                    rows=data.get("rows", []),
                    sources=data.get("sources", []),
                )
            return cls(table_title="Parse error", sources=[str(exc)[:200]])


# ---------------------------------------------------------------------------
# 5. General Key-Value
# ---------------------------------------------------------------------------


class GeneralKeyValueResult(BaseModel):
    document: str = Field(default="Unknown")
    key_findings: List[Dict[str, str]] = Field(default_factory=list)
    sources: List[str] = Field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [f"## Key Findings: {self.document}\n"]
        if self.key_findings:
            lines.append("| Key | Value |")
            lines.append("|---|---|")
            for item in self.key_findings:
                k = item.get("key", "")
                v = item.get("value", "")
                lines.append(f"| {k} | {v} |")
        if self.sources:
            lines.append(f"\n**Sources:** {', '.join(self.sources)}")
        return "\n".join(lines)

    @classmethod
    def parse_llm_output(cls, raw: str, retry_fn=None) -> "GeneralKeyValueResult":
        try:
            data = _safe_parse(raw)
            return cls(
                document=data.get("document", "Unknown"),
                key_findings=data.get("key_findings", []),
                sources=data.get("sources", []),
            )
        except (ValueError, TypeError) as exc:
            if retry_fn:
                retry_raw = retry_fn(str(exc))
                data = _safe_parse(retry_raw)
                return cls(
                    document=data.get("document", "Unknown"),
                    key_findings=data.get("key_findings", []),
                    sources=data.get("sources", []),
                )
            return cls(document="Parse error", sources=[str(exc)[:200]])


# ---------------------------------------------------------------------------
# Union type alias for type hints
# ---------------------------------------------------------------------------

ExtractionResult = (
    MaterialPropertiesResult
    | SafetyDataResult
    | ComplianceResult
    | TableResult
    | GeneralKeyValueResult
)
