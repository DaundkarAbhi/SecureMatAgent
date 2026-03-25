"""
DataAnomalyTool — statistical and range-based data integrity checker.

Detects outliers (IQR method), duplicate values, and values outside
physically reasonable ranges for common materials properties.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import NamedTuple

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Property range definitions
# ---------------------------------------------------------------------------


class _Range(NamedTuple):
    low: float
    high: float
    unit: str


_PROPERTY_RANGES: dict[str, _Range] = {
    "lattice_param": _Range(2.0, 20.0, "A"),
    "density": _Range(1.0, 23.0, "g/cm^3"),
    "two_theta": _Range(5.0, 90.0, "deg"),
    "d_spacing": _Range(0.5, 20.0, "A"),
    "temperature": _Range(0.0, 5000.0, "K"),
    "band_gap": _Range(0.0, 15.0, "eV"),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AnomalyReport:
    outliers: list[str] = field(default_factory=list)
    duplicates: list[str] = field(default_factory=list)
    range_violations: list[str] = field(default_factory=list)
    summary: str = ""

    def is_clean(self) -> bool:
        return not (self.outliers or self.duplicates or self.range_violations)

    def to_string(self) -> str:
        lines: list[str] = []
        if self.is_clean():
            lines.append("OK No anomalies detected.")
        else:
            lines.append("Anomaly Report:")
            if self.range_violations:
                lines.append("\n  Range violations:")
                lines.extend(f"    • {v}" for v in self.range_violations)
            if self.outliers:
                lines.append("\n  Statistical outliers (IQR method):")
                lines.extend(f"    • {o}" for o in self.outliers)
            if self.duplicates:
                lines.append("\n  Duplicate values:")
                lines.extend(f"    • {d}" for d in self.duplicates)
        if self.summary:
            lines.append(f"\n  Summary: {self.summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_labeled_values(text: str) -> list[tuple[str, float]]:
    """
    Extract (label, value) pairs from text like:
      'density = 5.3 g/cm3', 'lattice parameter: 3.52 A',
      or bare numbers like '3.52, 4.10, 2.98'.
    """
    pairs: list[tuple[str, float]] = []

    # Pattern 1: label = number  or  label: number
    labeled = re.findall(
        r"([a-zA-Z][a-zA-Z0-9 _-]{1,30}?)\s*[=:]\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)",
        text,
    )
    for label, val_str in labeled:
        try:
            pairs.append((label.strip(), float(val_str)))
        except ValueError:
            pass

    # Pattern 2: bare numbers in a list / table
    bare = re.findall(r"(?<![=:a-zA-Z])\b([-+]?\d*\.\d+(?:[eE][-+]?\d+)?)\b", text)
    for val_str in bare:
        try:
            v = float(val_str)
            # Avoid re-adding values already captured via labeled pattern
            if not any(abs(v - p[1]) < 1e-12 for p in pairs):
                pairs.append((f"value_{val_str}", v))
        except ValueError:
            pass

    return pairs


def _iqr_outliers(values: list[float]) -> list[int]:
    """Return indices of IQR outliers. Needs ≥4 points for reliability."""
    if len(values) < 4:
        return []
    sorted_v = sorted(values)
    n = len(sorted_v)
    q1 = statistics.median(sorted_v[: n // 2])
    q3 = statistics.median(sorted_v[(n + 1) // 2 :])
    iqr = q3 - q1
    if iqr == 0:
        return []
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return [i for i, v in enumerate(values) if v < lo or v > hi]


def _detect_duplicates(pairs: list[tuple[str, float]]) -> list[str]:
    seen: dict[float, list[str]] = {}
    for label, val in pairs:
        seen.setdefault(val, []).append(label)
    return [
        f"{val}: appears {len(labels)} times ({', '.join(labels)})"
        for val, labels in seen.items()
        if len(labels) > 1
    ]


def _infer_property(label: str, value: float) -> str | None:
    """Guess which property range to check based on label keywords."""
    label_l = label.lower()
    if any(
        kw in label_l
        for kw in ("lattice", "param", "a_", "b_", "c_", " a ", " b ", " c ")
    ):
        return "lattice_param"
    if any(kw in label_l for kw in ("density", "rho", "ρ")):
        return "density"
    if any(kw in label_l for kw in ("2theta", "2-theta", "two theta", "bragg")):
        return "two_theta"
    if any(kw in label_l for kw in ("d-spacing", "d spacing", "dspacing")):
        return "d_spacing"
    if any(kw in label_l for kw in ("temp", "kelvin")):
        return "temperature"
    if any(kw in label_l for kw in ("band gap", "bandgap", "eg ", "e_g")):
        return "band_gap"
    # Heuristic range-based fallback
    if 2.0 <= value <= 20.0:
        return "lattice_param"
    if 1.0 <= value <= 23.0:
        return "density"
    return None


def _check_ranges(pairs: list[tuple[str, float]]) -> list[str]:
    violations: list[str] = []
    for label, value in pairs:
        prop = _infer_property(label, value)
        if prop is None:
            continue
        r = _PROPERTY_RANGES[prop]
        if not (r.low <= value <= r.high):
            violations.append(
                f"{label} = {value} {r.unit} is outside expected range "
                f"[{r.low}–{r.high} {r.unit}] for {prop.replace('_', ' ')}"
            )
    return violations


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------


@tool
def data_anomaly_checker(query: str) -> str:
    """Use when asked to verify data integrity, check for anomalies, or validate \
material property values against expected ranges."""
    try:
        pairs = _extract_labeled_values(query)

        if not pairs:
            return (
                "No numeric data found in the input. "
                "Please provide values such as 'density = 5.3' or 'lattice param: 3.52'."
            )

        report = AnomalyReport()
        values = [v for _, v in pairs]

        # Range violations
        report.range_violations = _check_ranges(pairs)

        # IQR outliers
        outlier_indices = _iqr_outliers(values)
        for i in outlier_indices:
            label, val = pairs[i]
            report.outliers.append(
                f"{label} = {val} (statistical outlier among provided values)"
            )

        # Duplicates
        report.duplicates = _detect_duplicates(pairs)

        # Summary
        n_issues = (
            len(report.range_violations) + len(report.outliers) + len(report.duplicates)
        )
        report.summary = (
            f"Checked {len(pairs)} value(s). " f"{n_issues} issue(s) found."
            if n_issues
            else f"Checked {len(pairs)} value(s). All within expected ranges."
        )

        return report.to_string()

    except Exception as exc:
        return f"Anomaly checker error: {exc}"
