"""
MaterialsCalculatorTool — symbolic/numeric calculations for materials science.

Handles Bragg angles, lattice parameters, density from unit cell, and common
unit conversions (eV ↔ nm, eV ↔ cm⁻¹, A ↔ nm).
"""

from __future__ import annotations

import math
import re
from typing import Any

from langchain_core.tools import tool

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
_CU_KA = 1.5406  # A — Cu Kα wavelength
_NA = 6.02214076e23  # mol⁻¹ — Avogadro's number
_HC_EV_NM = 1239.84  # eV·nm — hc in convenient units
_HC_EV_CM = 8065.54  # cm⁻¹ per eV


# ---------------------------------------------------------------------------
# Safe expression evaluator (no builtins, math functions only)
# ---------------------------------------------------------------------------
_SAFE_NAMES: dict[str, Any] = {
    k: getattr(math, k) for k in dir(math) if not k.startswith("_")
}
_SAFE_NAMES["abs"] = abs
_SAFE_NAMES["round"] = round


def _safe_eval(expr: str) -> float:
    """Evaluate a numeric expression with math functions; no arbitrary builtins."""
    return float(eval(expr, {"__builtins__": {}}, _SAFE_NAMES))  # noqa: S307


# ---------------------------------------------------------------------------
# Individual calculation helpers
# ---------------------------------------------------------------------------


def _bragg_angle(d: float, lam: float = _CU_KA, n: int = 1) -> dict[str, float]:
    """Return 2θ in degrees for given d-spacing (Å) and wavelength (Å)."""
    ratio = (n * lam) / (2 * d)
    if not (-1 <= ratio <= 1):
        raise ValueError(f"n·λ/(2d) = {ratio:.4f} is outside [-1, 1]; check inputs.")
    theta_rad = math.asin(ratio)
    return {
        "theta_deg": math.degrees(theta_rad),
        "two_theta_deg": 2 * math.degrees(theta_rad),
        "d_spacing_A": d,
        "wavelength_A": lam,
    }


def _lattice_cubic(d: float, h: int, k: int, l: int) -> float:
    """Cubic: a = d · √(h²+k²+l²)."""
    return d * math.sqrt(h**2 + k**2 + l**2)


def _lattice_tetragonal(
    d: float, h: int, k: int, l: int, c_over_a: float
) -> dict[str, float]:
    """
    Tetragonal: 1/d² = (h²+k²)/a² + l²/c².
    Requires the c/a ratio as additional input.
    """
    # Express as: a² = (h²+k²) / (1/d² - l²/c²)  with c = c_over_a * a
    # 1/d² = (h²+k²)/a² + l²/(c_over_a*a)²
    # Let x = 1/a²:  x·(h²+k² + l²/c_over_a²) = 1/d²
    x = 1 / d**2 / (h**2 + k**2 + l**2 / c_over_a**2)
    a = 1 / math.sqrt(x)
    c = c_over_a * a
    return {"a_A": a, "c_A": c}


def _lattice_hexagonal(
    d: float, h: int, k: int, l: int, c_over_a: float
) -> dict[str, float]:
    """
    Hexagonal: 1/d² = (4/3)·(h²+hk+k²)/a² + l²/c².
    Requires the c/a ratio.
    """
    hk = h**2 + h * k + k**2
    # 1/d² = (4/3)·hk/a² + l²/(c_over_a·a)²
    # = (1/a²)·[(4/3)·hk + l²/c_over_a²]
    x = 1 / d**2 / ((4 / 3) * hk + l**2 / c_over_a**2)
    a = 1 / math.sqrt(x)
    c = c_over_a * a
    return {"a_A": a, "c_A": c}


def _density(Z: float, M: float, V_A3: float) -> float:
    """
    Unit-cell density in g/cm³.
    Z  — number of formula units per cell
    M  — molar mass in g/mol
    V  — unit cell volume in A³
    """
    V_cm3 = V_A3 * 1e-24  # A³ → cm³
    return (Z * M) / (_NA * V_cm3)


def _unit_conversion(value: float, from_unit: str, to_unit: str) -> float:
    """Convert between eV, nm, cm⁻¹, and A."""
    from_unit = from_unit.lower().strip()
    to_unit = to_unit.lower().strip()

    # Normalise aliases
    aliases = {
        "angstrom": "a",
        "angstroms": "a",
        "å": "a",
        "nanometer": "nm",
        "nanometers": "nm",
        "electron volt": "ev",
        "electron volts": "ev",
        "wavenumber": "cm-1",
        "wavenumbers": "cm-1",
        "cm^-1": "cm-1",
    }
    from_unit = aliases.get(from_unit, from_unit)
    to_unit = aliases.get(to_unit, to_unit)

    if from_unit == to_unit:
        return value

    # Build a canonical energy in eV first, then convert out
    ev: float
    if from_unit == "ev":
        ev = value
    elif from_unit == "nm":
        ev = _HC_EV_NM / value
    elif from_unit == "cm-1":
        ev = value / _HC_EV_CM
    elif from_unit == "a":
        ev = _HC_EV_NM / (value * 0.1)
    else:
        raise ValueError(f"Unknown source unit: {from_unit!r}")

    if to_unit == "ev":
        return ev
    elif to_unit == "nm":
        return _HC_EV_NM / ev
    elif to_unit == "cm-1":
        return ev * _HC_EV_CM
    elif to_unit == "a":
        return (_HC_EV_NM / ev) / 0.1
    else:
        raise ValueError(f"Unknown target unit: {to_unit!r}")


# ---------------------------------------------------------------------------
# Keyword-based intent parser
# ---------------------------------------------------------------------------


def _extract_floats(text: str) -> list[float]:
    return [float(m) for m in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)]


def _extract_miller(text: str) -> tuple[int, int, int] | None:
    m = re.search(r"\((\d)\s*(\d)\s*(\d)\)", text)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    m = re.search(r"h\s*=\s*(\d+).*?k\s*=\s*(\d+).*?l\s*=\s*(\d+)", text, re.I)
    if m:
        return int(m.group(1)), int(m.group(2)), int(m.group(3))
    return None


def _parse_and_calculate(query: str) -> str:
    q = query.lower()
    nums = _extract_floats(query)

    # ---- Bragg angle -------------------------------------------------------
    if any(
        kw in q
        for kw in ("bragg", "2-theta", "2theta", "two-theta", "diffraction angle")
    ):
        if not nums:
            return "Please provide a d-spacing value (in A)."
        d = nums[0]
        lam = nums[1] if len(nums) > 1 else _CU_KA
        res = _bragg_angle(d, lam)
        return (
            f"Bragg calculation:\n"
            f"  d-spacing     = {res['d_spacing_A']:.4f} A\n"
            f"  Wavelength    = {res['wavelength_A']:.4f} A\n"
            f"  theta         = {res['theta_deg']:.4f} deg\n"
            f"  2theta        = {res['two_theta_deg']:.4f} deg"
        )

    # ---- Density -----------------------------------------------------------
    if "density" in q or "rho" in q or "unit cell" in q:
        if len(nums) < 3:
            return (
                "Density calculation needs three values: "
                "Z (formula units), M (g/mol), V (Å³). "
                f"Found: {nums}"
            )
        Z, M, V = nums[0], nums[1], nums[2]
        rho = _density(Z, M, V)
        return (
            f"Density calculation:\n"
            f"  Z    = {Z}\n"
            f"  M    = {M} g/mol\n"
            f"  V    = {V} A^3\n"
            f"  rho  = {rho:.4f} g/cm^3"
        )

    # ---- Lattice parameter — cubic -----------------------------------------
    if "cubic" in q and (
        "lattice" in q or "parameter" in q or "d-spacing" in q or "d spacing" in q
    ):
        hkl = _extract_miller(query)
        if not nums:
            return "Please provide a d-spacing value (in A)."
        d = nums[0]
        if hkl is None:
            return "Please provide Miller indices, e.g. (1 1 0)."
        h, k, l = hkl
        a = _lattice_cubic(d, h, k, l)
        return (
            f"Cubic lattice parameter:\n"
            f"  d({h}{k}{l}) = {d:.4f} A\n"
            f"  a           = {a:.4f} A"
        )

    # ---- Lattice parameter — tetragonal ------------------------------------
    if "tetragonal" in q:
        hkl = _extract_miller(query)
        if len(nums) < 2 or hkl is None:
            return (
                "Tetragonal needs: d-spacing, c/a ratio, and Miller indices (hkl). "
                f"Found numbers: {nums}, indices: {hkl}"
            )
        d, c_over_a = nums[0], nums[1]
        h, k, l = hkl
        res = _lattice_tetragonal(d, h, k, l, c_over_a)
        return (
            f"Tetragonal lattice parameters:\n"
            f"  a = {res['a_A']:.4f} A\n"
            f"  c = {res['c_A']:.4f} A"
        )

    # ---- Lattice parameter — hexagonal -------------------------------------
    if "hexagonal" in q:
        hkl = _extract_miller(query)
        if len(nums) < 2 or hkl is None:
            return (
                "Hexagonal needs: d-spacing, c/a ratio, and Miller indices (hkl). "
                f"Found numbers: {nums}, indices: {hkl}"
            )
        d, c_over_a = nums[0], nums[1]
        h, k, l = hkl
        res = _lattice_hexagonal(d, h, k, l, c_over_a)
        return (
            f"Hexagonal lattice parameters:\n"
            f"  a = {res['a_A']:.4f} A\n"
            f"  c = {res['c_A']:.4f} A"
        )

    # ---- Unit conversion ---------------------------------------------------
    unit_patterns = [
        (
            r"(\d[\d.eE+\-]*)\s*(ev|nm|cm-1|cm\^-1|[aå]ngstrom\w*)\s+(?:to|in|→)\s*(ev|nm|cm-1|cm\^-1|[aå]ngstrom\w*)",
            1,
            2,
            3,
        ),
        (
            r"convert\s+(\d[\d.eE+\-]*)\s*(ev|nm|cm-1|cm\^-1|[aå]ngstrom\w*)\s+(?:to|into)\s*(ev|nm|cm-1|cm\^-1|[aå]ngstrom\w*)",
            1,
            2,
            3,
        ),
    ]
    for pat, vi, fi, ti in unit_patterns:
        m = re.search(pat, q)
        if m:
            val = float(m.group(vi))
            result = _unit_conversion(val, m.group(fi), m.group(ti))
            return f"{val} {m.group(fi)} = {result:.6g} {m.group(ti)}"

    if any(kw in q for kw in ("ev to", "nm to", "cm-1 to", "angstrom to", "convert")):
        if len(nums) < 1:
            return "Please provide the numeric value to convert."
        # Try to pick units from the query
        unit_map = {"ev": "eV", "nm": "nm", "cm-1": "cm-1", "angstrom": "a", "å": "a"}
        found_units = [u for u in unit_map if u in q]
        if len(found_units) >= 2:
            result = _unit_conversion(nums[0], found_units[0], found_units[1])
            return f"{nums[0]} {found_units[0]} = {result:.6g} {found_units[1]}"

    # ---- Generic numeric expression ----------------------------------------
    # Last resort: try to evaluate a plain math expression
    try:
        # Strip common English words to isolate the expression
        expr = re.sub(r"[a-zA-Z,?!]+", " ", query).strip()
        expr = re.sub(r"\s+", "", expr)
        if expr:
            result = _safe_eval(expr)
            return f"Result: {result}"
    except Exception:
        pass

    return (
        "I could not identify the calculation type. Supported calculations:\n"
        "- Bragg angle: provide d-spacing (Å) and optionally wavelength\n"
        "- Density: provide Z, M (g/mol), V (Å³)\n"
        "- Cubic/tetragonal/hexagonal lattice parameters: provide d, c/a, Miller indices\n"
        "- Unit conversion: e.g. '2.5 eV to nm'"
    )


# ---------------------------------------------------------------------------
# LangChain tool
# ---------------------------------------------------------------------------


@tool
def materials_calculator(query: str) -> str:
    """Use for calculating material properties like lattice parameters, density, \
Bragg angles, d-spacing, or unit conversions between eV, nm, cm-1, and angstroms."""
    try:
        return _parse_and_calculate(query)
    except Exception as exc:
        return f"Calculator error: {exc}"
