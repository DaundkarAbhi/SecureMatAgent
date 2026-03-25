"""
Unit tests for src/tools/calculator.py — MaterialsCalculatorTool.

All tests call _parse_and_calculate() directly (the pure function) as well as
the @tool wrapper to verify both layers.  No LangChain runtime or network needed.
"""

from __future__ import annotations

import math

import pytest

from src.tools.calculator import (_bragg_angle, _density, _parse_and_calculate,
                                  _unit_conversion, materials_calculator)

pytestmark = pytest.mark.unit


# ===========================================================================
# Bragg angle
# ===========================================================================


class TestBraggAngle:
    """Direct helper and string-interface tests for Bragg angle calculation."""

    def test_cu_ka_d287_two_theta(self):
        """d=2.87 Å, Cu Kα → 2θ ≈ 31.1 degrees."""
        res = _bragg_angle(d=2.87)
        assert abs(res["two_theta_deg"] - 31.1) < 0.2

    def test_theta_is_half_two_theta(self):
        res = _bragg_angle(d=2.87)
        assert abs(res["theta_deg"] * 2 - res["two_theta_deg"]) < 1e-9

    def test_d_and_wavelength_preserved(self):
        res = _bragg_angle(d=3.52, lam=1.5406)
        assert res["d_spacing_A"] == pytest.approx(3.52)
        assert res["wavelength_A"] == pytest.approx(1.5406)

    def test_custom_wavelength(self):
        """Mo Kα λ=0.7107 Å gives a smaller 2θ than Cu Kα for same d."""
        res_cu = _bragg_angle(d=2.0, lam=1.5406)
        res_mo = _bragg_angle(d=2.0, lam=0.7107)
        assert res_mo["two_theta_deg"] < res_cu["two_theta_deg"]

    def test_n_equals_2(self):
        """Second-order (n=2) reflection has larger angle."""
        res1 = _bragg_angle(d=2.87, n=1)
        res2 = _bragg_angle(d=2.87, n=2)
        assert res2["two_theta_deg"] > res1["two_theta_deg"]

    def test_invalid_ratio_raises(self):
        """n·λ/(2d) > 1 should raise ValueError."""
        with pytest.raises(ValueError, match="outside"):
            _bragg_angle(d=0.1, lam=1.5406, n=1)

    def test_query_string_bragg(self):
        """String interface: 'bragg d=2.87' returns result with 2theta."""
        result = _parse_and_calculate("bragg angle for d=2.87 A")
        assert "2theta" in result.lower() or "2θ" in result.lower() or "31." in result

    def test_query_string_two_theta_keyword(self):
        # "two-theta" keyword avoids "2" being parsed as d-spacing
        result = _parse_and_calculate("two-theta for d-spacing 2.87")
        assert "31." in result

    @pytest.mark.parametrize(
        "d,expected_2theta",
        [
            (2.87, 31.1),
            (2.09, 43.3),  # Fe (110) approximate
            (1.44, 64.8),  # approximate
        ],
    )
    def test_parametrize_bragg(self, d, expected_2theta):
        res = _bragg_angle(d=d)
        assert (
            abs(res["two_theta_deg"] - expected_2theta) < 0.5
        ), f"d={d}: got {res['two_theta_deg']:.2f}, expected ~{expected_2theta}"


# ===========================================================================
# Density
# ===========================================================================


class TestDensity:
    """Tests for unit-cell density calculation."""

    def test_fcc_iron_approx(self):
        """
        FCC iron: a=2.87 Å but Z=4 (FCC). Using Z=2 as per prompt (BCC iron):
        BCC iron: a=2.87 Å, Z=2, M=55.845 g/mol → ρ ≈ 7.87 g/cm³.
        V = (2.87)³ = 23.64 Å³
        """
        V = 2.87**3
        rho = _density(Z=2, M=55.845, V_A3=V)
        assert abs(rho - 7.87) < 0.1, f"Expected ~7.87, got {rho:.4f}"

    def test_known_bcc_iron(self):
        """Explicit volume check for BCC iron density."""
        rho = _density(Z=2, M=55.845, V_A3=23.64)
        assert 7.8 < rho < 8.0

    def test_density_query_string(self):
        """String interface: density query with Z, M, V returns rho value."""
        result = _parse_and_calculate("density Z=2 M=55.845 V=23.64")
        assert "rho" in result.lower() or "density" in result.lower()
        assert "7." in result

    def test_density_missing_values(self):
        """Fewer than 3 numeric values returns helpful message."""
        result = _parse_and_calculate("density Z=2 M=55.845")
        assert "three values" in result.lower() or "found" in result.lower()

    @pytest.mark.parametrize(
        "Z,M,V_A3,expected_min,expected_max",
        [
            (4, 58.693, 43.76, 8.5, 9.0),  # FCC Ni, a=3.524 Å
            (4, 63.546, 47.24, 8.8, 9.1),  # FCC Cu, a=3.615 Å
            (2, 55.845, 23.64, 7.7, 8.0),  # BCC Fe
        ],
    )
    def test_parametrize_density(self, Z, M, V_A3, expected_min, expected_max):
        rho = _density(Z=Z, M=M, V_A3=V_A3)
        assert (
            expected_min < rho < expected_max
        ), f"Z={Z} M={M} V={V_A3}: rho={rho:.4f} not in [{expected_min}, {expected_max}]"


# ===========================================================================
# Unit conversion
# ===========================================================================


class TestUnitConversion:
    """Tests for eV ↔ nm ↔ cm⁻¹ ↔ Å conversions."""

    def test_1_5_ev_to_nm(self):
        """1.5 eV → ~827 nm (visible red/near-IR)."""
        nm = _unit_conversion(1.5, "eV", "nm")
        assert abs(nm - 826.56) < 1.0, f"Expected ~826.6 nm, got {nm:.2f}"

    def test_nm_to_ev_roundtrip(self):
        """Convert 500 nm → eV → nm should give 500."""
        ev = _unit_conversion(500.0, "nm", "eV")
        nm = _unit_conversion(ev, "eV", "nm")
        assert abs(nm - 500.0) < 1e-6

    def test_ev_to_cm_inverse(self):
        """1 eV → 8065.54 cm⁻¹ (physical constant)."""
        cm_inv = _unit_conversion(1.0, "eV", "cm-1")
        assert abs(cm_inv - 8065.54) < 0.1

    def test_angstrom_to_nm(self):
        """10 Å = 1 nm."""
        nm = _unit_conversion(10.0, "a", "nm")
        assert abs(nm - 1.0) < 1e-6

    def test_same_unit_returns_value(self):
        assert _unit_conversion(3.5, "eV", "eV") == pytest.approx(3.5)

    def test_unknown_from_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown source unit"):
            _unit_conversion(1.0, "joules", "nm")

    def test_unknown_to_unit_raises(self):
        with pytest.raises(ValueError, match="Unknown target unit"):
            _unit_conversion(1.0, "eV", "joules")

    def test_query_string_ev_to_nm(self):
        """String interface: '1.5 eV to nm' returns ~827."""
        result = _parse_and_calculate("1.5 eV to nm")
        assert "826" in result or "827" in result

    @pytest.mark.parametrize(
        "value,from_u,to_u,expected,tol",
        [
            (2.0, "eV", "nm", 619.92, 0.1),
            (400.0, "nm", "eV", 3.0996, 0.01),
            (1.0, "eV", "cm-1", 8065.54, 0.1),
            (3.1, "eV", "nm", 399.95, 0.2),
        ],
    )
    def test_parametrize_conversion(self, value, from_u, to_u, expected, tol):
        result = _unit_conversion(value, from_u, to_u)
        assert (
            abs(result - expected) < tol
        ), f"{value} {from_u}->{to_u}: got {result:.4f}, expected {expected}"


# ===========================================================================
# Invalid / edge-case inputs
# ===========================================================================


class TestInvalidAndEdge:
    """Test that the calculator handles nonsense gracefully."""

    def test_no_exception_on_gibberish(self):
        """'what is love' must return a string, never raise."""
        result = materials_calculator.invoke("what is love")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_empty_string_no_exception(self):
        result = materials_calculator.invoke("")
        assert isinstance(result, str)

    def test_negative_d_spacing_invalid(self):
        """Negative d → ratio > 1 → ValueError caught by tool wrapper."""
        result = materials_calculator.invoke("bragg angle for d=-1.0")
        assert isinstance(result, str)
        # The error is caught and returned as a string message
        assert "error" in result.lower() or "outside" in result.lower() or "1" in result

    def test_zero_wavelength_handled(self):
        """Zero wavelength → division error, must not propagate."""
        result = materials_calculator.invoke("bragg angle d=2.87 wavelength=0")
        assert isinstance(result, str)

    def test_very_large_d_spacing(self):
        """Very large d → very small angle, still valid calculation."""
        res = _bragg_angle(d=1000.0)
        assert 0.0 < res["two_theta_deg"] < 0.1

    def test_math_expression_fallback(self):
        """Pure numeric expression like '2 + 2' is evaluated."""
        result = _parse_and_calculate("2 + 2")
        assert "4" in result

    def test_unrecognised_query_returns_help(self):
        """Completely unrecognised query returns guidance, not an exception."""
        result = _parse_and_calculate("tell me a joke")
        assert isinstance(result, str)
        assert len(result) > 10
