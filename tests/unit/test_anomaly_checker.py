"""
Unit tests for src/tools/anomaly_checker.py — DataAnomalyTool.

Tests cover: IQR outlier detection, range validation, duplicate detection,
and edge cases (empty input, too few points, clean data).
"""

from __future__ import annotations

import pytest

from src.tools.anomaly_checker import (AnomalyReport, _check_ranges,
                                       _detect_duplicates,
                                       _extract_labeled_values, _iqr_outliers,
                                       data_anomaly_checker)

pytestmark = pytest.mark.unit


# ===========================================================================
# _iqr_outliers
# ===========================================================================


class TestIQROutliers:
    def test_detects_obvious_outlier(self):
        """
        15.7 is a clear outlier among a tight cluster.
        Need enough points so 15.7 doesn't inflate Q3.
        Using 10 values: 9 tight + 1 extreme.
        """
        values = [3.90, 3.95, 3.96, 3.97, 3.98, 3.99, 4.00, 4.01, 4.02, 15.7]
        indices = _iqr_outliers(values)
        assert 9 in indices  # index of 15.7

    def test_no_outliers_in_clean_data(self):
        values = [3.9, 4.0, 3.95, 4.01]
        indices = _iqr_outliers(values)
        assert indices == []

    def test_fewer_than_4_points_returns_empty(self):
        """IQR needs ≥4 points; fewer → no outliers reported."""
        assert _iqr_outliers([1.0, 100.0, 2.0]) == []

    def test_empty_list(self):
        assert _iqr_outliers([]) == []

    def test_all_identical_values(self):
        """IQR=0 → no outliers (avoids divide-by-zero)."""
        assert _iqr_outliers([5.0, 5.0, 5.0, 5.0, 5.0]) == []

    def test_single_outlier_at_start(self):
        """0.001 is far below a tight cluster of ~4.0 values."""
        values = [0.001, 3.90, 3.95, 3.97, 3.98, 3.99, 4.00, 4.01, 4.02, 4.03]
        indices = _iqr_outliers(values)
        assert 0 in indices  # index of 0.001

    def test_multiple_outliers(self):
        """Two extreme outliers at both ends of a tight cluster."""
        values = [0.001, 3.90, 3.95, 3.97, 3.98, 3.99, 4.00, 4.01, 4.02, 100.0]
        indices = _iqr_outliers(values)
        assert len(indices) >= 1


# ===========================================================================
# _extract_labeled_values
# ===========================================================================


class TestExtractLabeledValues:
    def test_labeled_equals(self):
        pairs = _extract_labeled_values("density = 5.3 g/cm3")
        labels = [p[0] for p in pairs]
        values = [p[1] for p in pairs]
        assert any(abs(v - 5.3) < 1e-9 for v in values)

    def test_labeled_colon(self):
        pairs = _extract_labeled_values("lattice parameter: 3.52")
        values = [p[1] for p in pairs]
        assert any(abs(v - 3.52) < 1e-9 for v in values)

    def test_bare_numbers(self):
        pairs = _extract_labeled_values("3.52, 4.10, 2.98")
        assert len(pairs) == 3

    def test_empty_string(self):
        pairs = _extract_labeled_values("")
        assert pairs == []

    def test_mixed_labeled_and_bare(self):
        text = "density = 5.3 and values 3.52, 2.98"
        pairs = _extract_labeled_values(text)
        values = [p[1] for p in pairs]
        assert 5.3 in values or any(abs(v - 5.3) < 1e-9 for v in values)


# ===========================================================================
# _detect_duplicates
# ===========================================================================


class TestDetectDuplicates:
    def test_detects_duplicate(self):
        pairs = [("a", 3.52), ("b", 4.10), ("c", 3.52)]
        dupes = _detect_duplicates(pairs)
        assert len(dupes) == 1
        assert "3.52" in dupes[0]

    def test_no_duplicates(self):
        pairs = [("a", 3.52), ("b", 4.10), ("c", 2.98)]
        dupes = _detect_duplicates(pairs)
        assert dupes == []

    def test_empty_input(self):
        assert _detect_duplicates([]) == []

    def test_triple_duplicate(self):
        pairs = [("a", 1.0), ("b", 1.0), ("c", 1.0)]
        dupes = _detect_duplicates(pairs)
        assert len(dupes) == 1
        assert "3 times" in dupes[0]


# ===========================================================================
# _check_ranges
# ===========================================================================


class TestCheckRanges:
    def test_lattice_param_out_of_range(self):
        """Cubic perovskite lattice=45.2 Å is clearly above 20 Å max."""
        pairs = [("lattice param", 45.2)]
        violations = _check_ranges(pairs)
        assert len(violations) == 1
        assert "lattice" in violations[0].lower() or "param" in violations[0].lower()

    def test_normal_lattice_no_violation(self):
        pairs = [("lattice param", 3.99)]
        violations = _check_ranges(pairs)
        assert violations == []

    def test_density_in_range(self):
        pairs = [("density", 6.02)]
        violations = _check_ranges(pairs)
        assert violations == []

    def test_temperature_extreme(self):
        pairs = [("temperature", 6000.0)]
        violations = _check_ranges(pairs)
        # 6000 K is above the 5000 K max
        assert len(violations) == 1

    def test_band_gap_negative(self):
        pairs = [("band gap", -1.0)]
        violations = _check_ranges(pairs)
        assert len(violations) == 1


# ===========================================================================
# Full tool interface tests
# ===========================================================================


class TestDataAnomalyCheckerTool:
    def test_flags_statistical_outlier(self):
        """The tool should flag 15.7 as an outlier (needs 10 values for IQR)."""
        query = "3.90, 3.95, 3.96, 3.97, 3.98, 3.99, 4.00, 4.01, 4.02, 15.7"
        result = data_anomaly_checker.invoke(query)
        assert isinstance(result, str)
        assert "15.7" in result or "outlier" in result.lower()

    def test_clean_data_ok(self):
        """Four close values → clean data report."""
        query = "3.9, 4.0, 3.95, 4.01"
        result = data_anomaly_checker.invoke(query)
        assert isinstance(result, str)
        # Could be "OK" or just no anomalies
        assert "no anomal" in result.lower() or "ok" in result.lower()

    def test_range_violation_lattice(self):
        """Lattice param 45.2 Å → range violation."""
        query = "lattice param = 45.2"
        result = data_anomaly_checker.invoke(query)
        assert (
            "violation" in result.lower()
            or "outside" in result.lower()
            or "range" in result.lower()
        )

    def test_duplicate_detection(self):
        """Two identical labeled values → duplicate reported."""
        query = "density = 5.3, density = 5.3, density = 6.1, density = 5.8"
        result = data_anomaly_checker.invoke(query)
        assert "duplicate" in result.lower() or "5.3" in result

    def test_empty_input_returns_message(self):
        """No numeric data → informative message, no exception."""
        result = data_anomaly_checker.invoke("no numbers here at all")
        assert isinstance(result, str)
        assert "no numeric" in result.lower() or "please provide" in result.lower()

    def test_no_exception_on_malformed_input(self):
        """Malformed input must never raise; returns a string."""
        result = data_anomaly_checker.invoke("!!!@@@###")
        assert isinstance(result, str)

    def test_anomaly_report_is_clean(self):
        report = AnomalyReport()
        assert report.is_clean() is True

    def test_anomaly_report_not_clean(self):
        report = AnomalyReport(outliers=["val=99.0"])
        assert report.is_clean() is False

    def test_anomaly_report_to_string_clean(self):
        report = AnomalyReport()
        s = report.to_string()
        assert "no anomal" in s.lower() or "ok" in s.lower()

    def test_anomaly_report_to_string_with_outlier(self):
        report = AnomalyReport(outliers=["x = 99.0 (statistical outlier)"])
        s = report.to_string()
        assert "outlier" in s.lower()
