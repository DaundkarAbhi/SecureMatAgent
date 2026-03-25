"""Tests for agent tools (no external dependencies)."""

import pytest

from app.agent.tools import SympyCalculatorTool


@pytest.fixture
def calc():
    return SympyCalculatorTool()


def test_calculator_basic(calc):
    assert calc._run("2 + 2") == "4"


def test_calculator_float(calc):
    result = calc._run("3.14 * 2")
    assert "6.28" in result


def test_calculator_symbolic(calc):
    result = calc._run("x**2 + 2*x + 1")
    assert "x" in result


def test_calculator_blocks_import(calc):
    result = calc._run("import os")
    assert "disallowed" in result.lower()


def test_calculator_blocks_exec(calc):
    result = calc._run("exec('print(1)')")
    assert "disallowed" in result.lower()


def test_calculator_bad_expression(calc):
    result = calc._run("not_a_valid_expression!!!")
    assert "error" in result.lower()
