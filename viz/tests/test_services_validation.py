"""Tests for SQL identifier validation."""

import pytest

from server.services.validation import validate_identifier


@pytest.mark.parametrize(
    "name",
    [
        "edges",
        "my_table",
        "_private",
        "Table123",
        "a",
    ],
)
def test_valid_identifiers(name: str) -> None:
    """Valid SQL identifiers pass validation."""
    assert validate_identifier(name) == name


@pytest.mark.parametrize(
    "name",
    [
        "",
        "123abc",
        "table name",
        "drop;--",
        "table.column",
        "hello world",
        "name'injection",
        'table"name',
    ],
)
def test_invalid_identifiers(name: str) -> None:
    """Invalid SQL identifiers raise ValueError."""
    with pytest.raises(ValueError, match="Invalid SQL identifier"):
        validate_identifier(name)
