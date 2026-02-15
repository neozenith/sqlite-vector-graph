"""SQL identifier validation to prevent injection in dynamic queries."""

import re

# Match SQLite-safe identifiers: start with letter or underscore, then alphanumeric + underscore
_IDENTIFIER_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def validate_identifier(name: str) -> str:
    """Validate a SQL identifier (table or column name).

    Returns the name if valid, raises ValueError if not.
    This mirrors the C-side id_validate.c logic.
    """
    if not name or not _IDENTIFIER_RE.match(name):
        msg = f"Invalid SQL identifier: {name!r}"
        raise ValueError(msg)
    return name
