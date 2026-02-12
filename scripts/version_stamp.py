"""Stamp the current version from VERSION into known locations.

Only updates explicitly declared version references — never touches
version numbers belonging to other projects (e.g. SQLite 3.38.0).

Usage:
    python scripts/version_stamp.py          # stamp and report changes
    python scripts/version_stamp.py --check  # exit 1 if any file is out of date
"""

import logging
import pathlib
import re
import sys

log = logging.getLogger(__name__)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION = (PROJECT_ROOT / "VERSION").read_text().strip()

# Each entry: (file relative to project root, compiled regex, replacement template)
# The regex must capture a group so we can surgically replace only the version part.
TARGETS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "skills/muninn/SKILL.md",
        re.compile(r'(  version:\s*")[\d]+\.[\d]+\.[\d]+(")'),
        rf"\g<1>{VERSION}\2",
    ),
    (
        "npm/package.json",
        re.compile(r'("version":\s*")[\d]+\.[\d]+\.[\d]+(")'),
        rf"\g<1>{VERSION}\2",
    ),
]


def stamp(*, check_only: bool = False) -> bool:
    """Update version in all target files. Returns True if all files are up to date."""
    all_ok = True

    for rel_path, pattern, replacement in TARGETS:
        fpath = PROJECT_ROOT / rel_path
        if not fpath.exists():
            log.warning("skip: %s (not found)", rel_path)
            continue

        original = fpath.read_text()
        updated = pattern.sub(replacement, original)

        if original == updated:
            log.info("ok:   %s (already %s)", rel_path, VERSION)
            continue

        if check_only:
            log.error("stale: %s (needs update to %s)", rel_path, VERSION)
            all_ok = False
        else:
            fpath.write_text(updated)
            log.info("stamp: %s -> %s", rel_path, VERSION)

    return all_ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    check_only = "--check" in sys.argv

    ok = stamp(check_only=check_only)
    if not ok:
        log.error("Version mismatch detected. Run: python scripts/version_stamp.py")
        raise SystemExit(1)

    if not check_only:
        log.info("Version %s stamped into all targets.", VERSION)
