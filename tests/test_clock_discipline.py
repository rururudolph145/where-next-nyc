"""Clock-discipline regression guard (AC #11).

Ensures no production module under ``app/`` (excluding ``app/core/clock.py``)
contains bare ``datetime.utcnow``, ``datetime.now(`` without a ``tz=`` keyword
argument, or a direct ``time.sleep(`` call.

These patterns bypass the ``Clock`` protocol and break deterministic time
control in tests and the ``DEMO_MODE`` simulation loop.
"""

from __future__ import annotations

import re
from pathlib import Path


# Repository root is two levels above tests/
_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP_DIR = _REPO_ROOT / "app"
_CLOCK_MODULE = _APP_DIR / "core" / "clock.py"

# Patterns that indicate banned direct-time usage:
#   1. datetime.utcnow  — deprecated and returns naive datetime
#   2. datetime.now(     — without tz= keyword (positional tz is ambiguous)
#   3. time.sleep(       — blocks the thread; use Clock or Event.wait instead
_BANNED_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (
        re.compile(r"datetime\.utcnow"),
        "datetime.utcnow (use Clock.now() instead)",
    ),
    (
        re.compile(r"datetime\.now\((?!.*tz=)"),
        "datetime.now( without tz= keyword (use Clock.now() instead)",
    ),
    (
        re.compile(r"(?<!\w)time\.sleep\("),
        "time.sleep( (use Clock or threading.Event.wait instead)",
    ),
]


class TestClockDiscipline:
    """Repo-wide grep guard ensuring production code uses the Clock protocol."""

    def test_no_production_module_uses_banned_time_calls(self) -> None:
        """AC #11: no file under app/ (except clock.py) uses banned patterns.

        Scans every .py file under app/, skipping app/core/clock.py which
        legitimately wraps datetime.now(tz=timezone.utc).
        """
        # Arrange
        violations: list[str] = []
        py_files = sorted(_APP_DIR.rglob("*.py"))

        # Act
        for py_file in py_files:
            if py_file.resolve() == _CLOCK_MODULE.resolve():
                continue

            source = py_file.read_text(encoding="utf-8")
            for line_no, line in enumerate(source.splitlines(), start=1):
                # Skip comment-only lines
                stripped = line.lstrip()
                if stripped.startswith("#"):
                    continue
                for pattern, description in _BANNED_PATTERNS:
                    if pattern.search(line):
                        rel = py_file.relative_to(_REPO_ROOT)
                        violations.append(
                            f"  {rel}:{line_no}: {description}\n    {line.strip()}"
                        )

        # Assert
        assert not violations, (
            "Production code must use the Clock protocol instead of "
            "direct time calls.\n"
            "Violations found:\n" + "\n".join(violations)
        )
