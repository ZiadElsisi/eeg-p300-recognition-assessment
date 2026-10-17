"""Generate a concise Markdown report after every pytest run."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import shlex
import time
import sys

import pytest

_started_at = pytest.StashKey[float]()


def pytest_sessionstart(session: pytest.Session) -> None:
    session.config.stash[_started_at] = time.monotonic()


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write the latest test summary to reports/unit_test_report.md."""
    root = Path(session.config.rootpath)
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_path = reports_dir / "unit_test_report.md"

    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    stats = reporter.stats if reporter is not None else {}
    passed = len(stats.get("passed", []))
    failed = len(stats.get("failed", []))
    errors = len(stats.get("error", []))
    skipped = len(stats.get("skipped", []))
    xfailed = len(stats.get("xfailed", []))
    xpassed = len(stats.get("xpassed", []))

    files = Counter()
    for item in session.items:
        test_file = item.nodeid.split("::", 1)[0]
        files[test_file] += 1

    finished = datetime.now().astimezone()
    duration = time.monotonic() - session.config.stash[_started_at]
    duration_text = f"{duration:.2f} seconds"

    result = "PASSED" if exitstatus == pytest.ExitCode.OK else "FAILED"
    lines = [
        "# Unit Test Report",
        "",
        "- **Project:** EEG P300 Recognition Assessment",
        f"- **Run date:** {finished:%Y-%m-%d %H:%M:%S %Z}",
        f"- **Command:** `{shlex.join(sys.argv)}`",
        f"- **Overall status:** {result}",
        f"- **Results:** {passed} passed; {failed} failed; {errors} errors; {skipped} skipped; {xfailed} xfailed; {xpassed} xpassed",
        f"- **Duration:** {duration_text}",
        "",
        "## Test files collected",
        "",
    ]
    if files:
        lines.extend(f"- `{name}` — {count} test(s)" for name, count in sorted(files.items()))
    else:
        lines.append("No tests were collected.")
    lines.extend(
        [
            "",
            "## Scope",
            "",
            "This report is refreshed automatically at the end of every pytest run. It summarizes the tests collected and their outcomes; it does not represent validation of physical EEG hardware or trigger timing.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
