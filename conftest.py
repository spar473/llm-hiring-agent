"""Repository-wide pytest configuration.

Use a new process-owned temporary directory for every test run.  A fixed
repository basetemp and pytest's shared ``pytest-of-<user>`` directory can both
retain unusable Windows ACLs after an interrupted or sandboxed run.
"""

from __future__ import annotations

import tempfile
from typing import Any


def pytest_configure(config: Any) -> None:
    """Provide a safe basetemp unless the caller explicitly selected one."""
    if config.option.basetemp is not None:
        return

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="hiring-agent-pytest-"
    )
    config.option.basetemp = temporary_directory.name
    config._hiring_agent_temporary_directory = temporary_directory


def pytest_unconfigure(config: Any) -> None:
    """Remove the process-owned basetemp after pytest has released its files."""
    temporary_directory = getattr(
        config, "_hiring_agent_temporary_directory", None
    )
    if temporary_directory is not None:
        temporary_directory.cleanup()
