"""Small helpers shared by other modules.

Currently exposes :func:`setup_logging`, the single entry point for
configuring the root logger so that Ukrainian messages survive a
Windows PowerShell session.
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with UTF-8 stdout for Ukrainian messages.

    On Windows, the default PowerShell code page is cp1251 and Ukrainian
    log messages render as mojibake. This function reconfigures
    ``sys.stdout`` / ``sys.stderr`` to UTF-8 (where supported) and
    installs a :class:`logging.StreamHandler` with an explicit UTF-8
    formatter.

    Idempotency strategy: any handlers already attached to the root
    logger are removed and replaced. That way, re-calling with a
    different ``level`` actually changes the active threshold instead of
    silently retaining the first call's settings.

    Args:
        level: Root logger threshold (e.g. ``logging.INFO``,
            ``logging.DEBUG``).
    """
    # Windows без Developer Mode не дозволяє симлінки, кеш HuggingFace падає
    # назад на копії файлів — це нормально, попередження нам не потрібне.
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(_LOG_FORMAT))
    root.addHandler(handler)
    root.setLevel(level)
