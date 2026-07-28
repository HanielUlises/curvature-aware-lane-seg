"""Pytest bootstrap: put the repo root on ``sys.path`` for ``src``/``scripts`` imports."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).parent.resolve())
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
