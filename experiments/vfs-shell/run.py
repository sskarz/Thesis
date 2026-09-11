#!/usr/bin/env python3
"""Run toy VFS/shell tasks without setting PYTHONPATH."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from vfs_shell.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
