"""alliGAITor's GUI job queue."""

from __future__ import annotations

import sys
from pathlib import Path

# gui/ and tools/ modules use flat imports among themselves, so both
# directories (plus the repo root, for `import alligaitor`) must be on
# sys.path.
REPO_DIR = Path(__file__).resolve().parent.parent
for _p in (REPO_DIR, REPO_DIR / "gui", REPO_DIR / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
