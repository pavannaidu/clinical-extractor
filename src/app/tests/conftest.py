"""Make the app package importable when pytest runs from the repository root."""

import sys
from pathlib import Path

APP_ROOT = str(Path(__file__).resolve().parents[1])
if APP_ROOT not in sys.path:
    sys.path.insert(0, APP_ROOT)
