"""Put the checkout root on sys.path so the suite runs from anywhere.

memgraph/, injudge/, and util.py are top-level imports; pytest's prepend
import mode only adds the test directory itself, so the root must be added
explicitly for the tests' `memgraph.*` / `injudge.*` imports to resolve —
under any checkout directory name, from the root or from its parent.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
