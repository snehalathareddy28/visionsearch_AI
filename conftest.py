"""conftest.py — Add the project root to sys.path so all packages are importable."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
