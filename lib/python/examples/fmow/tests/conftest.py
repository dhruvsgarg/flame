import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_FMOW_DIR = _HERE.parent
_EXAMPLES_DIR = _HERE.parent.parent

sys.path.insert(0, str(_EXAMPLES_DIR))
sys.path.insert(0, str(_FMOW_DIR / "dependencies"))