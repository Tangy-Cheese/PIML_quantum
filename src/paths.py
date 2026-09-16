import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
OUTPUT = ROOT / "output"

# Where final report figures are copied, in addition to their output/ category.
# Override with the ATI_FIGURE_DIR environment variable when the dissertation
# source lives elsewhere.
FIGURE_DIR = Path(
    os.environ.get("ATI_FIGURE_DIR", Path.home() / "Documents" / "dissertation")
)


def out(category: str) -> Path:
    """Return output/<category>/, creating it if it does not exist."""
    d = OUTPUT / category
    d.mkdir(parents=True, exist_ok=True)
    return d


def figure_dir() -> Path:
    """Return the report figure directory if it exists, else None.

    Scripts fall back to writing only into output/ when the dissertation
    source tree is not present, so figure generation never hard-fails on a
    fresh clone.
    """
    return FIGURE_DIR if FIGURE_DIR.is_dir() else None
