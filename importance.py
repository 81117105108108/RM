"""
importance.py - Heuristic importance classification for repository files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

IMPORTANT_EXACT_FILENAMES = {
    # Project Documentation
    "readme", "readme.md", "readme.rst", "readme.txt",
    "changelog.md", "contributing.md", "license",
    # Python
    "pyproject.toml", "setup.py", "setup.cfg", "requirements.txt",
    "pipfile", "environment.yml", "tox.ini",
    # JavaScript / TypeScript / Node
    "package.json", "tsconfig.json", "pnpm-workspace.yaml",
    "biome.json", "turbo.json",
    # Rust & Systems
    "cargo.toml", "cargo.lock", "build.rs",
    # Go
    "go.mod", "go.sum",
    # Build & Infrastructure
    "makefile", "cmakelists.txt", "dockerfile", "docker-compose.yml",
    "compose.yaml", "flake.nix", "buf.yaml"
}

IMPORTANT_DIR_PATTERNS = [
    re.compile(r"^(src|lib|app|pkg|internal|cmd)(/.*)?$", re.IGNORECASE)
]


def calculate_file_importance_score(rel_path: str) -> float:
    """
    Assigns a priority weight in [0.0, 1.0] for context selection heuristics.
    """
    path = Path(rel_path)
    name_lower = path.name.lower()

    if name_lower in IMPORTANT_EXACT_FILENAMES:
        return 1.0

    for pattern in IMPORTANT_DIR_PATTERNS:
        if pattern.match(rel_path):
            return 0.7

    # Deprioritize test and mock files in high-level architectural maps
    if "test" in name_lower or "spec" in name_lower or "mock" in name_lower:
        return 0.1

    return 0.4


def is_important(rel_file_path: str) -> bool:
    """Returns True if the file matches critical configuration or documentation patterns."""
    return calculate_file_importance_score(rel_file_path) >= 0.7


def filter_important_files(file_paths: List[str]) -> List[str]:
    """Sorts candidate file paths descending by their structural importance score."""
    return sorted(file_paths, key=calculate_file_importance_score, reverse=True)
