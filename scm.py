"""
scm.py - Mapping and path resolution for Tree-sitter Scheme (.scm) query files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

SCM_FILE_REGISTRY = {
    "c": "c-tags.scm",
    "cpp": "cpp-tags.scm",
    "csharp": "c_sharp-tags.scm",
    "c_sharp": "c_sharp-tags.scm",
    "dart": "dart-tags.scm",
    "elixir": "elixir-tags.scm",
    "elm": "elm-tags.scm",
    "go": "go-tags.scm",
    "java": "java-tags.scm",
    "javascript": "javascript-tags.scm",
    "js": "javascript-tags.scm",
    "jsx": "javascript-tags.scm",
    "kotlin": "kotlin-tags.scm",
    "lua": "lua-tags.scm",
    "php": "php-tags.scm",
    "python": "python-tags.scm",
    "py": "python-tags.scm",
    "ruby": "ruby-tags.scm",
    "rust": "rust-tags.scm",
    "rs": "rust-tags.scm",
    "scala": "scala-tags.scm",
    "solidity": "solidity-tags.scm",
    "swift": "swift-tags.scm",
    "typescript": "typescript-tags.scm",
    "ts": "typescript-tags.scm",
    "tsx": "typescript-tags.scm",
}


def get_scm_fname(lang: str) -> Optional[str]:
    """
    Locates the matching .scm query file within packaged directories.
    """
    if not lang:
        return None

    normalized_lang = lang.lower().strip()
    scm_filename = SCM_FILE_REGISTRY.get(normalized_lang)
    if not scm_filename:
        scm_filename = f"{normalized_lang}-tags.scm"

    base_dir = Path(__file__).parent.resolve()
    candidate_locations = [
        base_dir / "queries" / "tree-sitter-language-pack" / scm_filename,
        base_dir / "queries" / "tree-sitter-languages" / scm_filename,
        base_dir / "queries" / scm_filename,
    ]

    for candidate in candidate_locations:
        if candidate.is_file():
            return str(candidate)

    return None
