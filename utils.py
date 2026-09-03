"""
utils.py - Unified primitives, token counters, and low-level I/O operations.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:
    import tiktoken
except ImportError:
    tiktoken = None  # type: ignore


@dataclass(slots=True, frozen=True)
class Tag:
    """Represents a code symbol definition or reference extracted via AST."""
    rel_fname: str
    fname: str
    line: int
    name: str
    kind: str  # 'def' | 'ref'
    category: str = "symbol"  # 'class' | 'function' | 'method' | 'variable'


def is_binary_file(filepath: Path | str) -> bool:
    """
    Determines whether a file contains binary content by sniffing the first 1024 bytes
    for null-byte sequences.
    """
    path = Path(filepath)
    if not path.is_file():
        return False
    try:
        with open(path, "rb") as f:
            chunk = f.read(1024)
            return b"\x00" in chunk
    except OSError:
        return True


def compute_file_hash(filepath: Path | str) -> str:
    """Computes a SHA-256 fingerprint of the target file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


_TOKEN_ENCODERS: dict[str, Any] = {}


def count_tokens(text: str, model_name: str = "gpt-4") -> int:
    """
    Calculates exact BPE token length using tiktoken with fallback estimation.
    """
    if not text:
        return 0

    if tiktoken is not None:
        global _TOKEN_ENCODERS
        if model_name not in _TOKEN_ENCODERS:
            try:
                _TOKEN_ENCODERS[model_name] = tiktoken.encoding_for_model(model_name)
            except KeyError:
                _TOKEN_ENCODERS[model_name] = tiktoken.get_encoding("cl100k_base")
        return len(_TOKEN_ENCODERS[model_name].encode(text, disallowed_special=()))

    # Fallback heuristic: 1 token ~= 4 characters for standard code
    return max(1, len(text) // 4)


def read_text(
    filename: str | Path,
    encoding: str = "utf-8",
    max_bytes: int = 2_000_000,
    silent: bool = False
) -> Optional[str]:
    """
    Safely reads text files, handling binary exclusions, encoding anomalies,
    and size boundaries.
    """
    path = Path(filename)
    if not path.is_file() or is_binary_file(path):
        return None

    try:
        if path.stat().st_size > max_bytes:
            if not silent:
                import sys
                print(f"Warning: Skipping oversized file {path} (> {max_bytes} bytes)", file=sys.stderr)
            return None

        return path.read_text(encoding=encoding, errors="replace")
    except (OSError, UnicodeError) as e:
        if not silent:
            import sys
            print(f"Warning: Failed to read {path}: {e}", file=sys.stderr)
        return None
