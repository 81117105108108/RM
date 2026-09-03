#!/usr/bin/env python3
"""
repomap.py - Production CLI for the RepoMapper engine.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys
from pathlib import Path
from typing import List, Set

from repomap_class import RepoMap


class GitIgnoreFilter:
    """Fast hierarchical matcher enforcing standard .gitignore exclusion rules."""

    def __init__(self, root: Path):
        self.root = root.resolve()
        self.ignore_dirs: Set[str] = {
            ".git", "node_modules", "__pycache__", "venv", "env",
            ".venv", "dist", "build", "target", ".cache", ".repomap*"
        }
        self.patterns: List[Tuple[str, bool]] = []
        self._load_rules()

    def _load_rules(self) -> None:
        gi_file = self.root / ".gitignore"
        if gi_file.is_file():
            try:
                for line in gi_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if line and not line.startswith("#"):
                        is_negation = line.startswith("!")
                        if is_negation:
                            line = line[1:]
                        if line.endswith("/"):
                            self.ignore_dirs.add(line.rstrip("/"))
                        else:
                            self.patterns.append((line, is_negation))
            except OSError:
                pass

    def is_ignored(self, path: Path) -> bool:
        try:
            rel = path.resolve().relative_to(self.root).as_posix()
        except ValueError:
            return False

        # Directory part matching
        parts = path.relative_to(self.root).parts
        for part in parts[:-1]:
            if part in self.ignore_dirs:
                return True

        if path.is_dir() and path.name in self.ignore_dirs:
            return True

        ignored = False
        for pat, is_neg in self.patterns:
            if fnmatch.fnmatch(rel, pat) or fnmatch.fnmatch(path.name, pat):
                ignored = not is_neg
        return ignored


def find_src_files(directory: str) -> List[str]:
    """Recursively discovers eligible source files, skipping ignored trees and binaries."""
    root_path = Path(directory).resolve()
    gi_filter = GitIgnoreFilter(root_path)
    discovered = []

    for dirpath, dirnames, filenames in os.walk(root_path):
        current_dir = Path(dirpath)
        # Prune ignored subtrees from in-place os.walk traversal
        dirnames[:] = [d for d in dirnames if not gi_filter.is_ignored(current_dir / d)]

        for f in filenames:
            file_path = current_dir / f
            if not f.startswith(".") and not gi_filter.is_ignored(file_path):
                discovered.append(str(file_path))

    return discovered


def main() -> int:
    parser = argparse.ArgumentParser(
        description="RepoMapper: AST-guided repository context mapping for AI coding agents."
    )
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directory trees to index.")
    parser.add_argument("--root", default=".", help="Repository root boundary.")
    parser.add_argument("--map-tokens", type=int, default=8192, help="Token ceiling for map output.")
    parser.add_argument("--chat-files", nargs="*", default=[], help="Actively modified or focused files.")
    parser.add_argument("--other-files", nargs="*", default=[], help="Secondary contextual scope files.")
    parser.add_argument("--outline", type=str, default=None, help="Generate structural outline for a single file.")
    parser.add_argument("--blast-radius", type=str, default=None, help="Compute impact radius for a modified file.")
    parser.add_argument("--json", action="store_true", help="Format structured output as JSON envelope.")
    parser.add_argument("--verbose", action="store_true", help="Emit diagnostic telemetry to stderr.")

    args = parser.parse_args()
    root_dir = Path(args.root).resolve()

    # Diagnostics dispatched strictly to stderr
    if args.verbose:
        print(f"[RepoMapper] Root: {root_dir}", file=sys.stderr)
        print(f"[RepoMapper] Chat files: {args.chat_files}", file=sys.stderr)

    mapper = RepoMap(
        map_tokens=args.map_tokens,
        root=str(root_dir),
        verbose=args.verbose
    )

    # Sub-command: File Outline
    if args.outline:
        outline = mapper.get_file_outline(args.outline)
        if args.json:
            sys.stdout.write(json.dumps({"outline": outline}) + "\n")
        else:
            sys.stdout.write(outline + "\n")
        return 0

    # Sub-command: Blast Radius
    if args.blast_radius:
        impact = mapper.compute_blast_radius(args.blast_radius)
        sys.stdout.write(json.dumps(impact, indent=2) + "\n")
        return 0

    # Discover candidate files
    candidates = []
    for p in args.paths:
        target = Path(p).resolve()
        if target.is_file():
            candidates.append(str(target))
        elif target.is_dir():
            candidates.extend(find_src_files(str(target)))

    all_other_files = sorted(list(set(candidates + [str(Path(f).resolve()) for f in args.other_files])))
    resolved_chat = [str(Path(f).resolve()) for f in args.chat_files]

    repo_map = mapper.get_repo_map(
        chat_files=resolved_chat,
        other_files=all_other_files
    )

    # Standard data output to stdout
    if args.json:
        sys.stdout.write(json.dumps({"repo_map": repo_map}) + "\n")
    else:
        sys.stdout.write(repo_map + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
