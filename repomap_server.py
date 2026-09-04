#!/usr/bin/env python3
"""
repomap_server.py - Model Context Protocol (MCP) server for RepoMapper.
FastMCP port (mcp>=2 removed the legacy Server.list_tools/call_tool decorators).
Tools: repo_map, file_outline, locate_symbol, blast_radius.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from fastmcp import FastMCP

from repomap import find_src_files
from repomap_class import RepoMap

mcp = FastMCP("repomap-server")


def _mapper(project_root: str) -> RepoMap:
    root_path = Path(project_root).resolve()
    if not root_path.is_dir():
        raise ValueError(f"Invalid directory '{root_path}'")
    return RepoMap(root=str(root_path), verbose=False)


@mcp.tool()
def repo_map(
    project_root: str,
    token_limit: int = 2048,
    chat_files: Optional[List[str]] = None,
) -> str:
    """Generates an AST-aware structural skeleton of a repository within a token budget."""
    try:
        mapper = _mapper(project_root)
        mapper.map_tokens = token_limit
        all_files = find_src_files(str(Path(project_root).resolve()))
        return mapper.get_repo_map(
            chat_files=chat_files or [], other_files=all_files
        )
    except Exception as e:
        return f"Internal tool execution error: {e}"


@mcp.tool()
def file_outline(project_root: str, file_path: str) -> str:
    """Generates the structural definition outline of a specific file."""
    try:
        return _mapper(project_root).get_file_outline(file_path)
    except Exception as e:
        return f"Internal tool execution error: {e}"


@mcp.tool()
def locate_symbol(project_root: str, identifier: str) -> str:
    """Locates declaration sites and references for a specific identifier."""
    try:
        matches = _mapper(project_root).locate_symbol(identifier)
        return json.dumps(matches, indent=2)
    except Exception as e:
        return f"Internal tool execution error: {e}"


@mcp.tool()
def blast_radius(project_root: str, file_path: str) -> str:
    """Computes dependent files affected by changes to a target file."""
    try:
        radius = _mapper(project_root).compute_blast_radius(file_path)
        return json.dumps(radius, indent=2)
    except Exception as e:
        return f"Internal tool execution error: {e}"


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
