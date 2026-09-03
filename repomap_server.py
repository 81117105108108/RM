#!/usr/bin/env python3
"""
repomap_server.py - Model Context Protocol (MCP) server for RepoMapper.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from repomap import find_src_files
from repomap_class import RepoMap

app = Server("repomap-server")


@app.list_tools()
async def list_tools() -> List[Tool]:
    """Exposes 4 specialized code navigation and mapping tools to MCP clients."""
    return [
        Tool(
            name="repo_map",
            description="Generates an AST-aware structural skeleton of a repository within a token budget.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {
                        "type": "string",
                        "description": "Absolute path to the target codebase root."
                    },
                    "token_limit": {
                        "type": "integer",
                        "default": 2048,
                        "description": "Maximum token budget allocated for the map."
                    },
                    "chat_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files actively under edit (elevates graph ranking)."
                    }
                },
                "required": ["project_root"]
            }
        ),
        Tool(
            name="file_outline",
            description="Generates the structural definition outline of a specific file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {"type": "string"},
                    "file_path": {"type": "string", "description": "Relative path to target file."}
                },
                "required": ["project_root", "file_path"]
            }
        ),
        Tool(
            name="locate_symbol",
            description="Locates declaration sites and references for a specific identifier.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {"type": "string"},
                    "identifier": {"type": "string", "description": "Exact symbol name."}
                },
                "required": ["project_root", "identifier"]
            }
        ),
        Tool(
            name="blast_radius",
            description="Computes dependent files affected by changes to a target file.",
            inputSchema={
                "type": "object",
                "properties": {
                    "project_root": {"type": "string"},
                    "file_path": {"type": "string", "description": "Target modified file."}
                },
                "required": ["project_root", "file_path"]
            }
        )
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
    """Handles tool invocations with isolated exception trapping."""
    try:
        root_path = Path(arguments["project_root"]).resolve()
        if not root_path.is_dir():
            return [TextContent(type="text", text=f"Error: Invalid directory '{root_path}'")]

        mapper = RepoMap(root=str(root_path), verbose=False)

        if name == "repo_map":
            token_limit = int(arguments.get("token_limit", 2048))
            chat_files = arguments.get("chat_files", [])
            all_files = find_src_files(str(root_path))
            mapper.map_tokens = token_limit
            result = mapper.get_repo_map(chat_files=chat_files, other_files=all_files)
            return [TextContent(type="text", text=result)]

        elif name == "file_outline":
            outline = mapper.get_file_outline(arguments["file_path"])
            return [TextContent(type="text", text=outline)]

        elif name == "locate_symbol":
            import json
            matches = mapper.locate_symbol(arguments["identifier"])
            return [TextContent(type="text", text=json.dumps(matches, indent=2))]

        elif name == "blast_radius":
            import json
            radius = mapper.compute_blast_radius(arguments["file_path"])
            return [TextContent(type="text", text=json.dumps(radius, indent=2))]

        return [TextContent(type="text", text=f"Error: Unknown tool '{name}'")]

    except Exception as e:
        # Standard error capture preventing stdio protocol crash
        return [TextContent(type="text", text=f"Internal tool execution error: {str(e)}")]


async def run() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":
    main()
