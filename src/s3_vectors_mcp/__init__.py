"""S3 Vectors MCP Server

A Model Context Protocol server providing tools to interact with AWS S3 Vectors service
for embedding and querying vector data using Amazon Bedrock models.
"""

from __future__ import annotations

import importlib.metadata


def get_version() -> str:
    """Single source of truth for package version."""
    try:
        return importlib.metadata.version("s3-vectors-mcp")
    except importlib.metadata.PackageNotFoundError:
        return "0.0.0"


__version__ = get_version()

from .server import serve  # noqa: E402

__all__ = ["serve", "__version__", "get_version"]
