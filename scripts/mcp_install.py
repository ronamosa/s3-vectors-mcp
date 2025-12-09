#!/usr/bin/env python3
"""
Bootstrap installer for the S3 Vectors MCP server.

This script ensures the CLI entry point is available, then writes a small
wrapper script plus environment templates under ~/.mcp/servers/<name>/.
The wrapper script is intended to be referenced from Claude Code/Desktop
or any other MCP-compatible client so configuration stays in one place.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER_NAME = "s3vectors"
DEFAULT_TIMEOUT_MS = 120_000
DEFAULT_TRANSPORT = "stdio"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install helper that prepares ~/.mcp/servers/s3vectors",
    )
    parser.add_argument(
        "--install-dir",
        type=Path,
        default=Path.home() / ".mcp" / "servers" / DEFAULT_SERVER_NAME,
        help="Where to write wrapper/env files (default: ~/.mcp/servers/s3vectors)",
    )
    parser.add_argument(
        "--transport",
        default=DEFAULT_TRANSPORT,
        help="Transport passed to the server wrapper (default: stdio)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_MS,
        help="Timeout in milliseconds to include in generated Claude snippet",
    )
    parser.add_argument(
        "--server-name",
        default=DEFAULT_SERVER_NAME,
        help="Name used when generating Claude config snippets",
    )
    return parser.parse_args()


def ensure_cli_binary() -> Path:
    """
    Ensure the CLI entry point exists and return its absolute path.
    """
    candidates = [
        shutil.which("s3-vectors-mcp"),
        Path.home() / ".local" / "bin" / "s3-vectors-mcp",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser().resolve()
        if candidate_path.exists():
            return candidate_path

    raise SystemExit(
        "Could not locate the 's3-vectors-mcp' executable. "
        "Run 'uv tool install --force --python 3.11 .' first."
    )


def write_env_files(install_dir: Path) -> None:
    """
    Create .env.example and .env (if missing) for MCP configuration.
    """
    env_example_content = "\n".join(
        [
            "# S3 Vectors MCP configuration",
            "S3VECTORS_BUCKET_NAME=",
            "S3VECTORS_INDEX_NAME=",
            "S3VECTORS_MODEL_ID=amazon.titan-embed-text-v2:0",
            "S3VECTORS_REGION=us-east-1",
            "S3VECTORS_DIMENSIONS=1024",
            "S3VECTORS_PROFILE=",
            "S3VECTORS_LOG_LEVEL=INFO",
            "",
            "# Optional override for the transport used by the wrapper script",
            "S3VECTORS_TRANSPORT=stdio",
            "",
        ]
    )
    env_example = install_dir / ".env.example"
    env_example.write_text(env_example_content, encoding="utf-8")

    env_file = install_dir / ".env"
    if not env_file.exists():
        env_file.write_text(env_example_content, encoding="utf-8")


def write_wrapper_script(
    install_dir: Path,
    cli_path: Path,
    transport: str,
) -> Path:
    """
    Write a small shell wrapper that sources .env then launches the CLI.
    """
    wrapper_path = install_dir / "serve.sh"
    cli_quoted = shlex.quote(str(cli_path))
    transport_quoted = shlex.quote(str(transport))
    wrapper_template = f"""#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
ENV_FILE="${{SCRIPT_DIR}}/.env"

# Source .env defaults, but do not overwrite existing environment variables.
# This allows Claude config (claude.json) to take precedence over the .env file.
if [ -f "${{ENV_FILE}}" ]; then
  while IFS='=' read -r key value || [ -n "$key" ]; do
    # Skip comments and empty lines
    [[ $key =~ ^#.*$ ]] || [ -z "$key" ] && continue
    
    # If variable is NOT set in current environment, export it from .env
    if [ -z "${{!key+x}}" ]; then
       # Remove potential surrounding quotes from value
       value="${{value%\\"}}"
       value="${{value#\\"}}"
       value="${{value%\\'}}"
       value="${{value#\\'}}"
       export "$key=$value"
    fi
  done < "${{ENV_FILE}}"
fi

TRANSPORT="${{S3VECTORS_TRANSPORT:-{transport_quoted}}}"
exec {cli_quoted} serve --transport "${{TRANSPORT}}" "$@"
"""
    wrapper_path.write_text(wrapper_template, encoding="utf-8")
    wrapper_path.chmod(wrapper_path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return wrapper_path


def write_claude_snippet(
    install_dir: Path,
    server_name: str,
    wrapper_path: Path,
    timeout: int,
) -> Path:
    """
    Generate a ready-to-use Claude Code/Desktop JSON snippet that points to the wrapper.
    """
    snippet = {
        "mcpServers": {
            server_name: {
                "command": str(wrapper_path),
                "args": [],
                "env": {},
                "timeout": timeout,
                "disabled": False,
            }
        }
    }
    snippet_path = install_dir / "claude-code.json"
    snippet_path.write_text(json.dumps(snippet, indent=2), encoding="utf-8")
    return snippet_path


def main() -> None:
    args = parse_args()
    install_dir: Path = args.install_dir.expanduser()
    install_dir.mkdir(parents=True, exist_ok=True)

    cli_path = ensure_cli_binary()
    wrapper_path = write_wrapper_script(install_dir, cli_path, args.transport)
    write_env_files(install_dir)
    snippet_path = write_claude_snippet(
        install_dir,
        args.server_name,
        wrapper_path,
        args.timeout,
    )

    print(f"✔ Wrapper script written to {wrapper_path}")
    print(f"✔ Environment templates available at {install_dir}")
    print(f"✔ Claude config snippet saved to {snippet_path}")
    print()
    print("Next steps:")
    print(f"  1. Edit {install_dir / '.env'} with your AWS + S3 vectors settings.")
    print("     (Or leave .env empty and configure via Claude's config file if preferred)")
    print(
        "  2. In Claude Code/Desktop, point the MCP server command at "
        f"{wrapper_path} (see claude-code.json for a ready-to-copy block)."
    )


if __name__ == "__main__":
    main()
