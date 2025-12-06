"""CLI entry point for the Claude-first S3 Vectors MCP fork."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Optional

import anyio

from . import __version__
from .server import serve, s3vectors_ingest_pdf, s3vectors_put, s3vectors_query

CONFIG_ARGS = [
    ("bucket_name", "S3VECTORS_BUCKET_NAME", "S3 bucket containing vector data"),
    ("index_name", "S3VECTORS_INDEX_NAME", "S3 Vectors index name"),
    ("model_id", "S3VECTORS_MODEL_ID", "Bedrock embedding model ID"),
    ("region", "S3VECTORS_REGION", "AWS region"),
    ("profile", "S3VECTORS_PROFILE", "AWS profile name"),
    ("dimensions", "S3VECTORS_DIMENSIONS", "Embedding dimensions"),
]


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Attach shared configuration arguments to a parser."""
    for arg_name, _, help_text in CONFIG_ARGS:
        flag = f"--{arg_name.replace('_', '-')}"
        kwargs: Dict[str, Any] = {"dest": arg_name, "help": help_text}
        if arg_name == "dimensions":
            kwargs["type"] = int
        parser.add_argument(flag, **kwargs)


def update_env_from_args(args: argparse.Namespace) -> None:
    """Propagate CLI overrides to environment variables for server use."""
    for attr, env_var, _ in CONFIG_ARGS:
        value = getattr(args, attr, None)
        if value is not None:
            os.environ[env_var] = str(value)


def parse_json_arg(value: Optional[str], arg_name: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid JSON for --{arg_name}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit(f"--{arg_name} must be a JSON object")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="s3-vectors-mcp",
        description="Claude Code/CLI-focused MCP server for Amazon S3 Vectors",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print version and exit",
    )
    subparsers = parser.add_subparsers(dest="command")

    # serve command
    serve_parser = subparsers.add_parser("serve", help="Run the MCP server")
    serve_parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        help="Transport to use for MCP server",
    )
    serve_parser.add_argument(
        "--log-level",
        dest="log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override server log level",
    )
    add_config_arguments(serve_parser)

    # put command
    put_parser = subparsers.add_parser("put", help="Embed text and store as vector")
    put_parser.add_argument("--text", required=True, help="Text to embed and store")
    put_parser.add_argument("--vector-id", help="Optional explicit vector ID")
    put_parser.add_argument(
        "--metadata",
        help="JSON object containing metadata to store alongside the vector",
    )
    add_config_arguments(put_parser)

    # query command
    query_parser = subparsers.add_parser("query", help="Query similar vectors")
    query_parser.add_argument("--query-text", required=True, help="Search text")
    query_parser.add_argument(
        "--top-k", type=int, default=10, help="Number of similar vectors to return"
    )
    query_parser.add_argument(
        "--filter",
        help="JSON object describing filter expression for metadata search",
    )
    query_parser.add_argument(
        "--no-metadata",
        action="store_false",
        dest="return_metadata",
        help="Skip metadata in query response",
    )
    query_parser.add_argument(
        "--return-distance",
        action="store_true",
        help="Include similarity distance in response",
    )
    add_config_arguments(query_parser)

    # ingest command
    ingest_parser = subparsers.add_parser(
        "ingest-pdf", help="Extract, embed, and upload a local PDF"
    )
    ingest_parser.add_argument("--pdf-path", required=True, help="Path to the PDF")
    ingest_parser.add_argument("--topic", required=True, help="Topic label for metadata")
    ingest_parser.add_argument(
        "--vault-root",
        help="Root folder for relative metadata paths (defaults to PDF directory)",
    )
    ingest_parser.add_argument(
        "--chunk-size", type=int, default=500, help="Words per chunk (default 500)"
    )
    ingest_parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=50,
        help="Word overlap between chunks (default 50)",
    )
    ingest_parser.add_argument(
        "--create-index",
        action="store_true",
        help="Create the index if it does not exist",
    )
    ingest_parser.add_argument(
        "--batch-size",
        type=int,
        default=200,
        help="Number of vectors per upload batch",
    )
    add_config_arguments(ingest_parser)

    return parser


async def run_put(args: argparse.Namespace) -> int:
    metadata = parse_json_arg(args.metadata, "metadata")
    result = await s3vectors_put(
        text=args.text,
        vector_id=args.vector_id,
        metadata=metadata,
        bucket_name=args.bucket_name,
        index_name=args.index_name,
        model_id=args.model_id,
        region=args.region,
        profile=args.profile,
        dimensions=args.dimensions,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


async def run_query(args: argparse.Namespace) -> int:
    filter_expr = parse_json_arg(args.filter, "filter")
    result = await s3vectors_query(
        query_text=args.query_text,
        top_k=args.top_k,
        filter_expr=filter_expr,
        return_metadata=args.return_metadata if args.return_metadata is not None else True,
        return_distance=args.return_distance,
        bucket_name=args.bucket_name,
        index_name=args.index_name,
        model_id=args.model_id,
        region=args.region,
        profile=args.profile,
        dimensions=args.dimensions,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


async def run_ingest(args: argparse.Namespace) -> int:
    result = await s3vectors_ingest_pdf(
        pdf_path=args.pdf_path,
        topic=args.topic,
        vault_root=args.vault_root,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        bucket_name=args.bucket_name,
        index_name=args.index_name,
        model_id=args.model_id,
        region=args.region,
        profile=args.profile,
        dimensions=args.dimensions,
        create_index=args.create_index,
        batch_size=args.batch_size,
    )
    print(json.dumps(result, indent=2))
    return 0 if result.get("success") else 1


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print(__version__)
        return 0

    if not getattr(args, "command", None):
        parser.error("command is required")

    if args.command == "serve":
        update_env_from_args(args)
        if getattr(args, "log_level", None):
            os.environ["S3VECTORS_LOG_LEVEL"] = args.log_level
        serve(args.transport, args.log_level)
        return 0

    if args.command == "put":
        return anyio.run(run_put, args)

    if args.command == "query":
        return anyio.run(run_query, args)

    if args.command == "ingest-pdf":
        return anyio.run(run_ingest, args)

    parser.error("Unknown command")
    return 1


if __name__ == "__main__":
    sys.exit(main())

