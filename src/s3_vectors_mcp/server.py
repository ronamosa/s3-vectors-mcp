"""
S3 Vectors MCP Server using FastMCP

This MCP server provides tools for interacting with AWS S3 Vectors service,
allowing users to embed and store vectors, as well as query for similar vectors.

Environment Variables:
    S3VECTORS_BUCKET_NAME: Default S3 bucket name for vector storage
    S3VECTORS_INDEX_NAME: Default vector index name
    S3VECTORS_MODEL_ID: Default Bedrock embedding model ID
    S3VECTORS_DIMENSIONS: Default embedding dimensions for supported models
    S3VECTORS_REGION: Default AWS region
    S3VECTORS_PROFILE: Default AWS profile name

Usage:
    # Using the installed script
    s3-vectors-mcp stdio
    s3-vectors-mcp sse
    s3-vectors-mcp streamable-http

    # Using Python module directly
    python -m mcp_s3vectors stdio
    python -m mcp_s3vectors sse
    python -m mcp_s3vectors streamable-http
"""

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import uuid
from typing import Any, Dict, List, Optional, Tuple

import anyio
import boto3
from mcp.server.fastmcp import Context, FastMCP
from s3vectors.core.services import BedrockService, S3VectorService
from s3vectors.utils.config import get_region
from .config import (
    MissingSettingError,
    Settings,
    settings_from_env_with_overrides,
)
from .ingest import build_vector_payloads, chunk_text, extract_text

# Configure logging
def configure_logging(level_name: Optional[str] = None) -> None:
    resolved = (level_name or os.getenv("S3VECTORS_LOG_LEVEL", "INFO")).upper()
    level = getattr(logging, resolved, logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stderr),
        ],
        force=True,
    )


logger = logging.getLogger(__name__)

mcp = FastMCP("S3 Vectors")

MAX_TOP_K = 1000


_SESSION_CACHE: Dict[Tuple[str, str], boto3.Session] = {}


def get_cached_session(
    profile_name: Optional[str], region: Optional[str]
) -> boto3.Session:
    """Cache boto3 sessions per profile/region combination."""
    key = (profile_name or "", region or "")
    if key not in _SESSION_CACHE:
        session_kwargs = {}
        if profile_name:
            session_kwargs["profile_name"] = profile_name
        if region:
            session_kwargs["region_name"] = region
        _SESSION_CACHE[key] = boto3.Session(**session_kwargs)
    return _SESSION_CACHE[key]


def resolve_settings(**overrides) -> Settings:
    """Resolve settings using overrides with environment fallback."""
    try:
        settings = settings_from_env_with_overrides(**overrides)
    except MissingSettingError as exc:
        raise ValueError(str(exc))

    if not settings.region:
        settings = settings.with_overrides(region=get_region())

    return settings


def validate_text_field(value: str, field_name: str) -> None:
    if not value or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")


def embed_text_with_compat(
    bedrock_service: BedrockService, model_id: str, text: str, dimensions: Optional[int]
) -> List[float]:
    """
    Call the Bedrock embedding helper while remaining compatible with older
    s3vectors-embed-cli releases that might use different method names.
    """
    if hasattr(bedrock_service, "embed_text"):
        return bedrock_service.embed_text(model_id, text, dimensions)
    if hasattr(bedrock_service, "embed_text_sync"):
        # Some older builds exposed a sync-only API
        return bedrock_service.embed_text_sync(model_id, text, dimensions)
    raise ValueError(
        "BedrockService missing embed_text; upgrade s3vectors-embed-cli to >=0.1.1 "
        "(reinstall with `uv tool install --force .`)."
    )


def validate_top_k(value: int) -> None:
    if not 1 <= value <= MAX_TOP_K:
        raise ValueError(f"top_k must be between 1 and {MAX_TOP_K}")


def serialize_filter_expression(
    filter_expr: Optional[Dict[str, Any]]
) -> Optional[str]:
    if filter_expr is None:
        return None
    try:
        return json.dumps(filter_expr)
    except (TypeError, ValueError) as exc:
        raise ValueError("filter_expr must be JSON-serializable") from exc


async def send_info(context: Optional[Context], message: str) -> None:
    if context:
        await context.info(message)


async def send_error(context: Optional[Context], message: str) -> None:
    if context:
        await context.error(message)


async def send_progress(
    context: Optional[Context], progress: float, total: float = 100, message: Optional[str] = None
) -> None:
    if context:
        await context.report_progress(progress, total, message)


async def ensure_index_exists(
    session: boto3.Session,
    settings: Settings,
    create_if_missing: bool,
    context: Optional[Context],
) -> None:
    vector_service = S3VectorService(session, settings.region, debug=False)
    client = session.client("s3vectors")
    try:
        await anyio.to_thread.run_sync(
            lambda: vector_service.get_index(
                settings.bucket_name,
                settings.index_name,
            ),
        )
        return
    except Exception as exc:
        if not create_if_missing:
            raise ValueError(
                f"Index {settings.index_name} not found in {settings.bucket_name}: {exc}"
            ) from exc

    if not settings.dimensions:
        raise ValueError(
            "S3VECTORS_DIMENSIONS must be set to create a new index automatically"
        )

    await send_info(
        context,
        f"Creating index {settings.index_name} (bucket {settings.bucket_name})",
    )
    # AWS API expects lowercase distance metric names (cosine, euclidean).
    # Using uppercase was causing ValidationException.
    await anyio.to_thread.run_sync(
        lambda: client.create_index(
            vectorBucketName=settings.bucket_name,
            indexName=settings.index_name,
            dataType="float32",
            dimension=settings.dimensions,
            distanceMetric="cosine",
        )
    )


async def embed_chunks(
    chunks: List[Any],
    bedrock_service: BedrockService,
    settings: Settings,
    context: Optional[Context],
) -> List[List[float]]:
    embeddings: List[List[float]] = []
    total = len(chunks)
    for idx, chunk in enumerate(chunks):
        await send_info(context, f"Embedding chunk {idx + 1}/{total}")
        await send_progress(
            context,
            10 + (idx / max(total, 1)) * 50,
            message="Generating embeddings",
        )
        embedding = await anyio.to_thread.run_sync(
            embed_text_with_compat,
            bedrock_service,
            settings.model_id,
            chunk.text[:4000],
            settings.dimensions,
        )
        embeddings.append(embedding)
    return embeddings


async def upload_vector_batches(
    vectors: List[dict],
    session: boto3.Session,
    settings: Settings,
    context: Optional[Context],
    batch_size: int = 200,
) -> Dict[str, Any]:
    client = session.client("s3vectors")
    failures = []
    total = len(vectors)
    for idx in range(0, total, batch_size):
        batch = vectors[idx : idx + batch_size]
        await send_info(
            context, f"Uploading batch {idx // batch_size + 1} ({len(batch)} vectors)"
        )
        await send_progress(
            context,
            60 + (idx / max(total, 1)) * 40,
            message="Uploading vectors",
        )
        response = await anyio.to_thread.run_sync(
            lambda: client.put_vectors(
                vectorBucketName=settings.bucket_name,
                indexName=settings.index_name,
                vectors=batch,
            )
        )
        failed = response.get("failedItems") if isinstance(response, dict) else None
        if failed:
            failures.extend(failed)
    return {"uploaded": total, "failed": failures}


@mcp.tool()
async def s3vectors_put(
    text: str,
    vector_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    bucket_name: Optional[str] = None,
    index_name: Optional[str] = None,
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    dimensions: Optional[int] = None,
    context: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Embed text and store as vector in S3 Vectors.

    Uses environment variables for configuration:
    - S3VECTORS_BUCKET_NAME: S3 bucket name for vector storage
    - S3VECTORS_INDEX_NAME: Vector index name
    - S3VECTORS_MODEL_ID: Bedrock embedding model ID
    - S3VECTORS_REGION: AWS region (optional)
    - S3VECTORS_PROFILE: AWS profile name (optional)
    - S3VECTORS_DIMENSIONS: Embedding dimensions (optional)

    Args:
        text: Text to embed and store
        vector_id: Optional vector ID (auto-generated if not provided)
        metadata: Optional metadata to store with the vector
        bucket_name/index_name/model_id: Optional overrides for destination resources
        region/profile/dimensions: Optional overrides for AWS settings
        context: Optional MCP context for streaming updates

    Returns:
        Dict with operation result metadata
    """
    logger.info(
        f"s3vectors_put called with text_length={len(text)}, vector_id={vector_id}, has_metadata={metadata is not None}"
    )

    try:
        validate_text_field(text, "text")

        settings = resolve_settings(
            bucket_name=bucket_name,
            index_name=index_name,
            model_id=model_id,
            region=region,
            profile=profile,
            dimensions=dimensions,
        )

        await send_info(
            context,
            f"Resolved config bucket={settings.bucket_name} index={settings.index_name} model={settings.model_id}",
        )
        await send_progress(context, 10, message="Configuration resolved")

        # Set defaults
        if vector_id is None:
            vector_id = str(uuid.uuid4())
        if metadata is None:
            metadata = {}

        logger.info(f"Using vector_id={vector_id}")
        await send_info(context, f"Using vector_id={vector_id}")

        # Initialize AWS session and services
        logger.info("Initializing AWS session and services...")
        session = get_cached_session(settings.profile, settings.region)
        bedrock_service = BedrockService(session, settings.region, debug=False)
        s3vector_service = S3VectorService(session, settings.region, debug=False)

        # Generate embedding
        await send_info(context, "Generating embedding with Bedrock")
        await send_progress(context, 30, message="Generating embedding")
        embedding = await anyio.to_thread.run_sync(
            embed_text_with_compat,
            bedrock_service,
            settings.model_id,
            text,
            settings.dimensions,
        )
        logger.info(f"Generated embedding with {len(embedding)} dimensions")
        await send_progress(context, 60, message="Embedding generated")

        # Store vector
        await send_info(context, "Storing vector in S3 Vectors")
        result_vector_id = await anyio.to_thread.run_sync(
            s3vector_service.put_vector,
            bucket_name=settings.bucket_name,
            index_name=settings.index_name,
            vector_id=vector_id,
            embedding=embedding,
            metadata=metadata,
        )
        logger.info(f"Successfully stored vector with ID: {result_vector_id}")
        await send_progress(context, 90, message="Vector stored")

        # Prepare result
        result = {
            "success": True,
            "vector_id": result_vector_id,
            "bucket": settings.bucket_name,
            "index": settings.index_name,
            "model_id": settings.model_id,
            "region": settings.region,
            "profile": settings.profile,
            "text_length": len(text),
            "embedding_dimensions": len(embedding),
            "metadata": metadata,
        }

        logger.info("s3vectors_put completed successfully")
        await send_progress(context, 100, message="Completed vector storage")
        await send_info(context, "Vector stored successfully")
        return result

    except Exception as e:
        logger.error(f"Error in s3vectors_put: {str(e)}", exc_info=True)
        await send_error(context, f"s3vectors_put failed: {e}")
        error_result = {"success": False, "error": str(e), "operation": "s3vectors_put"}
        return error_result


@mcp.tool()
async def s3vectors_query(
    query_text: str,
    top_k: int = 10,
    filter_expr: Optional[Dict[str, Any]] = None,
    return_metadata: bool = True,
    return_distance: bool = False,
    bucket_name: Optional[str] = None,
    index_name: Optional[str] = None,
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    dimensions: Optional[int] = None,
    context: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Query for similar vectors in S3 Vectors.

    Uses environment variables for configuration:
    - S3VECTORS_BUCKET_NAME: S3 bucket name for vector storage
    - S3VECTORS_INDEX_NAME: Vector index name
    - S3VECTORS_MODEL_ID: Bedrock embedding model ID
    - S3VECTORS_DIMENSIONS: Embedding dimensions (optional)
    - S3VECTORS_REGION: AWS region (optional)
    - S3VECTORS_PROFILE: AWS profile name (optional)

    Args:
        query_text: Text to query for similar vectors
        top_k: Number of similar vectors to return (default: 10)
        filter_expr: Optional metadata filter expression (JSON format)
        return_metadata: Include metadata in results (default: True)
        return_distance: Include similarity distance in results (default: False)
        bucket_name/index_name/model_id: Optional overrides for S3 resources
        region/profile/dimensions: Optional overrides for AWS settings
        context: Optional MCP context for streaming updates

    Filter Expression (filter_expr):
        Supports AWS S3 Vectors API operators for metadata-based filtering.

        Comparison Operators:
        - $eq: Equal to
        - $ne: Not equal to
        - $gt: Greater than
        - $gte: Greater than or equal to
        - $lt: Less than
        - $lte: Less than or equal to
        - $in: Value in array
        - $nin: Value not in array

        Logical Operators:
        - $and: Logical AND (all conditions must be true)
        - $or: Logical OR (at least one condition must be true)
        - $not: Logical NOT (condition must be false)

        Examples:

        Single condition filters:
        {"category": {"$eq": "documentation"}}
        {"status": {"$ne": "archived"}}
        {"version": {"$gte": "2.0"}}
        {"category": {"$in": ["docs", "guides", "tutorials"]}}

        Multiple condition filters:
        {"$and": [{"category": "tech"}, {"version": "1.0"}]}
        {"$or": [{"category": "docs"}, {"category": "guides"}]}
        {"$not": {"category": {"$eq": "archived"}}}

        Complex nested conditions:
        {"$and": [{"category": "tech"}, {"$or": [{"version": "1.0"}, {"version": "2.0"}]}]}
        {"$and": [{"category": "documentation"}, {"version": {"$gte": "1.0"}}, {"status": {"$ne": "draft"}}]}
        {"$or": [{"$and": [{"category": "docs"}, {"version": "1.0"}]}, {"$and": [{"category": "guides"}, {"version": "2.0"}]}]}

        Notes:
        - String comparisons are case-sensitive
        - Ensure filter values match the data types in your metadata
        - Use proper JSON format with double quotes for keys and string values

    Returns:
        Dict with query results
    """
    logger.info(
        f"s3vectors_query called with query_text_length={len(query_text)}, top_k={top_k}, has_filter={filter_expr is not None}, return_metadata={return_metadata}, return_distance={return_distance}"
    )

    try:
        validate_text_field(query_text, "query_text")
        validate_top_k(top_k)
        settings = resolve_settings(
            bucket_name=bucket_name,
            index_name=index_name,
            model_id=model_id,
            region=region,
            profile=profile,
            dimensions=dimensions,
        )

        await send_info(
            context,
            f"Resolved config bucket={settings.bucket_name} index={settings.index_name} model={settings.model_id}",
        )
        await send_progress(context, 5, message="Configuration resolved")

        # Initialize AWS session and services
        logger.info("Initializing AWS session and services...")
        session = get_cached_session(settings.profile, settings.region)
        bedrock_service = BedrockService(session, settings.region, debug=False)
        s3vector_service = S3VectorService(session, settings.region, debug=False)

        # Generate query embedding
        await send_info(context, "Generating query embedding with Bedrock")
        await send_progress(context, 20, message="Generating query embedding")
        query_embedding = await anyio.to_thread.run_sync(
            embed_text_with_compat,
            bedrock_service,
            settings.model_id,
            query_text,
            settings.dimensions,
        )
        logger.info(f"Generated query embedding with {len(query_embedding)} dimensions")
        await send_progress(context, 40, message="Query embedding generated")

        # Prepare query parameters
        query_params = {
            "bucket_name": settings.bucket_name,
            "index_name": settings.index_name,
            "query_embedding": query_embedding,
            "k": top_k,
            "return_metadata": return_metadata,
            "return_distance": return_distance,
        }

        # Add optional parameters if provided
        filter_expr_str = serialize_filter_expression(filter_expr)
        if filter_expr_str:
            query_params["filter_expr"] = filter_expr_str

        logger.info(f"Query parameters: {query_params}")
        await send_info(context, f"Searching for top {top_k} matches")

        # Perform vector search
        search_results = await anyio.to_thread.run_sync(
            s3vector_service.query_vectors,
            **query_params,
        )
        logger.info(f"Query completed successfully. Raw results: {search_results}")
        await send_progress(context, 80, message="Query completed")

        # Format results
        formatted_results = []
        for result in search_results:
            formatted_result = {
                "vector_id": result.get("vectorId"),
                "similarity": result.get("similarity"),
            }

            if return_metadata and "metadata" in result:
                formatted_result["metadata"] = result.get("metadata", {})

            if return_distance and "similarity" in result:
                formatted_result["distance"] = result.get("similarity")

            formatted_results.append(formatted_result)

        logger.info(f"Formatted {len(formatted_results)} results")
        await send_progress(context, 100, message="Results ready")

        # Prepare response
        response = {
            "success": True,
            "query_text": query_text,
            "bucket": settings.bucket_name,
            "index": settings.index_name,
            "model_id": settings.model_id,
            "region": settings.region,
            "profile": settings.profile,
            "top_k": top_k,
            "filter": filter_expr,
            "return_metadata": return_metadata,
            "return_distance": return_distance,
            "query_embedding_dimensions": len(query_embedding),
            "results_count": len(formatted_results),
            "results": formatted_results,
        }

        logger.info("s3vectors_query completed successfully")
        return response

    except Exception as e:
        logger.error(f"Error in s3vectors_query: {str(e)}", exc_info=True)
        await send_error(context, f"s3vectors_query failed: {e}")
        error_result = {
            "success": False,
            "error": str(e),
            "operation": "s3vectors_query",
        }
        return error_result


@mcp.tool()
async def s3vectors_ingest_pdf(
    pdf_path: str,
    topic: str,
    vault_root: Optional[str] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    bucket_name: Optional[str] = None,
    index_name: Optional[str] = None,
    model_id: Optional[str] = None,
    region: Optional[str] = None,
    profile: Optional[str] = None,
    dimensions: Optional[int] = None,
    create_index: bool = False,
    batch_size: int = 200,
    context: Optional[Context] = None,
) -> Dict[str, Any]:
    """
    Ingest a local PDF into S3 Vectors by extracting, chunking, embedding, and uploading.
    """

    pdf = Path(pdf_path).expanduser().resolve()
    if not pdf.exists() or pdf.suffix.lower() != ".pdf":
        raise ValueError(f"PDF not found: {pdf}")

    settings = resolve_settings(
        bucket_name=bucket_name,
        index_name=index_name,
        model_id=model_id,
        region=region,
        profile=profile,
        dimensions=dimensions,
    )

    await send_info(
        context,
        f"Ingesting {pdf.name} into {settings.bucket_name}/{settings.index_name}",
    )

    text = await anyio.to_thread.run_sync(extract_text, pdf)
    if not text.strip():
        raise ValueError(f"No text extracted from {pdf}")

    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise ValueError("PDF produced zero chunks; adjust chunk_size/overlap")

    await send_info(context, f"Chunked into {len(chunks)} segments")

    session = get_cached_session(settings.profile, settings.region)
    bedrock_service = BedrockService(session, settings.region, debug=False)

    await ensure_index_exists(session, settings, create_index, context)

    embeddings = await embed_chunks(chunks, bedrock_service, settings, context)

    root_path = Path(vault_root).expanduser().resolve() if vault_root else pdf.parent
    vectors = build_vector_payloads(pdf, topic, chunks, embeddings, root_path)

    upload_summary = await upload_vector_batches(
        vectors, session, settings, context, batch_size=batch_size
    )

    failures = upload_summary.get("failed") or []
    success = len(failures) == 0

    return {
        "success": success,
        "pdf": str(pdf),
        "topic": topic,
        "bucket": settings.bucket_name,
        "index": settings.index_name,
        "chunks": len(chunks),
        "uploaded": upload_summary["uploaded"],
        "failed": failures,
    }


def serve(transport: str = "stdio", log_level: Optional[str] = None) -> None:
    """Start the S3 Vectors MCP server with the specified transport."""
    configure_logging(log_level)
    logger.info("Starting S3 Vectors MCP Server...")
    logger.info(f"Using transport: {transport}")

    # Log environment variables for debugging
    env_vars = [
        "S3VECTORS_BUCKET_NAME",
        "S3VECTORS_INDEX_NAME",
        "S3VECTORS_MODEL_ID",
        "S3VECTORS_DIMENSIONS",
        "S3VECTORS_REGION",
        "S3VECTORS_PROFILE",
    ]
    for var in env_vars:
        value = os.getenv(var)
        if value:
            logger.info(f"Environment variable {var}={value}")
        else:
            logger.info(f"Environment variable {var} not set")

    # Run the FastMCP server with the selected transport
    logger.info("Starting FastMCP server...")
    mcp.run(transport=transport)


def main() -> None:
    """Main entry point with transport selection."""
    parser = argparse.ArgumentParser(
        description="S3 Vectors MCP Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  s3-vectors-mcp stdio                    # Use stdio transport (default)
  s3-vectors-mcp sse                      # Use Server-Sent Events transport
  s3-vectors-mcp streamable-http          # Use StreamableHTTP transport
        """,
    )

    parser.add_argument(
        "transport",
        choices=["stdio", "sse", "streamable-http"],
        default="stdio",
        nargs="?",
        help="Transport type to use (default: stdio)",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override server log level",
    )

    args = parser.parse_args()
    serve(args.transport, args.log_level)


if __name__ == "__main__":
    main()
