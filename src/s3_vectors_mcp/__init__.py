"""S3 Vectors MCP Server

A Model Context Protocol server providing tools to interact with AWS S3 Vectors service
for embedding and querying vector data using Amazon Bedrock models.
"""

from .server import serve

__version__ = "0.1.0"
__all__ = ["serve"]
