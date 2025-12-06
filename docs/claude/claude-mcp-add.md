## Adding the S3 Vectors MCP server to Claude CLI

1. Export your AWS/S3 Vectors configuration (or set them in your shell profile):

```bash
export S3VECTORS_BUCKET_NAME="my-vectors-bucket"
export S3VECTORS_INDEX_NAME="my-index"
export S3VECTORS_MODEL_ID="amazon.titan-embed-text-v2:0"
export S3VECTORS_REGION="us-east-1"
export S3VECTORS_DIMENSIONS="1024"
```

2. Register the MCP server with Claude CLI using `claude mcp add`. The `--` separates the Claude arguments from the command used to launch the MCP server (our CLI):

```bash
claude mcp add s3vectors -- uv run s3-vectors-mcp serve --transport stdio
```

- Use `-s user` if you want the server available for every Claude project on the machine:

```bash
claude mcp add s3vectors -s user -- uv run s3-vectors-mcp serve --transport stdio
```

3. Verify the registration and test connectivity:

```bash
claude mcp list
claude mcp info s3vectors
claude mcp call s3vectors s3vectors_query -- --help
claude mcp call s3vectors s3vectors_ingest_pdf -- --help
```

4. Start Claude CLI (`claude`) or Claude Code and run `/mcp` to load the new tools. The server command will reuse the environment variables you exported in step 1.

