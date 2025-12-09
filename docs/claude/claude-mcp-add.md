## Adding the S3 Vectors MCP server to Claude CLI

### Fast path (recommended)

1. Install the server assets and wrapper script:

   ```bash
   git clone https://github.com/ronamosa/s3-vectors-mcp.git
   cd s3-vectors-mcp
   make install
   ```

   This writes `~/.mcp/servers/s3vectors/serve.sh` plus `.env` / `.env.example`.

2. Edit `~/.mcp/servers/s3vectors/.env` with your AWS + S3 Vectors settings.

3. Register the wrapper with Claude CLI (add `-s user` for a global install):

   ```bash
   claude mcp add s3vectors -s user -- ~/.mcp/servers/s3vectors/serve.sh
   ```

### Manual path (fallback)

If you prefer exporting environment variables yourself, you can still register
the CLI directly:

```bash
export S3VECTORS_BUCKET_NAME="my-vectors-bucket"
export S3VECTORS_INDEX_NAME="my-index"
export S3VECTORS_MODEL_ID="amazon.titan-embed-text-v2:0"
export S3VECTORS_REGION="us-east-1"
export S3VECTORS_DIMENSIONS="1024"

claude mcp add s3vectors -- uv run s3-vectors-mcp serve --transport stdio
```

- Use `-s user` if you want the server available for every Claude project on
  the machine:

  ```bash
  claude mcp add s3vectors -s user -- uv run s3-vectors-mcp serve --transport stdio
  ```

4. Verify the registration and test connectivity:

```bash
claude mcp list
claude mcp info s3vectors
claude mcp call s3vectors s3vectors_query -- --help
claude mcp call s3vectors s3vectors_ingest_pdf -- --help
```

4. Start Claude CLI (`claude`) or Claude Code and run `/mcp` to load the new tools. The server command will reuse the environment variables you exported in step 1.

