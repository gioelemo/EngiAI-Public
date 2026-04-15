# Prusa 3D Printer Integration

The Prusa Agent enables control of Prusa 3D printers through Prusa Connect, allowing you to print designs directly from the EngiAI.

## Overview

The Prusa Agent communicates with Prusa Connect via an MCP (Model Context Protocol) server, providing tools for:
- Listing available printers
- Uploading STL/3MF files
- Starting print jobs
- Monitoring printer status
- Managing print files

## Prerequisites

- Prusa printer linked to a Prusa Connect account
- A web browser on the host machine (used only for the one-time login)
- MCP server — installed as a pip dependency from git
  (`prusa-mcp @ git+https://github.com/gioelemo/prusa-mcp.git`), declared in
  `services/prusa_mcp_server/requirements-mcp.txt`. No submodule
  initialization is required; `services/prusa_mcp_server/run.sh` and the
  `prusa-mcp-server` Docker image install it automatically.

## Setup

### 1. Authenticate with Prusa Account (one-time, on the host)

`prusa-mcp` authenticates to Prusa Connect via **OAuth2 Authorization Code + PKCE**
against `account.prusa3d.com` — the same public client that PrusaSlicer uses.
There are no credentials in `.env`: you log in once through your browser and the
server persists refresh tokens to a file on a mounted volume.

From the repo root, run:

```bash
make prusa-login
```

This opens `account.prusa3d.com` in your default browser, you sign in (SSO, 2FA,
passkeys — whatever you normally use), and the login helper receives the OAuth
callback on `127.0.0.1`. The resulting tokens are written to
`./data/prusa_tokens.json` (mode `0600`). That file is bind-mounted into the
`prusa-mcp-server` container at `/app/data/prusa_tokens.json` and the server
refreshes access tokens in place from then on.

You only need to re-run `make prusa-login` if you revoke the app in your Prusa
Account dashboard or delete the token file.

### 2. Docker Deployment (Recommended)

Once tokens exist on disk, start the stack:

```bash
docker-compose up -d
```

This starts:
- **Chatbot container**: Main application
- **PostgreSQL container**: Database
- **Prusa MCP Server**: Prusa Connect integration (reads tokens from `./data/prusa_tokens.json`)

### 3. Configuration

The chatbot only needs to know where the MCP server lives:

```bash
# MCP Configuration
SKIP_MCP=false
PRUSA_MCP_URL=http://prusa-mcp-server:8765  # For Docker
# PRUSA_MCP_URL=http://localhost:8765        # For local MCP server
```

And the MCP server only needs the token file path (already set in
`docker-compose.yml`):

```bash
PRUSA_TOKEN_FILE=./data/prusa_tokens.json
```

No `PRUSA_EMAIL`, `PRUSA_PASSWORD`, `HEADLESS`, or Playwright variables are
used any more — delete them from your local `.env` if they are still there.

### Disabling Prusa Integration

If you don't need 3D printing features, disable MCP:

```bash
SKIP_MCP=true
```

Then use the basic Docker deployment:
```bash
docker-compose up -d
```

## Using the Prusa Agent

### Listing Printers

Ask the assistant to show available printers:

```
Show my Prusa printers
```

The agent will return a list of your connected printers with their status.

### Printing a Design

After optimizing a design and exporting to STL:

```
Print the design on my MK4
```

The agent will:
1. Upload the STL file to Prusa Connect
2. Select the specified printer
3. Start the print job
4. Provide a confirmation with job details

### Checking Printer Status

```
What's the status of my Prusa printer?
```

Returns information about:
- Printer state (idle, printing, paused, etc.)
- Current print job (if any)
- Print progress
- Estimated time remaining

### Uploading Files

Upload a file without starting a print:

```
Upload design.stl to Prusa Connect
```

### Managing Files

List files stored in Prusa Connect:

```
Show my Prusa Connect files
```

Delete a file:

```
Delete old-design.stl from Prusa Connect
```

## Workflow Example

Here's a complete workflow from design to print:

```
User: Create an optimized 2D beam design with 35% material
Assistant: [Runs optimization using EngiBench]

User: Export that to STL with 10mm height
Assistant: [Generates STL file in outputs/ directory]

User: Print it on my Prusa MK4
Assistant: [Uploads to Prusa Connect and starts print job]
  ✓ File uploaded: beam_optimized.stl
  ✓ Printer selected: Original Prusa MK4
  ✓ Print job started
  ⏱ Estimated time: 2h 15min
```

## Advanced Usage

### Specifying Printer by Name

If you have multiple printers:

```
Print design.stl on "Workshop MK3S+"
```

### Print Settings

The agent uses default print settings from Prusa Connect. To customize:

1. Upload the file first:
   ```
   Upload beam.stl to Prusa Connect
   ```

2. Configure settings in Prusa Connect web interface

3. Start the print:
   ```
   Start printing beam.stl on MK4
   ```

### Monitoring Prints

Check print progress:

```
How's my print going?
```

Or for a specific printer:

```
Status of my MK4 printer
```

## MCP Server Architecture

The MCP server acts as a bridge between the EngiAI and Prusa Connect:

```
┌──────────────────┐
│  Engineer        │
│  Assistant       │
│  (Prusa Agent)   │
└────────┬─────────┘
         │ HTTP requests
         ▼
┌──────────────────┐
│  Prusa MCP       │
│  Server          │
│  (HTTP/SSE)      │
└────────┬─────────┘
         │ REST API
         ▼
┌──────────────────┐
│  Prusa Connect   │
│  (Cloud/Local)   │
└──────────────────┘
```

### Running MCP Server Standalone

For local development:

```bash
# Start MCP server
./services/prusa_mcp_server/run.sh

# Or via Make
make run-mcp
```

Then configure the main app:

```bash
export SKIP_MCP=false
export PRUSA_MCP_URL=http://localhost:8765
streamlit run src/ui/streamlit_app.py
```

## Troubleshooting

### MCP Connection Failed

**Check MCP server is running**:
```bash
# Docker
docker-compose ps prusa-mcp-server

# Test endpoint
curl http://localhost:8765/sse
```

**Solutions**:
1. Verify `SKIP_MCP=false` in `.env`
2. Check `PRUSA_MCP_URL` is correct
3. Restart Docker containers:
   ```bash
   docker-compose restart
   ```

### Not Authenticated / Login Failed

**Symptoms**: Prusa tools return `Not authenticated` or `Stored tokens are missing a refresh_token`.

**Solutions**:
1. Re-run `make prusa-login` on the host to refresh `./data/prusa_tokens.json`.
2. Verify the token file exists and is mounted into the container:
   ```bash
   ls -l ./data/prusa_tokens.json
   docker compose exec prusa-mcp-server ls -l /app/data/prusa_tokens.json
   ```
3. If the refresh token has been revoked (e.g. you changed your Prusa Account
   password or removed the authorized app), delete `./data/prusa_tokens.json`
   and run `make prusa-login` again.
4. For headless hosts, pass `--no-browser` to the underlying CLI and copy-paste
   the printed URL into any browser that can reach `account.prusa3d.com` and
   the loopback port printed by the login helper.

### Printer Not Found

**Solutions**:
- Check printer is connected to Prusa Connect
- Verify printer is online
- Try specifying printer name exactly as shown in Prusa Connect

### File Upload Failed

**Common causes**:
- File size too large (limit: 100MB)
- Invalid STL/3MF format
- Network connectivity issues

**Check file**:
```bash
# Verify STL file
ls -lh outputs/*.stl
```

### Print Job Won't Start

**Checklist**:
- ✅ Printer is idle (not printing, paused, or in error state)
- ✅ File was uploaded successfully
- ✅ Printer has filament loaded
- ✅ Print bed is clear

## Security Considerations

1. **Token File**: `./data/prusa_tokens.json` is a bearer credential for your
   Prusa Account — it is written with mode `0600` and must never be committed
   to git. `./data/` is already in `.gitignore`.
2. **No Passwords On Disk**: Your Prusa Account email and password are never
   stored locally; they are only ever entered on `account.prusa3d.com` during
   the one-time browser login.
3. **Public OAuth Client + PKCE**: Authentication uses the public PrusaSlicer
   OAuth client with PKCE, bound to a loopback `127.0.0.1` redirect. There is
   no client secret to leak.
4. **Revoking Access**: To revoke the MCP server's access, sign in to
   `account.prusa3d.com`, remove the authorized application, and delete
   `./data/prusa_tokens.json`.
5. **MCP Server**: Only accessible within the Docker network by default.
6. **Network**: Use HTTPS for production deployments.

## Limitations

- Only supports Prusa printers with Prusa Connect
- Requires internet connection for cloud Prusa Connect
- Print settings must be configured in Prusa Connect web interface
- Cannot modify prints in progress (pause/resume via Prusa Connect UI)

## Best Practices

1. **Test prints**: Start with small test prints to verify setup
2. **File naming**: Use descriptive names for uploaded files
3. **Clean up**: Regularly delete old files from Prusa Connect
4. **Monitor first print**: Watch the first layer of new designs
5. **Backup designs**: Keep local copies of all STL files

## Further Reading

- [Prusa Connect Documentation](https://help.prusa3d.com/guide/prusa-connect_245530)
- [MCP Server Code](https://github.com/gioelemo/EngiAI/tree/main/services/prusa_mcp_server)
- [Docker Deployment](../docker_deployment.md)
- [Architecture Overview](../architecture.md)

## Next Steps

- [Agent Documentation](agents.md) - Learn about other agents
- [Tool Reference](tools.md) - Explore available tools
- [Configuration Guide](../configuration.md) - Environment variables
