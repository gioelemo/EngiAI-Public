# Prusa 3D Printer Integration

The Prusa Agent enables control of Prusa 3D printers through Prusa Connect, allowing you to print designs directly from the Engineer Assistant.

## Overview

The Prusa Agent communicates with Prusa Connect via an MCP (Model Context Protocol) server, providing tools for:
- Listing available printers
- Uploading STL/3MF files
- Starting print jobs
- Monitoring printer status
- Managing print files

## Prerequisites

- Prusa printer with Prusa Connect (cloud or local)
- Prusa Connect account credentials
- MCP server setup (included in Docker deployment)

## Setup

### Docker Deployment (Recommended)

The MCP server is automatically included when using the full Docker deployment:

```bash
docker-compose -f docker-compose.mcp.yml up -d
```

This starts:
- **Chatbot container**: Main application
- **PostgreSQL container**: Database
- **Prusa MCP Server**: Prusa Connect integration

### Configuration

Add your Prusa Connect credentials to `.env`:

```bash
# Prusa Connect Credentials
PRUSA_EMAIL=your-email@example.com
PRUSA_PASSWORD=your-password

# MCP Configuration
SKIP_MCP=false
PRUSA_MCP_URL=http://prusa-mcp-server:8000  # For Docker
# PRUSA_MCP_URL=http://localhost:8765        # For local MCP server
```

**Note**: You can also log in through the UI settings page instead of storing credentials in `.env`.

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

The MCP server acts as a bridge between the Engineer Assistant and Prusa Connect:

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
./prusa_mcp_server/run.sh

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
   docker-compose -f docker-compose.mcp.yml restart
   ```

### Login Failed

**Check credentials**:
- Verify email and password are correct
- Try logging in at https://connect.prusa3d.com/
- Use UI settings page to update credentials

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

1. **Credentials**: Store credentials in `.env`, never commit to git
2. **MCP Server**: Only accessible within Docker network by default
3. **API Keys**: Prusa Connect API keys are per-user
4. **Network**: Use HTTPS for production deployments

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
- [MCP Server Code](https://github.com/gioelemo/engineer-assistant/tree/main/prusa_mcp_server)
- [Docker Deployment](../docker_deployment.md)
- [Architecture Overview](../architecture.md)

## Next Steps

- [Agent Documentation](agents.md) - Learn about other agents
- [Tool Reference](tools.md) - Explore available tools
- [Configuration Guide](../configuration.md) - Environment variables
