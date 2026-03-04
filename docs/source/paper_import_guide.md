# Paper Import Guide

This guide explains how to automatically import PDF papers from your local filesystem or mounted network share into your MMORE RAG system for document-based Q&A.

## Overview

The paper import workflow allows you to:
- Upload research papers, documentation, and PDFs to the MMORE service
- Search and query these documents using natural language
- Get answers with context from your document collection
- Track which documents have been uploaded

The import script (`scripts/import_local_papers.py`) provides automatic importing with:

- **Incremental Updates**: Only processes new or modified files
- **State Tracking**: Remembers which files have been imported (in database)
- **Error Handling**: Continues processing even if individual files fail
- **Retry Logic**: Automatic retry with exponential backoff for failed uploads (3 attempts)
- **Recursive Directory Scanning**: Processes PDFs in subdirectories
- **Metadata Tracking**: Tags documents with source information in database
- **Dry-Run Mode**: Preview what will be imported without making changes
- **Large File Support**: Handles files up to 100MB with progress tracking

## Workflow

```
┌─────────────────┐
│ Local PDFs      │
│ (Your Computer  │
│  or Network     │
│  Share)         │
└────────┬────────┘
         │
         ▼
┌─────────────────┐      ┌──────────────────┐
│ import_local_   │─────>│ MMORE Service    │
│ papers.py       │      │ (localhost:8000) │
│ (Upload Script) │      │                  │
└─────────────────┘      └────────┬─────────┘
         │                        │
         │                        ▼
         │               ┌─────────────────┐
         │               │ Document Store  │
         │               │ (Searchable)    │
         │               └─────────────────┘
         ▼
┌─────────────────┐
│ Database        │
│ (Track State)   │
└─────────────────┘
```

## Installation

### 1. Install Dependencies

All dependencies are automatically installed when you set up the project:

```bash
# If using conda
conda activate engineer-assistant

# If using Docker
# Dependencies are already included in the container
```

### 2. Verify MMORE Service

Ensure the MMORE RAG service is running at `http://localhost:8000`:

```bash
# Test MMORE service is responding
curl http://localhost:8000/health
# Should return: {"status": "healthy"} or similar
```

If MMORE is not running, see the [Installation Guide](installation.md) for setup instructions.

### 3. Configure Import Path

Add the following to your `.env` file:

```bash
# Paper Import Configuration
PAPERS_SOURCE_DIR=/path/to/your/papers
PAPERS_STATE_FILE=data/local_import_state.json
MMORE_RAG_URL=http://localhost:8000  # MMORE service URL
```

**Configuration Parameters:**

- `PAPERS_SOURCE_DIR`: Path to directory containing PDFs (can be a mounted network share)
  - Examples: `/Volumes/ShareName/Papers` (macOS), `/mnt/share/papers` (Linux), `C:/Papers` (Windows)
- `PAPERS_STATE_FILE`: JSON file tracking import state (default: `data/local_import_state.json`)
  - Tracks file paths, sizes, modification times, and upload status
  - Used to detect changes and avoid re-uploading unchanged files
- `MMORE_RAG_URL`: MMORE service URL (default: `http://localhost:8000`)

**Important Notes:**

- **State Tracking**: The script uses both the state file AND the database to track uploads
  - State file: Quick lookup for file metadata (size, mtime)
  - Database: Authoritative record of successfully uploaded documents
- **File Size Limit**: Files larger than 50MB will trigger a warning but still upload
- **Network Shares**: Works with any mounted filesystem (NFS, SMB, AFP, etc.)

**For Mounted Network Shares:**

If you have a network share mounted, use that path directly:

```bash
# macOS example
PAPERS_SOURCE_DIR=/Volumes/Soheyl/gioelemo/RAG/Papers

# Linux example
PAPERS_SOURCE_DIR=/mnt/network_share/papers

# Windows example (in WSL or Git Bash)
PAPERS_SOURCE_DIR=/mnt/c/Users/YourName/Documents/Papers
```

## Usage

### Quick Start

Once configured in `.env`, simply run:

```bash
# From project root directory
python scripts/import_local_papers.py

# This will:
# 1. Scan PAPERS_SOURCE_DIR for PDFs
# 2. Compare with previously imported files
# 3. Upload only new or modified files to MMORE
# 4. Update database with document metadata
# 5. Save progress to PAPERS_STATE_FILE
```

### Command-Line Usage

Specify the source directory directly:

```bash
python scripts/import_local_papers.py /Volumes/Soheyl/gioelemo/RAG/Papers
```

**Configuration Priority:**
1. Command-line argument (highest priority)
2. `PAPERS_SOURCE_DIR` environment variable from `.env`
3. Default: `./papers/` (relative to script location)

### Dry-Run Mode

Preview what will be imported without uploading:

```bash
python scripts/import_local_papers.py --dry-run

# Output example:
# Found 150 PDF files
# 120 already imported (unchanged)
# 25 new files to import
# 5 modified files to re-import
# Total to process: 30 files
```

### Command-Line Options

**Available Options:**

- `source_dir`: Path to directory containing PDFs (positional argument)
- `--state-file`: Import state file (default: from `PAPERS_STATE_FILE` or `data/local_import_state.json`)
- `--dry-run`: List files without importing
- `--max-files`: Maximum number of files to import
- `--verbose`, `-v`: Enable verbose logging

### Examples

```bash
# Use .env configuration (simplest)
python scripts/import_local_papers.py

# Dry run to preview
python scripts/import_local_papers.py --dry-run

# Specify path directly (overrides .env)
python scripts/import_local_papers.py /Volumes/Share/Papers

# Preview first 5 files
python scripts/import_local_papers.py --dry-run --max-files 5

# Import only 10 files at a time
python scripts/import_local_papers.py --max-files 10

# Verbose logging for debugging
python scripts/import_local_papers.py --verbose
```

## How It Works

### Import Process

1. **Scan for PDFs**: Recursively lists all PDF files in the source directory
2. **Filter New Files**: Compares against state file to find new/modified files
3. **Upload to MMORE**: Sends PDF file to MMORE service for processing
4. **Retry on Failure**: Automatically retries failed uploads with exponential backoff (3 attempts)
5. **Track in Database**: Records uploaded files in local SQLite database
6. **Update State**: Records imported files with metadata
7. **Save State**: Persists state after each successful upload

### State Tracking

The script maintains a state file (configured via `PAPERS_STATE_FILE` or default `data/local_import_state.json`) that tracks:

- File path (relative to source directory)
- File hash (size + modification time)
- Import timestamp
- MMORE file ID
- Full file path

This enables incremental updates - only new or modified files are re-imported.

### File ID Generation

Each uploaded document receives a unique file ID generated from its relative path:

```python
file_id = "papers_subdir_filename"  # Path separators replaced with underscores
```

This file ID is used to:
- Track documents in MMORE service
- Query specific documents
- Delete documents when needed

## Scheduling Automatic Imports

### Using Cron (Linux/macOS)

Add to your crontab to run daily at 2 AM:

```bash
crontab -e
```

Add this line:

```
0 2 * * * cd /path/to/engineer-assistant && /path/to/python scripts/import_local_papers.py >> data/local_import.log 2>&1
```

### Using Task Scheduler (Windows)

1. Open Task Scheduler
2. Create New Task
3. Set trigger (e.g., daily at 2 AM)
4. Set action to run:
   ```
   C:\path\to\python.exe C:\path\to\engineer-assistant\scripts\import_local_papers.py
   ```
5. Set working directory: `C:\path\to\engineer-assistant`

### Using systemd Timer (Linux)

Create `/etc/systemd/system/paper-import.service`:

```ini
[Unit]
Description=Import Papers to RAG
After=network.target

[Service]
Type=oneshot
User=youruser
WorkingDirectory=/path/to/engineer-assistant
EnvironmentFile=/path/to/engineer-assistant/.env
ExecStart=/path/to/python scripts/import_local_papers.py
StandardOutput=append:/path/to/engineer-assistant/data/local_import.log
StandardError=append:/path/to/engineer-assistant/data/local_import.log

[Install]
WantedBy=multi-user.target
```

Create `/etc/systemd/system/paper-import.timer`:

```ini
[Unit]
Description=Run Paper Import Daily

[Timer]
OnCalendar=daily
Persistent=true

[Install]
WantedBy=timers.target
```

Enable and start:

```bash
sudo systemctl enable paper-import.timer
sudo systemctl start paper-import.timer
```

### Using launchd (macOS)

Create `~/Library/LaunchAgents/com.engineer-assistant.paper-import.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.engineer-assistant.paper-import</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/python</string>
        <string>/path/to/engineer-assistant/scripts/import_local_papers.py</string>
    </array>
    <key>WorkingDirectory</key>
    <string>/path/to/engineer-assistant</string>
    <key>EnvironmentVariables</key>
    <dict>
        <key>PATH</key>
        <string>/usr/local/bin:/usr/bin:/bin</string>
    </dict>
    <key>StartCalendarInterval</key>
    <dict>
        <key>Hour</key>
        <integer>2</integer>
        <key>Minute</key>
        <integer>0</integer>
    </dict>
    <key>StandardOutPath</key>
    <string>/path/to/engineer-assistant/data/local_import.log</string>
    <key>StandardErrorPath</key>
    <string>/path/to/engineer-assistant/data/local_import.log</string>
</dict>
</plist>
```

Load it:

```bash
launchctl load ~/Library/LaunchAgents/com.engineer-assistant.paper-import.plist
```

## Querying Imported Documents

### Using the Streamlit UI

The easiest way to query imported papers is through the chat interface. The RAG agent will automatically search your indexed documents:

```
"What are the key findings in recent papers about FEA?"
"What does the EngiBench paper say about topology optimization?"
```

### Filtering by Source

Query specific documents using their file IDs:

```python
from src.tools import MMOREClient

mmore_client = MMOREClient()
results = mmore_client.retrieve(
    query="FEA optimization",
    file_ids=["papers_subfolder_paper1", "papers_subfolder_paper2"],
    max_matches=5
)
```

### Via Streamlit UI

The imported papers are automatically available in the Streamlit chat interface:

```bash
streamlit run src/ui/streamlit_app.py
```

Just ask questions and the RAG system will search through all documents in MMORE.

## Inspecting the Documents

### Using the Inspector Script

```bash
# Show all uploaded documents
python scripts/inspect_mmore.py

# Get detailed stats
python scripts/inspect_mmore.py --stats

# List all document IDs
python scripts/inspect_mmore.py --list
```

### Via Database Query

Check the files indexed in MMORE:

```python
from src.tools import MMOREClient

mmore_client = MMOREClient()
file_ids = mmore_client.list_files()

for file_id in file_ids:
    print(file_id)
```

## Troubleshooting

### Configuration Not Found

**Problem**: Script says "Missing source directory"

**Solutions**:
- Check that `PAPERS_SOURCE_DIR` is set in your `.env` file
- Verify `.env` file is in the project root directory
- Try providing path via command line: `python scripts/import_local_papers.py /path/to/papers`

### MMORE Connection Issues

**Problem**: "Connection aborted" or "Remote end closed connection without response"

**Solutions**:
- Verify MMORE service is running: `docker ps | grep mmore`
- Check `MMORE_RAG_URL` is correct in `.env` (default: `http://localhost:8000`)
- Test MMORE health: `curl http://localhost:8000/`
- For large files (>50MB), increase timeout or upload in smaller batches with `--max-files`
- Check MMORE service logs: `docker logs mmore-service`

### File Access Issues

**Problem**: Cannot read files from directory

**Solutions**:
- Verify the path exists: `ls -la $PAPERS_SOURCE_DIR`
- Check file permissions: `ls -l /path/to/papers/*.pdf`
- For mounted shares, ensure the mount is active: `mount | grep Volumes`
- Try with verbose mode: `--verbose`

### No Files Found

**Problem**: Script reports "0 PDF files found"

**Solutions**:
- Check the source directory path is correct
- Verify files have `.pdf` extension (case-insensitive)
- Look for hidden files (those starting with `.` are skipped)
- Run with `--verbose` to see scanning process

### File Processing Errors

**Problem**: PDFs fail to upload

**Solutions**:
- Check that files are valid PDFs (try opening them)
- Verify sufficient disk space for temporary files
- Look at detailed logs with `--verbose`
- Check MMORE service has enough memory (increase in docker-compose.yml if needed)
- Try processing with `--max-files 5` to upload in smaller batches
- The script will automatically retry failed uploads 3 times with exponential backoff

### State File Corruption

**Problem**: State file is corrupted

**Solution**: Delete the state file to start fresh (will re-import all files):
```bash
rm data/local_import_state.json
```

### Mount Disappears

**Problem**: Mounted network share disconnects

**Solutions**:
- Use macOS auto-mount on login (Finder → Go → Connect to Server → Add to login items)
- Set up automatic reconnection in your scheduling script
- Add mount check before import:
  ```bash
  #!/bin/bash
  if [ ! -d "/Volumes/Soheyl" ]; then
    mount_smbfs //user@server/share /Volumes/Soheyl
  fi
  python scripts/import_local_papers.py
  ```

## Logs

The script maintains logs in two places:

1. **File Log**: `data/local_import.log` - Persistent log of all imports
2. **Console Output**: Real-time progress and summary

Log format:
```
2025-11-06 10:30:15 - INFO - Scanning for PDF files in /Volumes/Papers...
2025-11-06 10:30:16 - INFO - Found 42 PDF files
2025-11-06 10:30:17 - INFO - New file: paper_2025.pdf
2025-11-06 10:30:18 - INFO - Processing paper_2025.pdf...
2025-11-06 10:30:22 - INFO - Successfully imported paper_2025.pdf (15 chunks)
```

## Best Practices

1. **Use .env**: Store configuration in `.env` for easy management
2. **Test First**: Always run with `--dry-run` before first import
3. **Start Small**: Use `--max-files` to test with a few files first
4. **Monitor Logs**: Check `data/local_import.log` for errors
5. **Backup State**: Keep backups of `data/local_import_state.json`
6. **Schedule Off-Hours**: Run imports during low-usage times
7. **Verify Mounts**: Ensure network shares are mounted before scheduled runs
8. **Clean Logs**: Periodically rotate or clean old log files

## Performance Considerations

- **File Size**: Large PDFs take longer to process
- **File Count**: Processing many files can take time
- **Embedding Generation**: OpenAI API calls are rate-limited
- **Disk I/O**: Reading from network shares is slower than local disk

**Optimization Tips**:
- Use `--max-files` to process in batches
- Schedule during off-peak hours
- Consider copying files locally for faster processing
- Monitor OpenAI API usage

## Advanced Usage

### Multiple Sources

Import from multiple directories with different .env values:

```bash
# Set different source for one-off import
PAPERS_SOURCE_DIR=/Volumes/OtherShare/Papers python scripts/import_local_papers.py

# Or override with command line
python scripts/import_local_papers.py /Volumes/OtherShare/Papers
```

### Custom Collections

Organize imports into separate collections:

```bash
python scripts/import_local_papers.py \
  /Volumes/Engineering/Papers \
  --collection engineering_papers

python scripts/import_local_papers.py \
  /Volumes/Research/Papers \
  --collection research_papers
```

### Programmatic Usage

Use the importer in your own scripts:

```python
from scripts.import_local_papers import LocalPaperImporter

importer = LocalPaperImporter(
    source_dir="/path/to/papers"
)

stats = importer.run(dry_run=False, max_files=10)
print(f"Imported {stats['successful']} files")
```

## Example: ETHZ Network Share

For ETHZ users with mounted shares, add to `.env`:

```bash
# Paper Import Configuration
PAPERS_SOURCE_DIR=/Volumes/Soheyl/gioelemo/RAG/Papers
PAPERS_STATE_FILE=data/local_import_state.json
```

Then simply run:

```bash
# Test it
python scripts/import_local_papers.py --dry-run

# Import papers
python scripts/import_local_papers.py

# Check what was imported
python scripts/inspect_mmore.py
```

## Support

For issues or questions:

1. Check logs in `data/local_import.log`
2. Run with `--verbose` for detailed debugging
3. Review this guide's troubleshooting section
4. Verify `.env` configuration is correct
5. Check project README for general setup issues

## Quick Reference

### Common Commands

| Command | Description |
|---------|-------------|
| `python scripts/import_local_papers.py` | Import all new/modified papers from `PAPERS_SOURCE_DIR` |
| `python scripts/import_local_papers.py --dry-run` | Preview what will be imported |
| `python scripts/import_local_papers.py --verbose` | Show detailed progress |
| `python scripts/import_local_papers.py --max-files 10` | Import only first 10 files |
| `python scripts/inspect_mmore.py` | List all uploaded documents |
| `python scripts/inspect_mmore.py --stats` | Show upload statistics |

### Key Files

| File | Purpose |
|------|---------|
| `.env` | Configuration (PAPERS_SOURCE_DIR, MMORE_RAG_URL) |
| `data/local_import_state.json` | Tracks file metadata and upload status |
| `data/conversations.db` | Database with document records |
| `data/local_import.log` | Import operation logs |
| `scripts/import_local_papers.py` | Main import script |
| `scripts/inspect_mmore.py` | Document inspection tool |

### Retry Behavior

| Attempt | Wait Time | Total Time |
|---------|-----------|------------|
| 1 | 0s | 0s |
| 2 | 5s | 5s |
| 3 | 10s | 15s |
| Failed | - | Give up after 15s |

Files larger than 50MB get extra warnings but still upload with retry logic.
