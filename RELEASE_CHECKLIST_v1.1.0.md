# Release Checklist for v1.1.0

## Pre-Release Preparation ✅

- [x] Version updated in `pyproject.toml` (1.1.0)
- [x] Version updated in `src/__init__.py` (1.1.0)
- [x] Version updated in `src/ui/settings.py` (1.1.0)
- [x] CHANGELOG.md updated with v1.1.0 section
- [x] Release notes created (RELEASE_NOTES_v1.1.0.md)

## Testing Checklist

- [ ] Run full test suite: `make test`
- [ ] Run linting: `make lint`
- [ ] Test Docker build: `docker build -t engineer-assistant:1.1.0 .`
- [ ] Test docker-compose basic: `docker-compose up -d`
- [ ] Test docker-compose full: `docker-compose -f docker-compose.mcp.yml up -d`
- [ ] Verify voice features work (both providers)
- [ ] Verify whiteboard/canvas functionality
- [ ] Test new engineering problems (Photonics2D, ThermoElastic2D)
- [ ] Verify MMORE integration works
- [ ] Test SLURM settings in UI

## Git Operations

### 1. Commit Changes
```bash
git add .
git commit -m "Release v1.1.0

- Updated version numbers to 1.1.0
- Updated CHANGELOG with release date
- Added release notes"
```

### 2. Create Tag
```bash
git tag -a v1.1.0 -m "Release version 1.1.0"
```

### 3. Push to GitHub
```bash
git push origin main
git push origin v1.1.0
```

## GitHub Release Steps

### 1. Create Release on GitHub
1. Go to: https://github.com/gioelemo/engineer-assistant/releases/new
2. Choose tag: `v1.1.0`
3. Release title: `Engineer Assistant v1.1.0`
4. Copy content from `RELEASE_NOTES_v1.1.0.md` into description

### 2. Attach Release Assets
Generate and attach the following:
```bash
# Generate release archives
./create_release_archives.sh

# This creates:
# - releases/engineer-assistant-v1.1.0.zip
# - releases/prusa-mcp-server-v1.1.0.zip
# - releases/mmore-service-v1.1.0.zip (if available)
```

Upload these files to the GitHub release.

### 3. Publish Release
- [ ] Check "Set as the latest release"
- [ ] Click "Publish release"

## Docker Hub (Optional)

If you want to publish to Docker Hub:

### 1. Tag Images
```bash
./docker-tag-version.sh 1.1.0
```

### 2. Push to Docker Hub
```bash
docker push gioelemo/engineer-assistant:1.1.0
docker push gioelemo/engineer-assistant:latest
docker push gioelemo/prusa-mcp-server:1.1.0
docker push gioelemo/prusa-mcp-server:latest
```

## Post-Release

- [ ] Update documentation links if needed
- [ ] Announce release (if applicable)
- [ ] Create new "Unreleased" section in CHANGELOG.md for next version
- [ ] Monitor GitHub issues for release-related problems
- [ ] Update README.md badges if needed

## Rollback Plan (If Needed)

If critical issues are found:

```bash
# Delete tag locally and remotely
git tag -d v1.1.0
git push origin :refs/tags/v1.1.0

# Delete GitHub release
# (Use GitHub UI to delete the release)

# Revert version changes
git revert <commit-hash>
```

## Notes
- Release date: 2025-12-15
- Previous version: v1.0.0 (2025-11-17)
- Major features: Voice providers, Excalidraw canvas, new engineering problems, MMORE integration
