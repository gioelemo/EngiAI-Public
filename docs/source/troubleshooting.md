# Troubleshooting Guide

This guide helps you diagnose and fix common issues with Engineer Assistant.

## Docker Issues

### Container Won't Start

**Symptoms**: Container exits immediately or fails health checks

**Check logs**:
```bash
docker-compose logs -f chatbot
```

**Common Causes**:

1. **Missing API keys**:
   ```bash
   # Check your .env file
   cat .env | grep API_KEY
   ```
   Solution: Ensure `OPENAI_API_KEY` and `TAVILY_API_KEY` are set

2. **Port already in use**:
   ```bash
   # Check what's using port 8501
   lsof -i :8501
   ```
   Solution: Stop the conflicting service or change port in `docker-compose.yml`

3. **Insufficient resources**:
   - Increase Docker memory allocation (Docker Desktop → Settings → Resources)
   - Recommended: 4GB RAM minimum

### Can't Access UI

**Symptom**: Browser can't connect to http://localhost:8501

**Steps**:
1. Check containers are running:
   ```bash
   docker-compose ps
   ```
   All containers should show "Up" status

2. Check Docker Desktop is running (if using Docker Desktop)

3. Try accessing via Docker host IP:
   ```bash
   # Find container IP
   docker inspect engineer-assistant-chatbot | grep IPAddress
   ```

4. Check firewall settings (Linux):
   ```bash
   sudo ufw allow 8501
   ```

### Permission Errors (Linux)

**Symptom**: "Permission denied" errors

**Solution**: Add user to docker group:
```bash
sudo usermod -aG docker $USER
# Log out and back in for changes to take effect
```

### Out of Disk Space

**Symptom**: Build fails or containers crash

**Check disk usage**:
```bash
docker system df
```

**Clean up**:
```bash
# Remove unused images and containers
docker system prune -a

# Remove specific images
docker rmi $(docker images -q)
```

## Local Development Issues

### Ruff Not Found in VS Code

**Symptom**: VS Code shows "Ruff not found"

**Solutions**:
1. Ensure conda environment is activated:
   ```bash
   conda activate engineer-assistant
   ```

2. Restart VS Code after activating environment

3. Set Python interpreter in VS Code:
   - `Cmd/Ctrl + Shift + P` → "Python: Select Interpreter"
   - Choose `engineer-assistant` conda environment

### Pre-commit Not Working

**Symptom**: Hooks don't run on commit

**Solutions**:
1. Reinstall hooks:
   ```bash
   pre-commit install
   ```

2. Run manually to test:
   ```bash
   pre-commit run --all-files
   ```

3. Skip hooks temporarily (not recommended):
   ```bash
   git commit --no-verify
   ```

### Environment Creation Fails

**Symptom**: `conda env create` fails

**Solutions**:
1. Update conda:
   ```bash
   conda update -n base conda
   ```

2. Clear conda cache:
   ```bash
   conda clean --all
   ```

3. Try creating environment again:
   ```bash
   conda env remove -n engineer-assistant
   conda env create -f environment.yml
   ```

### Import Errors

**Symptom**: `ModuleNotFoundError` when running Python

**Solutions**:
1. Verify environment is activated:
   ```bash
   conda activate engineer-assistant
   which python  # Should point to conda env
   ```

2. Verify you're in project root:
   ```bash
   pwd  # Should show .../engineer-assistant
   ```

3. Reinstall package:
   ```bash
   pip install -e .
   ```

### Streamlit Command Not Found

**Symptom**: `streamlit: command not found`

**Solution**: Activate conda environment:
```bash
conda activate engineer-assistant
streamlit run src/ui/streamlit_app.py
```

## API & Configuration Issues

### Missing API Keys Error

**Symptom**: `ValueError` about missing configuration

**Check .env file exists**:
```bash
ls -la .env
```

**If missing**, create from example:
```bash
cp .env.example .env
# Edit .env and add your keys
```

**Verify keys are set**:
```bash
# Don't run this in production - just for debugging
cat .env | grep API_KEY
```

### Chatbot Not Responding

**Symptoms**: No response or timeout errors

**Possible Causes**:

1. **Invalid OpenAI API key**:
   - Verify key at https://platform.openai.com/api-keys
   - Check you have sufficient credits

2. **API rate limiting**:
   - Wait a few minutes and try again
   - Check OpenAI dashboard for rate limits

3. **Network issues**:
   ```bash
   # Test connectivity
   curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"
   ```

### Web Search Not Working

**Symptom**: Search tool fails

**Check Tavily API key**:
- Verify key in `.env` file
- Test at https://tavily.com/

### Database Errors

**Symptom**: "database is locked" or connection errors

**For SQLite**:
```bash
# Close other applications using the database
# Or use PostgreSQL instead
```

**For PostgreSQL**:
1. Check PostgreSQL is running:
   ```bash
   # Docker
   docker-compose ps postgres

   # Local (macOS)
   brew services list | grep postgresql

   # Local (Linux)
   sudo systemctl status postgresql
   ```

2. Verify connection string:
   ```bash
   # Check DATABASE_URL in .env
   cat .env | grep DATABASE_URL
   ```

3. For Docker, ensure hostname is `postgres` not `localhost`:
   ```bash
   # Correct for Docker
   DATABASE_URL=postgresql://user:pass@postgres:5432/engineer_assistant

   # Correct for local
   DATABASE_URL=sqlite:///data/conversations.db
   ```

### "Could not translate host name 'postgres'" (Local)

**Symptom**: Error when running locally with Streamlit

**Cause**: Your `.env` is configured for Docker

**Solution**: Change `DATABASE_URL` to use SQLite:
```bash
# In .env file
DATABASE_URL=sqlite:///data/conversations.db
```

## Prusa MCP Issues

### MCP Connection Failed

**Symptom**: Prusa agent can't connect

**Check MCP server is running**:
```bash
# For Docker deployment
docker-compose ps prusa-mcp-server

# Test connection
curl http://localhost:8765/sse
```

**Solutions**:
1. Check `SKIP_MCP` setting in `.env`:
   ```bash
   SKIP_MCP=true   # Disables MCP integration
   SKIP_MCP=false  # Enables MCP integration
   ```

2. Verify MCP server is accessible:
   ```bash
   # From container
   docker exec engineer-assistant-chatbot curl -v http://prusa-mcp-server:8000/sse
   ```

### Tools Not Loading

**Symptom**: Prusa-related tools missing

**Solution**: Set `SKIP_MCP=true` to disable MCP integration if you don't need 3D printing features

### prusa-mcp Folder Not Found



Or use basic deployment without MCP:
```bash
docker-compose up -d  # Instead of docker-compose -f docker-compose.mcp.yml up -d
```

## Host Service & GUI Integration

### "Cannot connect to host service" Error

**Symptom**: Can't open PrusaSlicer or other GUI apps from Docker

**Check host service is running**:
```bash
lsof -i :9999
```

**Start host service** (on your local machine):
```bash
python host_service.py
```

**Verify from Docker**:
```bash
docker exec engineer-assistant-chatbot curl http://host.docker.internal:9999/health
```

### PrusaSlicer Won't Open

**Solutions**:
1. Verify PrusaSlicer is installed on host machine

2. Verify host service whitelist includes the app (in `host_service.py`)

### Port 9999 Already in Use

**Solution**: Change port in `.env`:
```bash
HOST_SERVICE_PORT=9998
```

Then restart both host service and Docker containers.

## SSH & HPC Connection Issues

### "SSH key is encrypted" Error (Docker)

**Symptom**: Can't connect to HPC cluster from Docker

**Solution**: Use SSH agent forwarding

1. **On host machine**, add key to SSH agent:
   ```bash
   # macOS
   ssh-add --apple-use-keychain ~/.ssh/id_ed25519

   # Linux
   ssh-add ~/.ssh/id_ed25519
   ```

2. **Verify key is loaded**:
   ```bash
   ssh-add -l
   ```

3. **Restart Docker container**:
   ```bash
   docker-compose restart chatbot
   ```

### "Agent has no identities" Error

**Check SSH agent is running**:
```bash
echo $SSH_AUTH_SOCK
# Should show a path like: /private/tmp/com.apple.launchd.*/Listeners
```

**Start SSH agent if needed**:
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Connection Works Locally But Fails in Docker

**Checklist**:
1. ✅ SSH agent is running on host: `ssh-add -l`
2. ✅ `SSH_AUTH_SOCK` environment variable is set
3. ✅ `~/.ssh/config` has `AddKeysToAgent yes` and `UseKeychain yes`
4. ✅ Restart Docker after adding keys

### SSH Key Permission Errors

**Fix permissions**:
```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/config
chmod 644 ~/.ssh/known_hosts
```

### HPC Job Submission Fails

**Common Issues**:

1. **Can't connect to cluster**:
   ```bash
   # Test SSH manually
   ssh your-cluster-alias
   ```

2. **SLURM commands not found**:
   - Verify you're on a login node
   - Check cluster documentation

3. **Job fails immediately**:
   ```bash
   # Check job logs
   cat slurm-<jobid>.out
   cat slurm-<jobid>.err
   ```

4. **Module not found**:
   ```bash
   # Check available modules
   module avail
   ```

## Performance Issues

### Slow Response Times

**Possible Causes**:

1. **Large document collection**:
   - Monitor MMORE service performance
   - Consider limiting search scope with file_ids

2. **Slow API responses**:
   - Check OpenAI API status
   - Try different model (e.g., gpt-3.5-turbo)

3. **Insufficient resources**:
   - Increase Docker memory allocation
   - Close other applications

### High Memory Usage

**Solutions**:
1. Limit conversation history length
2. Clear browser cache
3. Restart Docker containers:
   ```bash
   docker-compose restart
   ```

### Streamlit Performance Issues

**Solutions**:
1. Reduce max tokens in settings
2. Limit chat history length
3. Use a smaller LLM model
4. Close unused browser tabs

**Clear Streamlit cache**:
```bash
streamlit cache clear
```

## Testing Issues

### Tests Failing

**Run tests with verbose output**:
```bash
pytest -v
```

**Run specific test**:
```bash
pytest tests/test_specific_file.py -v
```

**Skip slow tests**:
```bash
pytest -m "not slow"
```

### Import Errors in Tests

**Ensure package is installed**:
```bash
pip install -e .
```

## Common Error Messages

### "ModuleNotFoundError: No module named 'src'"

**Solution**: Install package in development mode:
```bash
pip install -e .
```

### "ValueError: Invalid configuration"

**Solution**: Check all required environment variables in `.env`

### "Connection refused" (Port 8501)

**Solutions**:
1. Check Docker containers are running
2. Check firewall settings
3. Try different port in docker-compose.yml

### "OSError: [Errno 28] No space left on device"

**Solution**: Free up disk space or clean Docker:
```bash
docker system prune -a
```

## Getting Help

If your issue isn't covered here:

1. **Check logs**:
   ```bash
   # Docker
   docker-compose logs -f chatbot

   # Local
   tail -f data/*.log
   ```

2. **Search GitHub Issues**: https://github.com/gioelemo/engineer-assistant/issues

3. **Open a new issue** with:
   - Operating system
   - Python version
   - Full error message
   - Steps to reproduce
   - What you've tried

4. **Review documentation**:
   - [Installation Guide](installation.md)
   - [Configuration Guide](configuration.md)
   - [Docker Deployment](docker_deployment.md)

## Prevention Tips

1. ✅ Always activate conda environment before running commands
2. ✅ Keep `.env` file up to date with required keys
3. ✅ Use Docker for consistent deployment
4. ✅ Regularly update dependencies
5. ✅ Back up your database regularly
6. ✅ Monitor API usage and credits
7. ✅ Test in development before deploying to production
