"""
Pytest fixtures for Playwright e2e tests.

`streamlit_server` spawns a fresh Streamlit process on port 8502 (avoids
conflict with the Docker app on 8501), waits for it to be healthy, and tears
it down after the session.  `live_page` wraps pytest-playwright's `page`
fixture and navigates to the running server.
"""

import os
import subprocess
import time
from pathlib import Path

import pytest
import requests
from playwright.sync_api import Page

# ── constants ──────────────────────────────────────────────────────────────────

_PORT = 8502
_BASE_URL = f"http://localhost:{_PORT}"
_HEALTH_URL = f"{_BASE_URL}/_stcore/health"
_STARTUP_TIMEOUT = 60  # seconds — Streamlit cold-start can be slow
_PROJECT_ROOT = Path(__file__).parent.parent.parent


# ── helpers ────────────────────────────────────────────────────────────────────


def _wait_for_streamlit(url: str, timeout: int) -> None:
    """Poll the Streamlit health endpoint until 200 OK or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"Streamlit did not become healthy at {url} within {timeout}s")


# ── fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def streamlit_server(tmp_path_factory):
    """Start a Streamlit process for the e2e session and yield its base URL.

    Uses a temporary SQLite database and fake API keys so the process starts
    without any external services.  LangChain only validates API keys on the
    first model call, not at init, so the UI renders normally.
    """
    tmp_db = tmp_path_factory.mktemp("e2e_db") / "conversations.db"

    # Mirror pyproject.toml pythonpath = [".", "services"] for the subprocess.
    # Without this, `prusa_mcp_server` (in services/) is not importable.
    _services = str(_PROJECT_ROOT / "services")
    _existing = os.environ.get("PYTHONPATH", "")
    _pythonpath = os.pathsep.join(
        p for p in [str(_PROJECT_ROOT), _services, _existing] if p
    )

    env = {
        **os.environ,
        "PYTHONPATH": _pythonpath,
        # Disable optional external integrations
        "SKIP_MCP": "true",
        "SKIP_MMORE": "true",
        "TESTING": "true",
        # Real keys are used if available in the environment; otherwise placeholders
        "OPENAI_API_KEY": os.environ.get("OPENAI_API_KEY", "sk-e2e-placeholder"),
        "GOOGLE_API_KEY": os.environ.get("GOOGLE_API_KEY", "fake-google-key"),
        "TAVILY_API_KEY": os.environ.get("TAVILY_API_KEY", "tvly-e2e-placeholder"),
        # Use a temp SQLite file (app falls back to SQLite automatically anyway)
        "DATABASE_URL": f"sqlite:///{tmp_db}",
        # Disable Weave tracing to avoid auth errors with placeholder keys
        "USE_WEAVE": "false",
        "LANGCHAIN_TRACING_V2": "false",
    }

    proc = subprocess.Popen(
        [
            "streamlit",
            "run",
            "src/ui/streamlit_app.py",
            f"--server.port={_PORT}",
            "--server.headless=true",
            "--server.runOnSave=false",
            "--global.developmentMode=false",
        ],
        env=env,
        cwd=str(_PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    try:
        _wait_for_streamlit(_HEALTH_URL, timeout=_STARTUP_TIMEOUT)
    except TimeoutError:
        proc.terminate()
        stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
        stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""
        raise TimeoutError(
            f"Streamlit failed to start.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
        ) from None

    yield _BASE_URL

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()


@pytest.fixture
def live_page(page: Page, streamlit_server: str) -> Page:
    """Navigate to the running Streamlit app and wait for initial render."""
    page.goto(streamlit_server)
    # Wait until the main Streamlit React tree has settled
    page.wait_for_load_state("networkidle")
    return page
