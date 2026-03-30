"""
Tier-3 Playwright e2e tests for the Engineer Assistant Streamlit UI.

All tests in this file are auto-marked `e2e` and `slow` by conftest.py, so
they are excluded from normal `pytest -m "not slow"` CI runs.

Scope: UI structure only (page renders, navigation, input presence).
Agent responses are NOT tested here — those require real API keys and belong
to integration/manual tests.

Run locally:
    pip install -e ".[test]" && playwright install chromium
    pytest tests/e2e/ -v --headed    # watch in browser
    pytest tests/e2e/ -v             # headless
"""

import re

import pytest
from playwright.sync_api import Page, expect


@pytest.mark.e2e
class TestStreamlitUI:
    """Full-browser smoke tests for the Engineer Assistant UI."""

    # ── page identity ──────────────────────────────────────────────────────────

    def test_page_title(self, live_page: Page) -> None:
        """Browser tab title must contain 'Engineer'."""
        expect(live_page).to_have_title(re.compile("Engineer", re.IGNORECASE))

    def test_home_page_renders_heading(self, live_page: Page) -> None:
        """The home page must display at least one visible heading."""
        expect(live_page.get_by_role("heading").first).to_be_visible()

    # ── sidebar ────────────────────────────────────────────────────────────────

    def test_sidebar_new_chat_button_visible(self, live_page: Page) -> None:
        """The '+ New Chat' button must be visible in the sidebar."""
        sidebar = live_page.locator("[data-testid='stSidebar']")
        expect(sidebar.get_by_text("+ New Chat")).to_be_visible()

    # ── chat page ──────────────────────────────────────────────────────────────

    def test_navigate_to_chat_shows_input(self, live_page: Page) -> None:
        """Clicking the Chat nav link must reveal the chat input field."""
        # Click the Chat navigation link in the sidebar
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_placeholder(re.compile("ask me anything", re.IGNORECASE))
        ).to_be_visible()

    def test_chat_input_accepts_text(self, live_page: Page) -> None:
        """Typing into the chat input must update its value."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        chat_input = live_page.get_by_placeholder(
            re.compile("ask me anything", re.IGNORECASE)
        )
        chat_input.fill("hello world")
        expect(chat_input).to_have_value("hello world")

    def test_send_message_appears_in_history(self, live_page: Page) -> None:
        """Pressing Enter after typing must add a user message bubble to the chat."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        chat_input = live_page.get_by_placeholder(
            re.compile("ask me anything", re.IGNORECASE)
        )
        chat_input.fill("hello from playwright")
        chat_input.press("Enter")

        # Wait for the user message bubble to appear (agent response may be
        # skipped / errored with placeholder keys — that's acceptable here)
        expect(live_page.locator("[data-testid='stChatMessage']").first).to_be_visible(
            timeout=10_000
        )

    # ── settings page ──────────────────────────────────────────────────────────

    def test_navigate_to_settings_shows_heading(self, live_page: Page) -> None:
        """Clicking the Settings nav link must render a heading on that page."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(live_page.get_by_role("heading").first).to_be_visible()
