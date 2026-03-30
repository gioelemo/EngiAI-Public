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

    # ── chat page layout ───────────────────────────────────────────────────────────

    def test_chat_page_shows_whiteboard_heading(self, live_page: Page) -> None:
        """Chat page must display the Whiteboard column heading."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "heading", name=re.compile("Whiteboard", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_empty_chat_shows_welcome_heading(self, live_page: Page) -> None:
        """A freshly created chat must show the 'Start a conversation' welcome heading."""
        # Create a new chat so we start with an empty message list
        live_page.locator("[data-testid='stSidebar']").get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "heading", name=re.compile("Start a conversation", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_whiteboard_send_button_visible(self, live_page: Page) -> None:
        """The 'Send to Chat' button must be visible on the chat page."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "button", name=re.compile("Send to Chat", re.IGNORECASE)
            )
        ).to_be_visible()

    # ── whiteboard / excalidraw ────────────────────────────────────────────────────

    def test_whiteboard_iframe_loads_with_canvas(self, live_page: Page) -> None:
        """Excalidraw component must load an iframe containing a visible <canvas>."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        # Target the Excalidraw iframe specifically (the chat page also contains a
        # streamlit_adjustable_columns iframe, so "iframe" would be ambiguous).
        canvas = (
            live_page.frame_locator("iframe[title*='excalidraw_whiteboard']")
            .locator("canvas")
            .first
        )
        expect(canvas).to_be_visible(timeout=15_000)

    def test_whiteboard_message_input_accepts_text(self, live_page: Page) -> None:
        """The whiteboard message input must accept typed text."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        msg_input = live_page.get_by_placeholder(
            re.compile("Describe your drawing", re.IGNORECASE)
        )
        msg_input.fill("describe this truss structure")
        expect(msg_input).to_have_value("describe this truss structure")

    def test_send_to_chat_button_triggers_page_update(self, live_page: Page) -> None:
        """Clicking 'Send to Chat' must trigger a page rerun and keep the UI responsive.

        Note: the 'Canvas is empty' warning path is not tested here because the
        Excalidraw component returns truthy data (empty-but-valid JSON) even for a
        blank canvas, so the warning branch is never reached through the UI.
        """
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        live_page.get_by_role(
            "button", name=re.compile("Send to Chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        # After the rerun the chat input must still be present — page didn't crash
        expect(
            live_page.get_by_placeholder(re.compile("ask me anything", re.IGNORECASE))
        ).to_be_visible(timeout=10_000)

    # ── sidebar management ────────────────────────────────────────────────────────

    def test_new_chat_creates_sidebar_entry(self, live_page: Page) -> None:
        """After clicking '+ New Chat', the sidebar must show a chat entry."""
        sidebar = live_page.locator("[data-testid='stSidebar']")
        sidebar.get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        # The active newly-created chat always appears in the sidebar history
        expect(
            sidebar.get_by_role(
                "button", name=re.compile("New Chat", re.IGNORECASE)
            ).first
        ).to_be_visible()

    # ── navigation + settings ─────────────────────────────────────────────────────

    def test_navigate_to_wandb_report(self, live_page: Page) -> None:
        """Clicking the W&B Report nav link must render the page."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("W&B", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(live_page.get_by_role("heading").first).to_be_visible()

    def test_settings_shows_voice_section(self, live_page: Page) -> None:
        """Settings page must display the Voice Interaction section."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_text(re.compile("Voice Interaction", re.IGNORECASE))
        ).to_be_visible()
