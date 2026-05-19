"""
Tier-3 Playwright e2e tests for the EngiAI Streamlit UI.

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

from __future__ import annotations

import contextlib
import re

import pytest

with contextlib.suppress(ImportError):
    from playwright.sync_api import Page, expect


@pytest.mark.e2e
class TestStreamlitUI:
    """Full-browser smoke tests for the EngiAI UI."""

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

    # ── settings: toggles and inputs ──────────────────────────────────────────────

    def test_settings_streaming_checkbox_is_checked_by_default(
        self, live_page: Page
    ) -> None:
        """The 'Enable streaming text display' checkbox must be checked by default."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "checkbox", name=re.compile("Enable streaming", re.IGNORECASE)
            )
        ).to_be_checked()

    def test_settings_toggle_streaming_checkbox(self, live_page: Page) -> None:
        """Unchecking 'Enable streaming text display' must update the checkbox state."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        checkbox = live_page.get_by_role(
            "checkbox", name=re.compile("Enable streaming", re.IGNORECASE)
        )
        expect(checkbox).to_be_checked()
        # Streamlit hides the native <input> behind a styled overlay (opacity:0,
        # position:absolute). force=True skips actionability checks but Playwright
        # still blocks when the element is clipped by an overflow container.
        # dispatch_event bypasses all viewport/scroll logic entirely.
        checkbox.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")
        expect(checkbox).not_to_be_checked()

    def test_settings_3d_material_selectbox_changes(self, live_page: Page) -> None:
        """Changing the 3D Viewer Material selectbox to 'flat' must update the value."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        material_sel = live_page.locator(
            "[data-testid='stSelectbox']", has_text="Material"
        )
        material_sel.click()
        live_page.get_by_role("option", name="flat").click()
        expect(material_sel).to_contain_text("flat")

    def test_settings_hpc_password_auth_reveals_hostname_input(
        self, live_page: Page
    ) -> None:
        """Switching to 'Password Authentication' must reveal the Hostname input."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        # Same hidden-input pattern as checkboxes — dispatch_event bypasses the
        # overflow-container viewport clip that force=True cannot work around.
        live_page.get_by_role(
            "radio", name=re.compile("Password Authentication", re.IGNORECASE)
        ).dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_placeholder(
                re.compile("cluster.example.com", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_settings_job_monitor_interval_shows_default(self, live_page: Page) -> None:
        """The 'Job Monitor Refresh' number input must default to 60 seconds."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        interval_input = live_page.locator(
            "[data-testid='stNumberInput']", has_text="Job Monitor Refresh"
        ).get_by_role("spinbutton")
        expect(interval_input).to_have_value("60")

    # ── settings: sections and info ────────────────────────────────────────────────

    def test_settings_slurm_section_heading_visible(self, live_page: Page) -> None:
        """Settings page must show the SLURM Cluster Configuration section heading."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "heading", name=re.compile("SLURM Cluster", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_settings_api_usage_section_visible(self, live_page: Page) -> None:
        """Settings page must show the API Usage Monitor section and refresh button."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "heading", name=re.compile("API Usage Monitor", re.IGNORECASE)
            )
        ).to_be_visible()
        expect(
            live_page.get_by_role(
                "button", name=re.compile("Refresh Usage Stats", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_settings_about_expander_shows_version(self, live_page: Page) -> None:
        """Expanding the 'About' section must reveal the app version string."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        about_expander = live_page.locator("[data-testid='stExpander']").filter(
            has_text="About"
        )
        about_expander.locator("summary").click()
        expect(
            about_expander.get_by_text(re.compile("Version", re.IGNORECASE))
        ).to_be_visible()

    # ── export flow ─────────────────────────────────────────────────────────────────

    def test_settings_export_buttons_disabled_with_no_messages(
        self, live_page: Page
    ) -> None:
        """Export buttons must be disabled when the active chat has no messages."""
        # Create a fresh empty chat so we are guaranteed 0 messages
        live_page.locator("[data-testid='stSidebar']").get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "button", name=re.compile("Export JSON", re.IGNORECASE)
            )
        ).to_be_disabled()

    def test_settings_export_json_triggers_download(self, live_page: Page) -> None:
        """Clicking 'Export JSON' after having messages must trigger a file download."""
        # Create a new chat and send one message to enable the export buttons
        live_page.locator("[data-testid='stSidebar']").get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        chat_input = live_page.get_by_placeholder(
            re.compile("ask me anything", re.IGNORECASE)
        )
        chat_input.fill("export test message")
        chat_input.press("Enter")
        expect(live_page.locator("[data-testid='stChatMessage']").first).to_be_visible(
            timeout=10_000
        )

        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        with live_page.expect_download() as dl_info:
            live_page.get_by_role(
                "button", name=re.compile("Export JSON", re.IGNORECASE)
            ).click()

        assert dl_info.value.suggested_filename == "chat_export.json"

    # ── conversation management ────────────────────────────────────────────────────

    def test_settings_clear_conversation_disables_exports(
        self, live_page: Page
    ) -> None:
        """Clicking 'Clear' empties messages so export buttons become disabled."""
        sidebar = live_page.locator("[data-testid='stSidebar']")

        # Create a new chat and send a message so there is something to clear
        sidebar.get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        # "+New Chat" auto-navigates to chat page; wait for the input
        chat_input = live_page.get_by_placeholder(
            re.compile("ask me anything", re.IGNORECASE)
        )
        expect(chat_input).to_be_visible(timeout=10_000)
        chat_input.fill("clear test message")
        chat_input.press("Enter")
        expect(live_page.locator("[data-testid='stChatMessage']").first).to_be_visible(
            timeout=10_000
        )

        # Navigate to settings
        sidebar.get_by_role("link", name=re.compile("settings", re.IGNORECASE)).click()
        live_page.wait_for_load_state("networkidle")

        # Verify Export JSON is currently enabled (chat has messages)
        export_btn = live_page.get_by_role(
            "button", name=re.compile("Export JSON", re.IGNORECASE)
        )
        expect(export_btn).to_be_enabled(timeout=10_000)

        # Click the conversation Clear button (name ends with "Clear")
        live_page.get_by_role("button", name=re.compile(r"Clear$")).click()
        live_page.wait_for_load_state("networkidle")

        # After clear + st.rerun(), messages are empty → export buttons disabled
        export_btn_after = live_page.get_by_role(
            "button", name=re.compile("Export JSON", re.IGNORECASE)
        )
        expect(export_btn_after).to_be_disabled(timeout=10_000)

    # ── sidebar: chat management ───────────────────────────────────────────────────

    def test_sidebar_chat_click_navigates_to_chat(self, live_page: Page) -> None:
        """Clicking a chat entry in the sidebar navigates to the chat page."""
        sidebar = live_page.locator("[data-testid='stSidebar']")

        # Navigate to settings so we are NOT on the chat page
        sidebar.get_by_role("link", name=re.compile("settings", re.IGNORECASE)).click()
        live_page.wait_for_load_state("networkidle")

        # Chat title buttons have key=f"chat_{chat_id}" which Streamlit exposes
        # as a CSS class st-key-chat_{chat_id}. Match any of them.
        chat_entry = sidebar.locator("[class*='st-key-chat_'] button").first
        expect(chat_entry).to_be_visible(timeout=10_000)
        chat_entry.scroll_into_view_if_needed()
        chat_entry.click()
        live_page.wait_for_load_state("networkidle")

        # Clicking the active chat entry triggers navigation to the chat page
        expect(
            live_page.get_by_placeholder(re.compile("ask me anything", re.IGNORECASE))
        ).to_be_visible(timeout=10_000)

    # ══════════════════════════════════════════════════════════════════════════════
    # Per-widget settings tests
    # ══════════════════════════════════════════════════════════════════════════════

    # ── 3D Viewer ─────────────────────────────────────────────────────────────────

    def test_settings_3d_color_picker_visible(self, live_page: Page) -> None:
        """The 3D color picker widget must be visible on the settings page."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(live_page.locator("[data-testid='stColorPicker']")).to_be_visible(
            timeout=10_000
        )

    def test_settings_3d_opacity_slider_default(self, live_page: Page) -> None:
        """The opacity slider must default to 1 (fully opaque)."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        slider_container = live_page.locator(
            "[data-testid='stSlider']", has_text="Opacity"
        )
        expect(slider_container).to_be_visible(timeout=10_000)
        expect(slider_container.get_by_role("slider")).to_have_attribute(
            "aria-valuenow", "1"
        )

    def test_settings_3d_auto_rotate_checkbox_default(self, live_page: Page) -> None:
        """The auto-rotate checkbox must be checked by default."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "checkbox", name=re.compile("Auto-rotate", re.IGNORECASE)
            )
        ).to_be_checked()

    # ── Chat & UI additional widgets ──────────────────────────────────────────────

    def test_settings_auto_monitor_jobs_default_unchecked(
        self, live_page: Page
    ) -> None:
        """'Automatically monitor submitted jobs' must be unchecked by default."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "checkbox",
                name=re.compile("Automatically monitor", re.IGNORECASE),
            )
        ).not_to_be_checked()

    def test_settings_export_md_disabled_with_no_messages(
        self, live_page: Page
    ) -> None:
        """Export MD button must be disabled when the active chat has no messages."""
        live_page.locator("[data-testid='stSidebar']").get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        # Wait for the new (empty) chat to fully load before navigating away
        expect(
            live_page.get_by_placeholder(re.compile("ask me anything", re.IGNORECASE))
        ).to_be_visible(timeout=10_000)

        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role("button", name=re.compile("Export MD", re.IGNORECASE))
        ).to_be_disabled(timeout=10_000)

    def test_settings_export_pdf_disabled_with_no_messages(
        self, live_page: Page
    ) -> None:
        """Export PDF button must be disabled when the active chat has no messages."""
        live_page.locator("[data-testid='stSidebar']").get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        # Wait for the new (empty) chat to fully load before navigating away
        expect(
            live_page.get_by_placeholder(re.compile("ask me anything", re.IGNORECASE))
        ).to_be_visible(timeout=10_000)

        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "button", name=re.compile("Export PDF", re.IGNORECASE)
            )
        ).to_be_disabled(timeout=10_000)

    # ── Voice Interaction ─────────────────────────────────────────────────────────

    def test_settings_voice_enabled_default_unchecked(self, live_page: Page) -> None:
        """'Enable voice features' must be unchecked by default."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "checkbox",
                name=re.compile("Enable voice features", re.IGNORECASE),
            )
        ).not_to_be_checked()

    def test_settings_voice_disabled_shows_info_message(self, live_page: Page) -> None:
        """When voice is disabled, an info message must prompt the user to enable it."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_text(
                re.compile(
                    "Enable voice features to access voice settings", re.IGNORECASE
                )
            )
        ).to_be_visible()

    def test_settings_voice_toggle_reveals_provider_radio(
        self, live_page: Page
    ) -> None:
        """Enabling voice features must reveal the voice provider radio buttons."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

        # The radio group label text is visible even though native inputs are hidden
        expect(
            live_page.locator("[data-testid='stRadio']", has_text="voice provider")
        ).to_be_visible(timeout=10_000)

        # Restore: uncheck voice to avoid affecting later tests
        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

    # ══════════════════════════════════════════════════════════════════════════════
    # Home page: deep content coverage (home.py)
    # ══════════════════════════════════════════════════════════════════════════════

    # ── agent grid ────────────────────────────────────────────────────────────────

    def test_home_page_agent_routing_section_visible(self, live_page: Page) -> None:
        """Home page must display the 'Intelligent LLM-Based Agent Routing' section."""
        expect(
            live_page.get_by_text(
                re.compile("Intelligent LLM-Based Agent Routing", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_home_page_engineering_agent_card_visible(self, live_page: Page) -> None:
        """Home page agent grid must show the Engineering agent card heading."""
        expect(
            live_page.get_by_role(
                "heading", name=re.compile(r"^.*Engineering.*$", re.IGNORECASE)
            ).first
        ).to_be_visible()

    def test_home_page_hpc_agent_card_visible(self, live_page: Page) -> None:
        """Home page agent grid must show the HPC agent card heading."""
        expect(
            live_page.get_by_role("heading", name=re.compile(r"\bHPC\b", re.IGNORECASE))
        ).to_be_visible()

    def test_home_page_arxiv_agent_card_visible(self, live_page: Page) -> None:
        """Home page agent grid must show the ArXiv agent card heading."""
        expect(
            live_page.get_by_role("heading", name=re.compile("ArXiv", re.IGNORECASE))
        ).to_be_visible()

    # ── getting started ────────────────────────────────────────────────────────────

    def test_home_page_getting_started_heading_visible(self, live_page: Page) -> None:
        """Home page must display the 'Getting Started' section heading."""
        expect(
            live_page.get_by_role(
                "heading", name=re.compile("Getting Started", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_home_page_example_queries_visible(self, live_page: Page) -> None:
        """Home page Getting Started section must list example queries."""
        expect(
            live_page.get_by_text(re.compile("Example queries", re.IGNORECASE))
        ).to_be_visible()

    # ── system info ────────────────────────────────────────────────────────────────

    def test_home_page_system_info_heading_visible(self, live_page: Page) -> None:
        """Home page must display the 'System Information' section heading."""
        expect(
            live_page.get_by_role(
                "heading", name=re.compile("System Information", re.IGNORECASE)
            )
        ).to_be_visible()

    def test_home_page_key_capabilities_visible(self, live_page: Page) -> None:
        """Home page system info must show the 'Key Capabilities' column."""
        expect(
            live_page.get_by_text(re.compile("Key Capabilities", re.IGNORECASE))
        ).to_be_visible()

    # ══════════════════════════════════════════════════════════════════════════════
    # Chat page: additional coverage (chat.py)
    # ══════════════════════════════════════════════════════════════════════════════

    def test_chat_page_welcome_caption_visible(self, live_page: Page) -> None:
        """Empty chat must show the welcome caption beneath the start heading."""
        live_page.locator("[data-testid='stSidebar']").get_by_text("+ New Chat").click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_text(
                re.compile("engineering design, optimization", re.IGNORECASE)
            )
        ).to_be_visible(timeout=10_000)

    def test_chat_page_voice_icon_absent_when_disabled(self, live_page: Page) -> None:
        """When voice is disabled (default), the chat input must not show the mic icon."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("chat", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        # With voice disabled the placeholder is the base text without " 🎤"
        chat_input = live_page.get_by_placeholder(
            re.compile("Ask me anything about engineering design", re.IGNORECASE)
        )
        expect(chat_input).to_be_visible()
        # Placeholder must NOT contain the microphone emoji added when voice is on
        expect(chat_input).not_to_have_attribute("placeholder", re.compile("🎤"))

    def test_settings_voice_toggle_reveals_microphone_checkbox(
        self, live_page: Page
    ) -> None:
        """Enabling voice must reveal the 'Enable microphone input' checkbox (checked)."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

        # Streamlit checkboxes have opacity:0, so to_be_visible() fails.
        # to_be_checked() confirms both existence and default state.
        mic_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable microphone input", re.IGNORECASE)
        )
        expect(mic_cb).to_be_checked(timeout=10_000)

        # Restore
        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

    def test_settings_voice_toggle_reveals_voice_responses_checkbox(
        self, live_page: Page
    ) -> None:
        """Enabling voice must reveal the 'Enable voice responses' checkbox (checked)."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

        resp_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice responses", re.IGNORECASE)
        )
        expect(resp_cb).to_be_checked(timeout=10_000)

        # Restore
        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

    def test_settings_voice_toggle_reveals_auto_play_checkbox(
        self, live_page: Page
    ) -> None:
        """Enabling voice must reveal the 'Auto-play responses' checkbox (checked)."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

        auto_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Auto-play responses", re.IGNORECASE)
        )
        expect(auto_cb).to_be_checked(timeout=10_000)

        # Restore
        voice_cb = live_page.get_by_role(
            "checkbox", name=re.compile("Enable voice features", re.IGNORECASE)
        )
        voice_cb.dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

    # ── Media Saving ──────────────────────────────────────────────────────────────

    def test_settings_media_save_dir_visible(self, live_page: Page) -> None:
        """The media save directory input must be visible and default to 'outputs'."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        save_dir = live_page.locator(
            "[data-testid='stTextInput']", has_text="Save directory"
        ).get_by_role("textbox")
        expect(save_dir).to_be_visible(timeout=10_000)
        expect(save_dir).to_have_value(re.compile(r"outputs$"))

    def test_settings_auto_save_media_default_unchecked(self, live_page: Page) -> None:
        """'Auto-save displayed media' must be unchecked by default."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        expect(
            live_page.get_by_role(
                "checkbox",
                name=re.compile("Auto-save displayed media", re.IGNORECASE),
            )
        ).not_to_be_checked()

    # ── File Management ───────────────────────────────────────────────────────────

    def test_settings_clear_output_folder_button_visible(self, live_page: Page) -> None:
        """The 'Clear Output Folder' button must be visible on settings."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        btn = live_page.get_by_role(
            "button", name=re.compile("Clear Output Folder", re.IGNORECASE)
        )
        btn.scroll_into_view_if_needed()
        expect(btn).to_be_visible()

    def test_settings_clear_artifact_folder_button_visible(
        self, live_page: Page
    ) -> None:
        """The 'Clear Artifact Folder' button must be visible on settings."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        btn = live_page.get_by_role(
            "button", name=re.compile("Clear Artifact Folder", re.IGNORECASE)
        )
        btn.scroll_into_view_if_needed()
        expect(btn).to_be_visible()

    # ── HPC Connection ────────────────────────────────────────────────────────────

    def test_settings_hpc_ssh_config_test_button_visible(self, live_page: Page) -> None:
        """In SSH Config mode (default), the 'Test SSH Config Connection' button must be visible."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        btn = live_page.get_by_role(
            "button",
            name=re.compile("Test SSH Config Connection", re.IGNORECASE),
        )
        btn.scroll_into_view_if_needed()
        expect(btn).to_be_visible()

    def test_settings_password_auth_credential_inputs_visible(
        self, live_page: Page
    ) -> None:
        """Switching to Password Auth must show Hostname, Username, Port, and Password."""
        live_page.locator("[data-testid='stSidebar']").get_by_role(
            "link", name=re.compile("settings", re.IGNORECASE)
        ).click()
        live_page.wait_for_load_state("networkidle")

        # Switch to Password Authentication
        live_page.get_by_role(
            "radio", name=re.compile("Password Authentication", re.IGNORECASE)
        ).dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

        # Hostname with default
        hostname = live_page.locator(
            "[data-testid='stTextInput']", has_text="Hostname"
        ).get_by_role("textbox")
        hostname.scroll_into_view_if_needed()
        expect(hostname).to_have_value("", timeout=10_000)

        # Username visible
        username = live_page.locator(
            "[data-testid='stTextInput']", has_text="Username"
        ).get_by_role("textbox")
        expect(username).to_be_visible()

        # Port default 22
        port = live_page.locator(
            "[data-testid='stNumberInput']", has_text="Port"
        ).get_by_role("spinbutton")
        expect(port).to_have_value("22")

        # Password visible
        password = live_page.locator(
            "[data-testid='stTextInput']", has_text="Password"
        ).get_by_role("textbox")
        expect(password).to_be_visible()

        # Both action buttons visible (Save + Test Connection)
        save_btn = live_page.get_by_role(
            "button", name=re.compile("Save Credentials", re.IGNORECASE)
        )
        save_btn.scroll_into_view_if_needed()
        expect(save_btn).to_be_visible()
        test_btn = live_page.get_by_role(
            "button", name=re.compile("Test Connection", re.IGNORECASE)
        ).first
        expect(test_btn).to_be_visible()

        # Restore to SSH Config mode
        live_page.get_by_role(
            "radio", name=re.compile("SSH Config", re.IGNORECASE)
        ).dispatch_event("click")
        live_page.wait_for_load_state("networkidle")

    # ── Sidebar widgets ───────────────────────────────────────────────────────────
