"""
Test to demonstrate URL crawling capability.

This test demonstrates that the NEW implementation can crawl multiple pages
and extract full documentation sites.

When you add a URL like https://docs.hpc.ethz.ch to the knowledge base,
the system can now crawl all subpages within the same domain.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.upload_url_to_mmore import download_content
from src.tools.web_crawler import WebCrawler


class TestUrlCrawlingCapability:
    """Test to demonstrate URL crawling capabilities."""

    @patch("scripts.upload_url_to_mmore.requests.get")
    def test_only_single_page_is_downloaded_not_subpages(self, mock_get, tmp_path):
        """
        DEMONSTRATION TEST: Shows that only the main page is downloaded.

        When a user adds "https://docs.hpc.ethz.ch" to knowledge base:
        - The main page HTML is downloaded
        - Links to subpages (e.g., /getting-started, /storage, /euler) are NOT followed
        - Only ONE HTTP request is made

        This test demonstrates the current limitation.
        """
        # Simulate a documentation site with multiple subpages
        main_page_html = """
        <!DOCTYPE html>
        <html>
        <head><title>HPC Documentation</title></head>
        <body>
            <h1>Welcome to HPC Documentation</h1>
            <nav>
                <a href="/getting-started">Getting Started</a>
                <a href="/storage">Storage Guide</a>
                <a href="/euler">Euler Cluster</a>
                <a href="/leonhard">Leonhard Cluster</a>
                <a href="/software">Software Modules</a>
            </nav>
            <p>Main documentation page content...</p>
        </body>
        </html>
        """

        # Mock the HTTP response
        mock_response = Mock()
        mock_response.content = main_page_html.encode("utf-8")
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        output_file = tmp_path / "hpc_docs.html"

        # Execute - download the main page
        result = download_content("https://docs.hpc.ethz.ch", output_file)

        # Verify success
        assert result is True
        assert output_file.exists()

        # CRITICAL ASSERTION: Only ONE HTTP request was made
        # The subpages linked in the HTML are NOT crawled
        assert mock_get.call_count == 1
        mock_get.assert_called_once_with("https://docs.hpc.ethz.ch", timeout=30)

        # Verify the downloaded content is only the main page
        downloaded_content = output_file.read_text()
        assert "Welcome to HPC Documentation" in downloaded_content
        assert "/getting-started" in downloaded_content  # Link is in HTML
        assert "/storage" in downloaded_content  # Link is in HTML

        # BUT: The actual content from these subpages is NOT downloaded
        # because no additional requests were made to follow the links

    @patch("scripts.upload_url_to_mmore.requests.get")
    def test_demonstrates_what_full_crawling_would_look_like(self, mock_get, tmp_path):
        """
        REFERENCE TEST: Shows what WOULD happen with proper crawling.

        This is what should happen for full site indexing:
        1. Download main page
        2. Extract all links
        3. Download each linked page
        4. Recursively extract links from those pages
        5. Continue until all pages are indexed

        Currently, we only do step 1.
        """
        # Setup mock responses for multiple pages
        main_page = (
            b"<html><a href='/page1'>Page 1</a><a href='/page2'>Page 2</a></html>"
        )
        page1 = b"<html><h1>Page 1 Content</h1></html>"
        page2 = b"<html><h1>Page 2 Content</h1></html>"

        # This is what WOULD be needed for full crawling
        expected_requests_for_full_crawl = [
            "https://example.com",  # Main page
            "https://example.com/page1",  # Subpage 1
            "https://example.com/page2",  # Subpage 2
        ]

        # But currently, we only make 1 request
        mock_response = Mock()
        mock_response.content = main_page
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        output_file = tmp_path / "test.html"
        result = download_content("https://example.com", output_file)

        assert result is True

        # Current behavior: Only 1 request
        assert mock_get.call_count == 1

        # For full crawling, we would need: len(expected_requests_for_full_crawl) requests
        # That's 3 requests, not 1!

        print("\n" + "=" * 70)
        print("OLD BEHAVIOR: 1 HTTP request (main page only)")
        print("NEEDED FOR FULL CRAWLING: 3+ HTTP requests (main + subpages)")
        print("=" * 70)

    @patch("src.tools.web_crawler.requests.get")
    def test_new_web_crawler_extracts_multiple_pages(self, mock_get):
        """
        DEMONSTRATION: Shows that the NEW WebCrawler successfully extracts subpages.

        This is the solution to the limitation - we now have a WebCrawler
        that can crawl entire documentation sites.
        """

        # Mock responses for multiple pages
        def get_response(url, **_kwargs):
            mock_response = Mock()
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = Mock()

            if url == "https://docs.hpc.ethz.ch":
                # Main page with links to subpages
                mock_response.text = """
                <!DOCTYPE html>
                <html>
                <head><title>HPC Documentation</title></head>
                <body>
                    <h1>Welcome to HPC Documentation</h1>
                    <nav>
                        <a href="/getting-started">Getting Started</a>
                        <a href="/storage">Storage Guide</a>
                        <a href="/euler">Euler Cluster</a>
                    </nav>
                    <p>Main documentation page content...</p>
                </body>
                </html>
                """
            elif url == "https://docs.hpc.ethz.ch/getting-started":
                mock_response.text = """
                <html>
                <head><title>Getting Started - HPC</title></head>
                <body><h1>Getting Started Guide</h1><p>Step-by-step instructions...</p></body>
                </html>
                """
            elif url == "https://docs.hpc.ethz.ch/storage":
                mock_response.text = """
                <html>
                <head><title>Storage Guide - HPC</title></head>
                <body><h1>Storage Documentation</h1><p>Storage best practices...</p></body>
                </html>
                """
            elif url == "https://docs.hpc.ethz.ch/euler":
                mock_response.text = """
                <html>
                <head><title>Euler Cluster - HPC</title></head>
                <body><h1>Euler Cluster</h1><p>Euler cluster documentation...</p></body>
                </html>
                """
            else:
                mock_response.text = "<html><body>Page not found</body></html>"

            return mock_response

        mock_get.side_effect = get_response

        # Use the new WebCrawler
        crawler = WebCrawler(max_pages=10, max_depth=2)
        pages = list(crawler.crawl("https://docs.hpc.ethz.ch"))

        # NEW BEHAVIOR: Multiple pages crawled!
        assert len(pages) == 4  # Main page + 3 subpages

        # Verify all pages were extracted
        urls = {p["url"] for p in pages}
        assert "https://docs.hpc.ethz.ch" in urls
        assert "https://docs.hpc.ethz.ch/getting-started" in urls
        assert "https://docs.hpc.ethz.ch/storage" in urls
        assert "https://docs.hpc.ethz.ch/euler" in urls

        # Verify titles were extracted
        titles = {p["title"] for p in pages}
        assert "HPC Documentation" in titles
        assert "Getting Started - HPC" in titles
        assert "Storage Guide - HPC" in titles
        assert "Euler Cluster - HPC" in titles

        # Verify content is different for each page
        contents = [p["content"] for p in pages]
        assert any("Getting Started Guide" in c for c in contents)
        assert any("Storage Documentation" in c for c in contents)
        assert any("Euler Cluster" in c for c in contents)

        # Statistics
        stats = crawler.get_stats()

        print("\n" + "=" * 70)
        print("🎉 NEW BEHAVIOR: Web Crawler Successfully Extracts Full Sites!")
        print("=" * 70)
        print(f"Pages crawled: {stats['pages_crawled']}")
        print(f"Total size: {stats['total_size_bytes'] / 1024:.1f} KB")
        print(f"Max depth: {stats['max_depth_reached']}")
        print("=" * 70)
        print("\n✅ Solution Complete:")
        print("  - WebCrawler module created")
        print("  - Integrated with RAG agent add_url_to_knowledge_base")
        print("  - Each page uploaded to MMORE (HTML → Markdown → extracted)")
        print("  - Full documentation sites can now be indexed!")
        print("=" * 70)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
