"""
Tests for web crawler module.

Tests the functionality of crawling websites for multi-page indexing.
"""

import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.tools.web_crawler import WebCrawler, crawl_website


class TestWebCrawler:
    """Test web crawler functionality."""

    def test_normalize_url_removes_fragment(self):
        """Test that URL fragments are removed."""
        crawler = WebCrawler()

        url = "https://example.com/page#section"
        normalized = crawler.normalize_url(url)

        assert normalized == "https://example.com/page"

    def test_normalize_url_removes_trailing_slash(self):
        """Test that trailing slashes are removed."""
        crawler = WebCrawler()

        assert (
            crawler.normalize_url("https://example.com/page/")
            == "https://example.com/page"
        )
        assert crawler.normalize_url("https://example.com/") == "https://example.com"

    def test_normalize_url_preserves_query_params(self):
        """Test that query parameters are preserved."""
        crawler = WebCrawler()

        url = "https://example.com/page?id=123&lang=en"
        normalized = crawler.normalize_url(url)

        assert normalized == "https://example.com/page?id=123&lang=en"

    def test_is_same_domain_returns_true_for_same_domain(self):
        """Test domain comparison for same domain."""
        crawler = WebCrawler()

        url1 = "https://example.com/page1"
        url2 = "https://example.com/page2"

        assert crawler.is_same_domain(url1, url2) is True

    def test_is_same_domain_returns_false_for_different_domain(self):
        """Test domain comparison for different domains."""
        crawler = WebCrawler()

        url1 = "https://example.com/page"
        url2 = "https://other.com/page"

        assert crawler.is_same_domain(url1, url2) is False

    def test_is_same_domain_handles_subdomains(self):
        """Test that subdomains are treated as different domains."""
        crawler = WebCrawler()

        url1 = "https://docs.example.com/page"
        url2 = "https://www.example.com/page"

        assert crawler.is_same_domain(url1, url2) is False

    def test_is_valid_url_accepts_http_and_https(self):
        """Test that HTTP and HTTPS URLs are valid."""
        crawler = WebCrawler()

        assert (
            crawler.is_valid_url("https://example.com/page", "https://example.com")
            is True
        )
        assert (
            crawler.is_valid_url("http://example.com/page", "http://example.com")
            is True
        )

    def test_is_valid_url_rejects_invalid_schemes(self):
        """Test that non-HTTP(S) URLs are rejected."""
        crawler = WebCrawler()

        assert (
            crawler.is_valid_url("ftp://example.com/file", "https://example.com")
            is False
        )
        assert (
            crawler.is_valid_url("mailto:test@example.com", "https://example.com")
            is False
        )

    def test_is_valid_url_skips_non_html_extensions(self):
        """Test that non-HTML file extensions are skipped."""
        crawler = WebCrawler()
        base = "https://example.com"

        # Should skip
        assert crawler.is_valid_url("https://example.com/image.jpg", base) is False
        assert crawler.is_valid_url("https://example.com/file.pdf", base) is False
        assert crawler.is_valid_url("https://example.com/style.css", base) is False
        assert crawler.is_valid_url("https://example.com/script.js", base) is False

        # Should accept
        assert crawler.is_valid_url("https://example.com/page.html", base) is True
        assert crawler.is_valid_url("https://example.com/page", base) is True

    def test_is_valid_url_enforces_same_domain_restriction(self):
        """Test that same_domain_only restriction is enforced."""
        crawler = WebCrawler(same_domain_only=True)
        base = "https://example.com"

        assert crawler.is_valid_url("https://example.com/page", base) is True
        assert crawler.is_valid_url("https://other.com/page", base) is False

    def test_is_valid_url_allows_cross_domain_when_disabled(self):
        """Test that cross-domain is allowed when same_domain_only=False."""
        crawler = WebCrawler(same_domain_only=False)
        base = "https://example.com"

        assert crawler.is_valid_url("https://example.com/page", base) is True
        assert crawler.is_valid_url("https://other.com/page", base) is True

    def test_extract_links_finds_all_links(self):
        """Test that all links are extracted from HTML."""
        crawler = WebCrawler()

        html = """
        <html>
            <a href="/page1">Page 1</a>
            <a href="/page2">Page 2</a>
            <a href="https://example.com/page3">Page 3</a>
        </html>
        """

        links = crawler.extract_links(html, "https://example.com")

        assert "https://example.com/page1" in links
        assert "https://example.com/page2" in links
        assert "https://example.com/page3" in links

    def test_extract_links_converts_relative_to_absolute(self):
        """Test that relative URLs are converted to absolute."""
        crawler = WebCrawler()

        html = '<html><a href="../parent/page">Link</a></html>'

        links = crawler.extract_links(html, "https://example.com/docs/current/")

        assert "https://example.com/docs/parent/page" in links

    def test_extract_links_filters_invalid_urls(self):
        """Test that invalid URLs are filtered out."""
        crawler = WebCrawler()

        html = """
        <html>
            <a href="/valid-page">Valid</a>
            <a href="/image.jpg">Image</a>
            <a href="javascript:void(0)">JavaScript</a>
            <a href="https://other.com/page">External</a>
        </html>
        """

        links = crawler.extract_links(html, "https://example.com")

        assert "https://example.com/valid-page" in links
        assert not any("jpg" in link for link in links)
        assert not any("javascript" in link for link in links)
        assert not any("other.com" in link for link in links)  # same_domain_only=True

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_single_page(self, mock_get):
        """Test crawling a single page."""
        # Mock HTTP response
        mock_response = Mock()
        mock_response.text = (
            "<html><head><title>Test Page</title></head><body>Content</body></html>"
        )
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response

        crawler = WebCrawler(max_pages=1, max_depth=0)
        pages = list(crawler.crawl("https://example.com"))

        assert len(pages) == 1
        assert pages[0]["url"] == "https://example.com"
        assert pages[0]["title"] == "Test Page"
        assert pages[0]["depth"] == 0
        assert "Content" in pages[0]["content"]

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_multiple_pages_with_links(self, mock_get):
        """Test crawling multiple linked pages."""

        # Mock responses for different pages
        def get_response(url, **_kwargs):
            mock_response = Mock()
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = Mock()

            if url == "https://example.com":
                mock_response.text = """
                <html>
                    <head><title>Home</title></head>
                    <body>
                        <a href="/page1">Page 1</a>
                        <a href="/page2">Page 2</a>
                    </body>
                </html>
                """
            elif url == "https://example.com/page1":
                mock_response.text = "<html><head><title>Page 1</title></head><body>Content 1</body></html>"
            elif url == "https://example.com/page2":
                mock_response.text = "<html><head><title>Page 2</title></head><body>Content 2</body></html>"

            return mock_response

        mock_get.side_effect = get_response

        crawler = WebCrawler(max_pages=10, max_depth=2)
        pages = list(crawler.crawl("https://example.com"))

        # Should crawl home + 2 linked pages
        assert len(pages) == 3
        titles = {p["title"] for p in pages}
        assert titles == {"Home", "Page 1", "Page 2"}

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_respects_max_pages(self, mock_get):
        """Test that max_pages limit is enforced."""
        # Mock response with many links
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = Mock()
        mock_response.text = """
        <html>
            <a href="/page1">1</a>
            <a href="/page2">2</a>
            <a href="/page3">3</a>
            <a href="/page4">4</a>
            <a href="/page5">5</a>
        </html>
        """
        mock_get.return_value = mock_response

        crawler = WebCrawler(max_pages=3, max_depth=2)
        pages = list(crawler.crawl("https://example.com"))

        # Should stop at max_pages
        assert len(pages) <= 3

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_respects_max_depth(self, mock_get):
        """Test that max_depth limit is enforced."""

        def get_response(url, **_kwargs):
            mock_response = Mock()
            mock_response.headers = {"Content-Type": "text/html"}
            mock_response.raise_for_status = Mock()
            # Each page links to a deeper page
            mock_response.text = f'<html><a href="{url}/next">Next</a></html>'
            return mock_response

        mock_get.side_effect = get_response

        crawler = WebCrawler(max_pages=100, max_depth=2)
        pages = list(crawler.crawl("https://example.com"))

        # Check no page exceeds max depth
        max_depth_found = max(p["depth"] for p in pages)
        assert max_depth_found <= 2

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_skips_non_html_content(self, mock_get):
        """Test that non-HTML content is skipped."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "application/json"}
        mock_response.raise_for_status = Mock()
        mock_response.text = '{"key": "value"}'
        mock_get.return_value = mock_response

        crawler = WebCrawler()
        pages = list(crawler.crawl("https://example.com/api/data"))

        # Should skip non-HTML
        assert len(pages) == 0

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_handles_http_errors_gracefully(self, mock_get):
        """Test that HTTP errors don't crash the crawler."""

        # First page succeeds, second fails
        def get_response(url, **_kwargs):
            if url == "https://example.com":
                mock_response = Mock()
                mock_response.headers = {"Content-Type": "text/html"}
                mock_response.raise_for_status = Mock()
                mock_response.text = '<html><a href="/broken">Link</a></html>'
                return mock_response
            # Simulate 404
            import requests

            error_msg = "404 Not Found"
            raise requests.HTTPError(error_msg)

        mock_get.side_effect = get_response

        crawler = WebCrawler(max_pages=10, max_depth=2)
        pages = list(crawler.crawl("https://example.com"))

        # Should successfully crawl the first page despite error on second
        assert len(pages) == 1

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_avoids_duplicate_visits(self, mock_get):
        """Test that pages are not visited twice."""
        # Mock response that links back to itself
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = Mock()
        mock_response.text = '<html><a href="/">Home</a></html>'
        mock_get.return_value = mock_response

        crawler = WebCrawler(max_pages=10, max_depth=3)
        pages = list(crawler.crawl("https://example.com"))

        # Should only visit once
        assert len(pages) == 1
        assert len(crawler.visited_urls) == 1

    def test_get_stats_returns_correct_stats(self):
        """Test that get_stats returns accurate statistics."""
        crawler = WebCrawler()
        crawler.pages_crawled = [
            {"url": "url1", "depth": 0, "size_bytes": 100},
            {"url": "url2", "depth": 1, "size_bytes": 200},
            {"url": "url3", "depth": 2, "size_bytes": 150},
        ]
        crawler.visited_urls = {"url1", "url2", "url3"}

        stats = crawler.get_stats()

        assert stats["pages_crawled"] == 3
        assert stats["total_size_bytes"] == 450
        assert stats["max_depth_reached"] == 2
        assert stats["urls_visited"] == 3

    @patch("src.tools.web_crawler.requests.get")
    def test_crawl_website_convenience_function(self, mock_get):
        """Test the convenience crawl_website function."""
        mock_response = Mock()
        mock_response.headers = {"Content-Type": "text/html"}
        mock_response.raise_for_status = Mock()
        mock_response.text = "<html><title>Test</title><body>Content</body></html>"
        mock_get.return_value = mock_response

        pages = crawl_website("https://example.com", max_pages=5, max_depth=1)

        assert len(pages) >= 1
        assert pages[0]["url"] == "https://example.com"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
