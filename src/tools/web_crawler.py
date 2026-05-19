"""
Web crawler for extracting multiple pages from documentation sites.

This module crawls websites starting from a root URL and extracts all pages
within the same domain. Each page is then uploaded to MMORE for HTML processing.
"""

import logging
from collections.abc import Iterator
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


class WebCrawler:
    """Crawls websites and extracts HTML pages for indexing."""

    def __init__(
        self,
        max_pages: int = 100,
        max_depth: int = 3,
        same_domain_only: bool = True,
        timeout: int = 30,
    ):
        """
        Initialize web crawler.

        Args:
            max_pages: Maximum number of pages to crawl
            max_depth: Maximum depth to crawl from root URL
            same_domain_only: Only crawl pages on the same domain
            timeout: HTTP request timeout in seconds
        """
        self.max_pages = max_pages
        self.max_depth = max_depth
        self.same_domain_only = same_domain_only
        self.timeout = timeout

        # Tracking
        self.visited_urls: set[str] = set()
        self.pages_crawled: list[dict[str, Any]] = []

    def normalize_url(self, url: str) -> str:
        """
        Normalize URL by removing fragments and trailing slashes.

        Args:
            url: URL to normalize

        Returns:
            Normalized URL
        """
        parsed = urlparse(url)
        # Remove fragment (#section) and rebuild without trailing slash
        normalized = f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        if parsed.query:
            normalized += f"?{parsed.query}"

        # Remove trailing slash (even for root, to normalize https://example.com/ -> https://example.com)
        normalized = normalized.rstrip("/")

        return normalized

    def is_same_domain(self, url1: str, url2: str) -> bool:
        """
        Check if two URLs are on the same domain.

        Args:
            url1: First URL
            url2: Second URL

        Returns:
            True if same domain, False otherwise
        """
        domain1 = urlparse(url1).netloc
        domain2 = urlparse(url2).netloc
        return domain1 == domain2

    def is_valid_url(self, url: str, base_domain: str) -> bool:
        """
        Check if URL should be crawled.

        Args:
            url: URL to check
            base_domain: Base domain for crawling

        Returns:
            True if URL should be crawled, False otherwise
        """
        parsed = urlparse(url)

        # Must have http or https scheme
        if parsed.scheme not in ("http", "https"):
            return False

        # Skip common non-HTML resources
        skip_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
            ".gif",
            ".pdf",
            ".zip",
            ".tar",
            ".gz",
            ".mp4",
            ".mp3",
            ".css",
            ".js",
            ".json",
            ".xml",
            ".svg",
            ".ico",
        }
        path_lower = parsed.path.lower()
        if any(path_lower.endswith(ext) for ext in skip_extensions):
            return False

        # Check domain restriction
        if self.same_domain_only:
            return self.is_same_domain(url, base_domain)

        return True

    def extract_links(self, html_content: str, base_url: str) -> list[str]:
        """
        Extract all valid links from HTML content.

        Args:
            html_content: HTML content as string
            base_url: Base URL for resolving relative links

        Returns:
            List of absolute URLs
        """
        soup = BeautifulSoup(html_content, "html.parser")
        links = []

        # Find all <a> tags with href
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # BeautifulSoup can return list for multi-valued attributes, take first
            if isinstance(href, list):
                href = href[0] if href else ""
            href = str(href)

            # Convert relative URLs to absolute
            absolute_url = urljoin(base_url, href)

            # Normalize and validate
            normalized_url = self.normalize_url(absolute_url)

            if self.is_valid_url(normalized_url, base_url):
                links.append(normalized_url)

        return links

    def crawl(self, start_url: str) -> Iterator[dict[str, Any]]:
        """
        Crawl website starting from a URL.

        Yields pages as they are crawled for immediate processing.

        Args:
            start_url: URL to start crawling from

        Yields:
            Dictionary with page information:
                - url: Page URL
                - content: HTML content
                - title: Page title (if available)
                - depth: Crawl depth
        """
        start_url = self.normalize_url(start_url)
        base_domain = start_url

        # Queue stores tuples of (url, depth) for BFS traversal
        queue: list[tuple[str, int]] = [(start_url, 0)]

        while queue and len(self.visited_urls) < self.max_pages:
            url, depth = queue.pop(0)

            # Skip if already visited or too deep
            if url in self.visited_urls or depth > self.max_depth:
                continue

            # Mark as visited
            self.visited_urls.add(url)

            try:
                logger.info(f"Crawling [{depth}]: {url}")

                # Fetch page
                response = requests.get(url, timeout=self.timeout)
                response.raise_for_status()

                # Only process HTML content
                content_type = response.headers.get("Content-Type", "").lower()
                if "text/html" not in content_type:
                    logger.debug(f"Skipping non-HTML content: {content_type}")
                    continue

                html_content = response.text

                # Extract title
                soup = BeautifulSoup(html_content, "html.parser")
                title = (
                    soup.title.string.strip()
                    if soup.title and soup.title.string
                    else None
                )

                # Yield page data
                page_data = {
                    "url": url,
                    "content": html_content,
                    "title": title,
                    "depth": depth,
                    "size_bytes": len(html_content),
                }
                self.pages_crawled.append(page_data)
                yield page_data

                # Extract links for further crawling (if not at max depth)
                if depth < self.max_depth:
                    links = self.extract_links(html_content, base_domain)

                    # Add unvisited links to queue
                    new_links = [
                        (link, depth + 1)
                        for link in links
                        if link not in self.visited_urls
                    ]
                    queue.extend(new_links)

                    logger.debug(f"Found {len(links)} links on {url}")

            except requests.RequestException as e:
                logger.warning(f"Failed to fetch {url}: {e}")
                continue
            except Exception:
                logger.exception(f"Error processing {url}")
                continue

    def get_stats(self) -> dict[str, Any]:
        """
        Get crawling statistics.

        Returns:
            Dictionary with crawling stats
        """
        return {
            "pages_crawled": len(self.pages_crawled),
            "total_size_bytes": sum(p["size_bytes"] for p in self.pages_crawled),
            "max_depth_reached": (
                max(p["depth"] for p in self.pages_crawled) if self.pages_crawled else 0
            ),
            "urls_visited": len(self.visited_urls),
        }


def crawl_website(
    url: str,
    max_pages: int = 100,
    max_depth: int = 3,
) -> list[dict[str, Any]]:
    """
    Convenience function to crawl a website and return all pages.

    Args:
        url: Starting URL
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth to crawl

    Returns:
        List of page dictionaries
    """
    crawler = WebCrawler(max_pages=max_pages, max_depth=max_depth)
    pages = list(crawler.crawl(url))

    logger.info(f"Crawling complete. Stats: {crawler.get_stats()}")

    return pages
