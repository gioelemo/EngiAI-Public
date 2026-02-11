"""
Shared RAG tool factory for MMORE-powered document retrieval.

Creates LangChain tools bound to a given MMOREClient instance.
Used by both RAGAgent and EngineeringAgent.
"""

import logging
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

from src.tools.mmore_client import MMOREClient, _report_progress
from src.tools.web_crawler import WebCrawler

logger = logging.getLogger(__name__)


def create_rag_tools(mmore_client: MMOREClient) -> list:
    """Create RAG tools bound to the given MMOREClient.

    Args:
        mmore_client: Initialized MMOREClient instance.

    Returns:
        List of LangChain tools for document search, add, list, delete, and URL ingestion.
    """
    return [
        _create_search_tool(mmore_client),
        _create_add_document_tool(mmore_client),
        _create_add_url_tool(mmore_client),
        _create_list_documents_tool(mmore_client),
        _create_delete_document_tool(mmore_client),
    ]


# ---------------------------------------------------------------------------
# Individual tool factories
# ---------------------------------------------------------------------------


def _create_search_tool(mmore_client: MMOREClient):
    """Create the search documents tool."""

    @tool
    def search_documents(
        query: Annotated[str, "The question to search for in documents"],
        num_results: Annotated[int, "Number of relevant documents to retrieve"] = 5,
    ) -> str:
        """
        Search through uploaded documents to find relevant information using MMORE.

        Use this tool to answer questions about papers, technical documents,
        or any previously uploaded files. MMORE provides advanced multimodal
        retrieval with support for images, tables, and complex layouts.
        """
        try:
            docs = mmore_client.retrieve(
                query=query,
                max_matches=num_results,
                min_similarity=0.3,
            )

            if not docs:
                return "No relevant documents found. Try uploading documents first or rephrase your query."

            response_parts = []
            for i, doc in enumerate(docs, 1):
                file_id = doc.metadata.get("source", "unknown")
                score = doc.metadata.get("score", 0.0)
                content = doc.page_content[:1500]

                response_parts.append(
                    f"**Result {i}** (relevance: {score:.2f})\n"
                    f"Source: {file_id}\n"
                    f"{content}...\n"
                )

            response = "\n\n".join(response_parts)
            response += f"\n\n📚 Found {len(docs)} relevant passage(s)"
        except Exception as e:
            logger.exception("Error searching documents with MMORE")
            return f"Error searching documents: {e}"
        else:
            return response

    return search_documents


def _create_add_document_tool(mmore_client: MMOREClient):
    """Create the add document tool."""

    @tool
    def add_document(
        file_path: Annotated[str, "Path to the document file to add"],
        file_id: Annotated[str, "Optional unique ID for the document"] = "",
    ) -> str:
        """
        Add a new document to the MMORE knowledge base.

        Supports PDF, Office docs, images, and more. MMORE automatically
        extracts text, images, tables, and other multimodal content.
        """
        try:
            path = Path(file_path)
            if not path.exists():
                return f"Error: File not found at {file_path}"

            if not file_id:
                file_id = path.stem

            mmore_client.upload_file(file_path=str(path), file_id=file_id)

            _report_progress(
                "complete",
                f"✓ Successfully indexed '{path.name}' - ready for queries!",
            )
        except Exception as e:
            logger.exception("Error adding document to MMORE")
            _report_progress("error", f"✗ Error adding document: {e}")
            return f"Error adding document: {e}"
        else:
            return (
                f"✓ Successfully added '{path.name}' to MMORE knowledge base\n"
                f"File ID: {file_id}\n"
                f"MMORE will process multimodal content (text, images, tables) automatically."
            )

    return add_document


def _create_add_url_tool(mmore_client: MMOREClient):  # noqa: PLR0915
    """Create the add URL tool."""

    @tool
    def add_url_to_knowledge_base(  # noqa: PLR0912, PLR0915
        url: Annotated[str, "The URL to download and add to the knowledge base"],
        crawl_subpages: Annotated[
            bool,
            "Whether to crawl and index all linked pages (default: True for documentation sites)",
        ] = True,
        max_pages: Annotated[
            int,
            "Maximum number of pages to crawl (default: 50)",
        ] = 50,
        file_id: Annotated[
            str,
            "Optional custom ID prefix for documents (auto-generated if not provided)",
        ] = "",
    ) -> str:
        """
        Download content from a URL and add it to the MMORE knowledge base.

        Supports:
        - GitHub documentation (automatically converts to raw URLs)
        - HTML pages with optional crawling of linked subpages
        - Markdown files
        - Any web-accessible document

        When crawl_subpages is True, will crawl all pages within the same domain
        up to max_pages, making it ideal for adding entire documentation sites.

        MMORE processes HTML by converting it to Markdown using markdownify,
        then extracts images and cleans the text.

        Use this when users want to add web documentation, GitHub docs, or
        online resources to the knowledge base.
        """
        try:
            download_url = url
            is_github_raw = False
            if "github.com" in url and "/blob/" in url:
                download_url = url.replace(
                    "github.com", "raw.githubusercontent.com"
                ).replace("/blob/", "/")
                logger.info(f"Converted GitHub URL to raw: {download_url}")
                is_github_raw = True
                crawl_subpages = False

            if crawl_subpages and not is_github_raw:
                logger.info(f"Crawling website starting from {url}...")
                _report_progress(
                    "crawl", f"Crawling website (max {max_pages} pages)..."
                )

                crawler = WebCrawler(max_pages=max_pages, max_depth=3)
                pages = list(crawler.crawl(url))

                if not pages:
                    return f"No pages found at {url}. The site may be inaccessible or contains no HTML content."

                uploaded_count = 0
                failed_count = 0
                temp_files = []

                try:
                    for i, page_data in enumerate(pages):
                        page_url = page_data["url"]
                        content = page_data["content"]

                        parsed_url = urlparse(page_url)
                        path_parts = Path(parsed_url.path)

                        page_file_id = file_id or parsed_url.netloc.replace(".", "_")
                        if path_parts.parts:
                            path_suffix = "_".join(
                                "".join(c for c in part if c.isalnum() or c in "_-")
                                for part in path_parts.parts
                                if part
                            )
                            if path_suffix:
                                page_file_id += f"_{path_suffix}"

                        if i > 0 or not file_id:
                            page_file_id += f"_{i}"

                        try:
                            with tempfile.NamedTemporaryFile(
                                mode="w",
                                suffix=".html",
                                delete=False,
                                encoding="utf-8",
                            ) as tmp_file:
                                tmp_file.write(content)
                                temp_path = tmp_file.name
                                temp_files.append(temp_path)

                            mmore_client.upload_file(
                                file_path=temp_path, file_id=page_file_id
                            )

                            uploaded_count += 1
                            _report_progress(
                                "upload",
                                f"Uploaded {uploaded_count}/{len(pages)} pages...",
                            )

                        except Exception as e:
                            logger.warning(f"Failed to upload {page_url}: {e}")
                            failed_count += 1

                finally:
                    for temp_path in temp_files:
                        Path(temp_path).unlink(missing_ok=True)

                _report_progress(
                    "complete",
                    f"✓ Successfully indexed {uploaded_count} pages - ready for queries!",
                )

                stats = crawler.get_stats()
                return (
                    f"✓ Successfully crawled and indexed website!\n"
                    f"Source: {url}\n"
                    f"Pages indexed: {uploaded_count}\n"
                    f"Failed: {failed_count}\n"
                    f"Total size: {stats['total_size_bytes'] / 1024:.1f} KB\n"
                    f"Max depth: {stats['max_depth_reached']}\n\n"
                    f"You can now ask questions about this documentation."
                )

            else:
                parsed_url = urlparse(download_url)
                path_parts = Path(parsed_url.path)
                extension = path_parts.suffix or ".html"

                if not file_id:
                    file_id = "".join(
                        c for c in path_parts.stem if c.isalnum() or c in "_-"
                    )
                    if not file_id:
                        file_id = parsed_url.netloc.replace(".", "_")

                logger.info(f"Downloading content from {download_url}...")
                _report_progress("download", "Downloading content from URL...")
                response = requests.get(download_url, timeout=30)
                response.raise_for_status()

                with tempfile.NamedTemporaryFile(
                    mode="wb", suffix=extension, delete=False
                ) as tmp_file:
                    tmp_file.write(response.content)
                    temp_path = tmp_file.name

                try:
                    logger.info(f"Uploading to MMORE with file_id: {file_id}...")
                    mmore_client.upload_file(file_path=temp_path, file_id=file_id)

                    _report_progress(
                        "complete",
                        "✓ Successfully indexed content from URL - ready for queries!",
                    )

                    return (
                        f"✓ Successfully added URL content to knowledge base!\n"
                        f"Source: {url}\n"
                        f"File ID: {file_id}\n"
                        f"Content size: {len(response.content)} bytes\n\n"
                        f"You can now ask questions about this document."
                    )
                finally:
                    Path(temp_path).unlink(missing_ok=True)

        except requests.HTTPError as e:
            logger.exception("HTTP error downloading URL")
            _report_progress("error", f"✗ HTTP error: {e.response.status_code}")
            return (
                f"Error downloading URL: {e.response.status_code} - {e.response.reason}"
            )
        except requests.RequestException as e:
            logger.exception("Error downloading URL")
            _report_progress("error", f"✗ Error downloading URL: {e}")
            return f"Error downloading URL: {e}"
        except Exception as e:
            logger.exception("Error adding URL to MMORE")
            _report_progress("error", f"✗ Error adding URL: {e}")
            return f"Error adding URL to knowledge base: {e}"

    return add_url_to_knowledge_base


def _create_list_documents_tool(mmore_client: MMOREClient):
    """Create the list documents tool."""

    @tool
    def list_documents() -> str:
        """
        List all documents uploaded to MMORE (across all sessions).

        Returns a summary of all documents in the knowledge base.
        """
        try:
            file_ids = mmore_client.list_files()

            if not file_ids:
                return "No documents in MMORE knowledge base yet.\n\nUse 'add_document' to upload PDF, Office, or image files."

            result = f"📚 MMORE Knowledge Base ({len(file_ids)} document(s)):\n\n"
            for fid in file_ids:
                result += f"• {fid}\n"

            result += "\n✓ All documents are indexed with multimodal content extraction"
            result += "\n✓ Documents persist across all chat sessions"
        except Exception as e:
            logger.exception("Error listing documents")
            return f"Error listing documents: {e}"
        else:
            return result

    return list_documents


def _create_delete_document_tool(mmore_client: MMOREClient):
    """Create the delete document tool."""

    @tool
    def delete_document(
        file_id: Annotated[str, "ID of the document to delete"],
    ) -> str:
        """
        Delete a document from the MMORE knowledge base.

        Use the file ID from the list_documents tool.
        """
        try:
            mmore_client.delete_file(file_id)
        except Exception as e:
            logger.exception("Error deleting document from MMORE")
            return f"Error deleting document: {e}"
        else:
            return f"✓ Deleted '{file_id}' from MMORE"

    return delete_document
