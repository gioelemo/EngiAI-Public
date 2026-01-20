"""
RAG Agent for document-based question answering.

This agent handles queries about uploaded documents using the RAG system.
Now powered by MMORE for advanced multimodal document processing.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import Annotated
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.tools import MMOREClient
from src.tools.mmore_client import _report_progress
from src.tools.web_crawler import WebCrawler
from src.utils.prompts import RAG_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Agent for document-based question answering using MMORE RAG."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        mmore_url: str | None = None,
    ):
        """Initialize the RAG agent with MMORE client.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            mmore_url: URL of MMORE service (defaults to MMORE_RAG_URL env var)
        """
        # Check if MMORE should be skipped (e.g., during benchmarks)
        skip_mmore = os.getenv("SKIP_MMORE", "false").lower() == "true"

        if skip_mmore:
            logger.info("SKIP_MMORE=true: RAG Agent initialized without MMORE client")
            self.mmore_client: MMOREClient | None = None
        else:
            # Initialize MMORE client
            self.mmore_client = MMOREClient(base_url=mmore_url)

        super().__init__(model_name=model_name, temperature=temperature)

        # Verify MMORE connection if client was initialized
        if not skip_mmore and self.mmore_client:
            if self.mmore_client.health_check():
                logger.info("RAG Agent initialized with MMORE service")
            else:
                logger.warning(
                    "MMORE service not reachable - some features may not work"
                )

    def _create_tools(self) -> list:
        """Create LangChain tools for the RAG agent."""
        # If MMORE is skipped, return empty tools list
        if self.mmore_client is None:
            return []

        return [
            self._create_search_tool(),
            self._create_add_document_tool(),
            self._create_add_url_tool(),
            self._create_list_documents_tool(),
            self._create_delete_document_tool(),
        ]

    def _create_search_tool(self):
        """Create the search documents tool using MMORE."""

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
            assert self.mmore_client is not None, "MMORE client not initialized"
            try:
                # Retrieve documents from MMORE
                docs = self.mmore_client.retrieve(
                    query=query,
                    max_matches=num_results,
                    min_similarity=0.3,  # Filter low-quality matches
                )

                if not docs:
                    return "No relevant documents found. Try uploading documents first or rephrase your query."

                # Format results with context
                response_parts = []
                for i, doc in enumerate(docs, 1):
                    file_id = doc.metadata.get("source", "unknown")
                    score = doc.metadata.get("score", 0.0)
                    content = doc.page_content[:500]  # Limit content length

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

    def _create_add_document_tool(self):
        """Create the add document tool using MMORE."""

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
            assert self.mmore_client is not None, "MMORE client not initialized"
            try:
                path = Path(file_path)
                if not path.exists():
                    return f"Error: File not found at {file_path}"

                # Generate file_id from filename if not provided
                if not file_id:
                    file_id = path.stem

                # Upload to MMORE
                self.mmore_client.upload_file(file_path=str(path), file_id=file_id)

                # Report completion
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

    def _create_add_url_tool(self):  # noqa: PLR0915
        """Create the add URL tool using MMORE."""

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
            assert self.mmore_client is not None, "MMORE client not initialized"
            try:
                # Convert GitHub URLs to raw URLs if needed
                download_url = url
                is_github_raw = False
                if "github.com" in url and "/blob/" in url:
                    download_url = url.replace(
                        "github.com", "raw.githubusercontent.com"
                    ).replace("/blob/", "/")
                    logger.info(f"Converted GitHub URL to raw: {download_url}")
                    is_github_raw = True
                    # Disable crawling for single GitHub files
                    crawl_subpages = False

                # Check if we should crawl multiple pages
                if crawl_subpages and not is_github_raw:
                    logger.info(f"Crawling website starting from {url}...")
                    _report_progress(
                        "crawl", f"Crawling website (max {max_pages} pages)..."
                    )

                    # Use web crawler to get all pages
                    crawler = WebCrawler(max_pages=max_pages, max_depth=3)
                    pages = list(crawler.crawl(url))

                    if not pages:
                        return f"No pages found at {url}. The site may be inaccessible or contains no HTML content."

                    # Process and upload each page
                    uploaded_count = 0
                    failed_count = 0
                    temp_files = []

                    try:
                        for i, page_data in enumerate(pages):
                            page_url = page_data["url"]
                            content = page_data["content"]

                            # Generate unique file ID for this page
                            parsed_url = urlparse(page_url)
                            path_parts = Path(parsed_url.path)

                            # Create a unique file ID based on URL path
                            page_file_id = file_id or parsed_url.netloc.replace(
                                ".", "_"
                            )
                            if path_parts.parts:
                                # Add path to file ID
                                path_suffix = "_".join(
                                    "".join(c for c in part if c.isalnum() or c in "_-")
                                    for part in path_parts.parts
                                    if part
                                )
                                if path_suffix:
                                    page_file_id += f"_{path_suffix}"

                            # Ensure unique file ID
                            if i > 0 or not file_id:
                                page_file_id += f"_{i}"

                            try:
                                # Save to temporary file
                                with tempfile.NamedTemporaryFile(
                                    mode="w",
                                    suffix=".html",
                                    delete=False,
                                    encoding="utf-8",
                                ) as tmp_file:
                                    tmp_file.write(content)
                                    temp_path = tmp_file.name
                                    temp_files.append(temp_path)

                                # Upload to MMORE
                                self.mmore_client.upload_file(
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
                        # Clean up all temporary files
                        for temp_path in temp_files:
                            Path(temp_path).unlink(missing_ok=True)

                    # Report completion
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
                    # Single page mode (original behavior)
                    parsed_url = urlparse(download_url)
                    path_parts = Path(parsed_url.path)
                    extension = path_parts.suffix or ".html"

                    # Generate file_id if not provided
                    if not file_id:
                        file_id = "".join(
                            c for c in path_parts.stem if c.isalnum() or c in "_-"
                        )
                        if not file_id:
                            file_id = parsed_url.netloc.replace(".", "_")

                    # Download content
                    logger.info(f"Downloading content from {download_url}...")
                    _report_progress("download", "Downloading content from URL...")
                    response = requests.get(download_url, timeout=30)
                    response.raise_for_status()

                    # Save to temporary file
                    with tempfile.NamedTemporaryFile(
                        mode="wb", suffix=extension, delete=False
                    ) as tmp_file:
                        tmp_file.write(response.content)
                        temp_path = tmp_file.name

                    try:
                        # Upload to MMORE
                        logger.info(f"Uploading to MMORE with file_id: {file_id}...")
                        self.mmore_client.upload_file(
                            file_path=temp_path, file_id=file_id
                        )

                        # Report completion
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
                        # Clean up temporary file
                        Path(temp_path).unlink(missing_ok=True)

            except requests.HTTPError as e:
                logger.exception("HTTP error downloading URL")
                _report_progress("error", f"✗ HTTP error: {e.response.status_code}")
                return f"Error downloading URL: {e.response.status_code} - {e.response.reason}"
            except requests.RequestException as e:
                logger.exception("Error downloading URL")
                _report_progress("error", f"✗ Error downloading URL: {e}")
                return f"Error downloading URL: {e}"
            except Exception as e:
                logger.exception("Error adding URL to MMORE")
                _report_progress("error", f"✗ Error adding URL: {e}")
                return f"Error adding URL to knowledge base: {e}"

        return add_url_to_knowledge_base

    def _create_list_documents_tool(self):
        """Create the list documents tool."""

        @tool
        def list_documents() -> str:
            """
            List all documents uploaded to MMORE (across all sessions).

            Returns a summary of all documents in the knowledge base.
            """
            assert self.mmore_client is not None, "MMORE client not initialized"
            try:
                # Get all files directly from MMORE API
                file_ids = self.mmore_client.list_files()

                if not file_ids:
                    return "No documents in MMORE knowledge base yet.\n\nUse 'add_document' to upload PDF, Office, or image files."

                # Format output
                result = f"📚 MMORE Knowledge Base ({len(file_ids)} document(s)):\n\n"
                for file_id in file_ids:
                    result += f"• {file_id}\n"

                result += (
                    "\n✓ All documents are indexed with multimodal content extraction"
                )
                result += "\n✓ Documents persist across all chat sessions"
            except Exception as e:
                logger.exception("Error listing documents")
                return f"Error listing documents: {e}"
            else:
                return result

        return list_documents

    def _create_delete_document_tool(self):
        """Create the delete document tool."""

        @tool
        def delete_document(
            file_id: Annotated[str, "ID of the document to delete"],
        ) -> str:
            """
            Delete a document from the MMORE knowledge base.

            Use the file ID from the list_documents tool.
            """
            assert self.mmore_client is not None, "MMORE client not initialized"
            try:
                # Delete from MMORE
                self.mmore_client.delete_file(file_id)
            except Exception as e:
                logger.exception("Error deleting document from MMORE")
                return f"Error deleting document: {e}"
            else:
                return f"✓ Deleted '{file_id}' from MMORE"

        return delete_document

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the RAG agent.

        Returns:
            System prompt string
        """
        return RAG_AGENT_SYSTEM_PROMPT
