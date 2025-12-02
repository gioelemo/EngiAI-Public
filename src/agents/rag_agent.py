"""
RAG Agent for document-based question answering.

This agent handles queries about uploaded documents using the RAG system.
Now powered by MMORE for advanced multimodal document processing.
"""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated
from urllib.parse import urlparse

import requests
from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.tools import MMOREClient

if TYPE_CHECKING:
    pass

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
        # Initialize MMORE client
        self.mmore_client = MMOREClient(base_url=mmore_url)

        super().__init__(model_name=model_name, temperature=temperature)

        # Verify MMORE connection
        if self.mmore_client.health_check():
            logger.info("RAG Agent initialized with MMORE service")
        else:
            logger.warning("MMORE service not reachable - some features may not work")

    @property
    def db(self):
        """Lazy-load database manager to avoid circular import."""
        if not hasattr(self, "_db"):
            from src.ui.database import DatabaseManager  # noqa: PLC0415

            self._db = DatabaseManager()
        return self._db

    @db.setter
    def db(self, value):
        """Allow setting database manager (useful for testing)."""
        self._db = value

    def _create_tools(self) -> list:
        """Create LangChain tools for the RAG agent."""
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
            try:
                path = Path(file_path)
                if not path.exists():
                    return f"Error: File not found at {file_path}"

                # Generate file_id from filename if not provided
                if not file_id:
                    file_id = path.stem

                # Upload to MMORE
                self.mmore_client.upload_file(file_path=str(path), file_id=file_id)

                # Track uploaded document in database
                self.db.add_mmore_document(
                    file_id=file_id, file_name=path.name, file_path=str(path)
                )
            except Exception as e:
                logger.exception("Error adding document to MMORE")
                return f"Error adding document: {e}"
            else:
                return (
                    f"✓ Successfully added '{path.name}' to MMORE knowledge base\n"
                    f"File ID: {file_id}\n"
                    f"MMORE will process multimodal content (text, images, tables) automatically."
                )

        return add_document

    def _create_add_url_tool(self):
        """Create the add URL tool using MMORE."""

        @tool
        def add_url_to_knowledge_base(
            url: Annotated[str, "The URL to download and add to the knowledge base"],
            file_id: Annotated[
                str,
                "Optional custom ID for the document (auto-generated if not provided)",
            ] = "",
        ) -> str:
            """
            Download content from a URL and add it to the MMORE knowledge base.

            Supports:
            - GitHub documentation (automatically converts to raw URLs)
            - HTML pages
            - Markdown files
            - Any web-accessible document

            Use this when users want to add web documentation, GitHub docs, or
            online resources to the knowledge base.
            """
            try:
                # Convert GitHub URLs to raw URLs if needed
                download_url = url
                if "github.com" in url and "/blob/" in url:
                    download_url = url.replace(
                        "github.com", "raw.githubusercontent.com"
                    ).replace("/blob/", "/")
                    logger.info(f"Converted GitHub URL to raw: {download_url}")

                # Determine file extension from URL
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
                    self.mmore_client.upload_file(file_path=temp_path, file_id=file_id)

                    # Track in database
                    self.db.add_mmore_document(
                        file_id=file_id,
                        file_name=path_parts.name or "web_content",
                        file_path=url,  # Store original URL
                        uploaded_by="url_upload",
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
                return f"Error downloading URL: {e.response.status_code} - {e.response.reason}"
            except requests.RequestException as e:
                logger.exception("Error downloading URL")
                return f"Error downloading URL: {e}"
            except Exception as e:
                logger.exception("Error adding URL to MMORE")
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
            try:
                # Get all documents from database
                documents = self.db.get_all_mmore_documents()

                if not documents:
                    return "No documents in MMORE knowledge base yet.\n\nUse 'add_document' to upload PDF, Office, or image files."

                # Format output
                result = f"📚 MMORE Knowledge Base ({len(documents)} document(s)):\n\n"
                for doc in documents:
                    file_name = doc["file_name"]
                    file_id = doc["file_id"]
                    uploaded_at = doc["uploaded_at"].strftime("%Y-%m-%d %H:%M")
                    result += (
                        f"• {file_name} (ID: {file_id}) - uploaded {uploaded_at}\n"
                    )

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
            try:
                # Get document info before deleting
                doc_info = self.db.get_mmore_document(file_id)

                # Delete from MMORE
                self.mmore_client.delete_file(file_id)

                # Remove from database
                self.db.delete_mmore_document(file_id)

                if doc_info:
                    file_name = doc_info["file_name"]
                    return f"✓ Deleted '{file_name}' (ID: {file_id}) from MMORE"
                else:
                    return f"✓ Deleted file ID '{file_id}' from MMORE"

            except Exception as e:
                logger.exception("Error deleting document from MMORE")
                return f"Error deleting document: {e}"

        return delete_document

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the RAG agent.

        Returns:
            System prompt string
        """
        return """You are a specialized document assistant for engineering research, powered by MMORE.

MMORE (Massive Multimodal Open RAG & Extraction) provides advanced capabilities for
processing technical documents including PDFs, images, tables, and complex layouts.

Your role is to help users understand and extract information from technical documents,
research papers, and engineering specifications they have uploaded.

CRITICAL RULES:
1. **ALWAYS use the search_documents tool FIRST**: For EVERY question, you MUST call search_documents before answering
2. **NEVER answer from your training data**: All answers must be based ONLY on documents retrieved via search_documents
3. **Always cite sources**: Include document file IDs and relevance scores from the search results
4. **If no documents found**: Tell the user no relevant documents were found

Guidelines:
1. **First call search_documents**: Use the search tool for every user question - even questions about MMORE, file formats, or system capabilities
2. **Base answers ONLY on search results**: Do not use your general knowledge - only use what search_documents returns
3. **Cite sources explicitly**: Always include file IDs and relevance scores in your response
4. **Be precise**: Engineering work requires accuracy - cite specific sections
5. **Ask for clarification**: If a question is ambiguous, call search_documents first, then ask for clarification if needed
6. **Acknowledge limitations**: If information isn't in the documents, say so clearly
7. **Leverage multimodal content**: MMORE extracts text, images, and tables - mention when visual content is relevant

When users upload documents or URLs:
- Confirm successful processing with MMORE
- Explain that MMORE will extract multimodal content (text, images, tables)
- Suggest 2-3 initial questions they could ask about the document

Available tools:
- **search_documents**: Search through all uploaded documents (use this for every question!)
- **add_document**: Upload a local file to the knowledge base
- **add_url_to_knowledge_base**: Download and add web content (GitHub docs, HTML pages, markdown files)
- **list_documents**: Show all documents in the knowledge base
- **delete_document**: Remove a document by its file ID

Remember: ALWAYS call search_documents FIRST for every question, even if you think you know the answer from your training!"""
