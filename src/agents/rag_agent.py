"""
RAG Agent for document-based question answering.

This agent handles queries about uploaded documents using the RAG system.
"""

import json
import logging
from pathlib import Path
from typing import Annotated, Any

from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.tools import EngineeringRAGChain, EngineerRAGStore, MultimodalDocumentProcessor

logger = logging.getLogger(__name__)


class RAGAgent(BaseAgent):
    """Agent for document-based question answering using RAG."""

    def __init__(self, model_name: str | None = None, temperature: float | None = None):
        """Initialize the RAG agent with vector store and tools.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
        """
        # Initialize RAG components before calling super().__init__()
        self.document_processor = MultimodalDocumentProcessor()
        self.vector_store = EngineerRAGStore(collection_name="engineer_docs")
        self.rag_chain = EngineeringRAGChain(self.vector_store)

        super().__init__(model_name=model_name, temperature=temperature)
        logger.info("RAG Agent initialized with vector store")

    def _create_tools(self) -> list:
        """Create LangChain tools for the RAG agent."""
        return [
            self._create_search_tool(),
            self._create_add_document_tool(),
            self._create_list_documents_tool(),
            self._create_clear_memory_tool(),
        ]

    def _create_search_tool(self):
        """Create the search documents tool."""

        @tool
        def search_documents(
            query: Annotated[str, "The question to search for in documents"],
            num_results: Annotated[int, "Number of relevant documents to retrieve"] = 5,
        ) -> str:
            """
            Search through uploaded documents to find relevant information.

            Use this tool to answer questions about papers, technical documents,
            or any previously uploaded files. It returns contextual information
            with source citations.
            """
            try:
                result = self.rag_chain.ask(query, k=num_results)
                answer = result["answer"]
                sources = result["num_sources"]

            except Exception as e:
                logger.exception("Error searching documents")
                return f"Error searching documents: {e}"

            else:
                # Format response with source information
                response = f"{answer}\n\n📚 Sources: {sources} document(s)"
                return response

        return search_documents

    def _create_add_document_tool(self):
        """Create the add document tool."""

        @tool
        def add_document(
            file_path: Annotated[str, "Path to the document file to add"],
            metadata: Annotated[str, "Optional metadata as JSON string"] = "{}",
        ) -> str:
            """
            Add a new document to the knowledge base.

            Processes PDF files and adds them to the vector store for future queries.
            The document will be chunked and embedded automatically.
            """
            try:
                meta = json.loads(metadata) if metadata != "{}" else {}

                # Process document
                docs = self.document_processor.process_file(file_path)

                # Add metadata
                for doc in docs:
                    doc.metadata.update(meta)

                # Store in vector database
                self.vector_store.add_documents(docs)

                file_name = Path(file_path).name
                return f"✓ Successfully added '{file_name}' to knowledge base ({len(docs)} chunks)"

            except Exception as e:
                logger.exception("Error adding document")
                return f"Error adding document: {e}"

        return add_document

    def _create_list_documents_tool(self):
        """Create the list documents tool."""

        @tool
        def list_documents() -> str:
            """
            List all documents currently in the knowledge base.

            Returns a summary of stored documents with their sources and page counts.
            """
            try:
                # Get all documents (limited retrieval to avoid overload)
                all_docs = self.vector_store.similarity_search("", k=100)

                if not all_docs:
                    return "No documents in the knowledge base yet."

                # Organize by source
                sources: dict[str, dict[str, Any]] = {}
                for doc in all_docs:
                    source = doc.metadata.get("source", "unknown")
                    if source not in sources:
                        sources[source] = {"pages": set(), "chunks": 0}

                    sources[source]["chunks"] += 1
                    if "page" in doc.metadata:
                        sources[source]["pages"].add(doc.metadata["page"])

                # Format output
                result = f"📚 Knowledge Base ({len(sources)} documents):\n\n"
                for source, info in sources.items():
                    file_name = Path(source).name
                    pages_set: set[Any] = info["pages"]  # type: ignore[assignment]
                    pages = len(pages_set) if pages_set else "N/A"
                    chunks: int = info["chunks"]  # type: ignore[assignment]
                    result += f"• {file_name}\n  - Pages: {pages}, Chunks: {chunks}\n"

                total_count = self.vector_store.get_collection_count()
                result += f"\nTotal chunks: {total_count}"

            except Exception as e:
                logger.exception("Error listing documents")
                return f"Error listing documents: {e}"

            else:
                return result

        return list_documents

    def _create_clear_memory_tool(self):
        """Create the clear memory tool."""

        @tool
        def clear_document_memory() -> str:
            """
            Clear the conversation history for document Q&A.

            Use this when starting a new topic or when the user wants to reset
            the conversation context.
            """
            try:
                self.rag_chain.clear_history()
            except Exception as e:
                logger.exception("Error clearing history")
                return f"Error clearing history: {e}"
            else:
                return "✓ Conversation history cleared"

        return clear_document_memory

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the RAG agent.

        Returns:
            System prompt string
        """
        return """You are a specialized document assistant for engineering research.

Your role is to help users understand and extract information from technical documents,
research papers, and engineering specifications they have uploaded.

Guidelines:
1. **Always cite sources**: Include document names and page numbers when answering
2. **Be precise**: Engineering work requires accuracy - cite specific sections
3. **Ask for clarification**: If a question is ambiguous, ask for more details
4. **Acknowledge limitations**: If information isn't in the documents, say so clearly
5. **Use conversation history**: Reference previous questions for better context
6. **Suggest related topics**: When appropriate, suggest related questions users might ask

When users upload documents:
- Confirm successful processing
- Provide a brief summary of what was added
- Suggest 2-3 initial questions they could ask about the document

Always be helpful, accurate, and cite your sources!"""
