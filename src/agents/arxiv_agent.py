"""
ArXiv Research Agent for searching and analyzing research papers.

This agent combines ArXiv search capabilities with RAG for paper analysis.
"""

import json
import logging
import tempfile
from pathlib import Path
from typing import Annotated

import arxiv
from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.tools import EngineeringRAGChain, EngineerRAGStore, MultimodalDocumentProcessor
from src.utils.prompts import ARXIV_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Constants for formatting
MAX_AUTHORS_DISPLAY = 3
SUMMARY_PREVIEW_LENGTH = 300
AUTHORS_PREVIEW_LENGTH = 100
AUTHORS_LIST_LENGTH = 80


class ArXivAgent(BaseAgent):
    """Agent for ArXiv paper search and analysis with RAG integration."""

    def __init__(self, model_name: str | None = None):
        """Initialize the ArXiv agent with search and RAG capabilities.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
        """
        # Initialize RAG components for paper analysis
        # Use the same collection as RAG agent for unified knowledge base
        self.document_processor = MultimodalDocumentProcessor()
        self.vector_store = EngineerRAGStore(collection_name="engineer_docs")
        self.rag_chain = EngineeringRAGChain(self.vector_store)

        super().__init__(model_name=model_name)
        logger.info("ArXiv Agent initialized with RAG system")

    def _create_tools(self) -> list:
        """Create LangChain tools for the ArXiv agent."""
        return [
            self._create_search_tool(),
            self._create_get_paper_tool(),
            self._create_download_and_analyze_tool(),
            self._create_ask_papers_tool(),
            self._create_list_papers_tool(),
            self._create_clear_memory_tool(),
        ]

    def _create_search_tool(self):
        """Create the search ArXiv tool."""

        @tool
        def search_arxiv(
            query: Annotated[str, "Search query for ArXiv papers"],
            max_results: Annotated[int, "Maximum number of results"] = 5,
        ) -> str:
            """
            Search ArXiv for research papers matching a query.

            Returns paper titles, authors, IDs, and summaries.
            Use this to find papers on specific topics or by authors.

            Example queries:
            - "quantum computing"
            - "attention mechanisms in transformers"
            - "au:Hinton" (papers by Geoffrey Hinton)
            """
            try:
                search = arxiv.Search(
                    query=query,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.Relevance,
                )

                results = []
                for i, paper in enumerate(search.results(), start=1):
                    authors = ", ".join(
                        [author.name for author in paper.authors[:MAX_AUTHORS_DISPLAY]]
                    )
                    if len(paper.authors) > MAX_AUTHORS_DISPLAY:
                        authors += f" et al. ({len(paper.authors)} authors)"

                    summary_preview = paper.summary[:SUMMARY_PREVIEW_LENGTH]
                    summary_ellipsis = (
                        "..." if len(paper.summary) > SUMMARY_PREVIEW_LENGTH else ""
                    )

                    result = f"""
{i}. **{paper.title}**
   - ArXiv ID: {paper.entry_id.split("/")[-1]}
   - Authors: {authors}
   - Published: {paper.published.strftime("%Y-%m-%d")}
   - Summary: {summary_preview}{summary_ellipsis}
"""
                    results.append(result)

                if not results:
                    return f"No papers found for query: '{query}'"

                response = f"📚 Found {len(results)} papers for '{query}':\n"
                response += "\n".join(results)
                response += "\n\n💡 Use download_and_analyze_paper(arxiv_id) to analyze any paper."
            except Exception as e:
                logger.exception("Error searching ArXiv")
                return f"Error searching ArXiv: {e}"
            else:
                return response

        return search_arxiv

    def _create_get_paper_tool(self):
        """Create the get paper details tool."""

        @tool
        def get_arxiv_paper(
            arxiv_id: Annotated[
                str, "ArXiv paper ID (e.g., '1605.08386' or 'arxiv:1605.08386')"
            ],
        ) -> str:
            """
            Get detailed information about a specific ArXiv paper by its ID.

            Returns full paper details including title, authors, abstract,
            publication date, and PDF link.
            """
            try:
                clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")
                search = arxiv.Search(id_list=[clean_id])
                paper = next(search.results())

                authors = ", ".join([author.name for author in paper.authors])
                categories = ", ".join(paper.categories)

                response = f"""
**{paper.title}**

**ArXiv ID:** {paper.entry_id.split("/")[-1]}
**Authors:** {authors}
**Published:** {paper.published.strftime("%Y-%m-%d")}
**Categories:** {categories}

**Abstract:**
{paper.summary}

**PDF:** {paper.pdf_url}
**DOI:** {paper.doi if paper.doi else "N/A"}

💡 Use download_and_analyze_paper("{clean_id}") to download and analyze this paper.
"""
            except StopIteration:
                return f"Paper with ID '{arxiv_id}' not found on ArXiv"
            except Exception as e:
                logger.exception("Error fetching ArXiv paper")
                return f"Error fetching paper: {e}"
            else:
                return response

        return get_arxiv_paper

    def _create_download_and_analyze_tool(self):
        """Create the download and analyze paper tool."""

        @tool
        def download_and_analyze_paper(
            arxiv_id: Annotated[str, "ArXiv paper ID to download and analyze"],
            metadata: Annotated[str, "Optional metadata as JSON string"] = "{}",
        ) -> str:
            """
            Download an ArXiv paper PDF and add it to the knowledge base for analysis.

            The paper will be downloaded, processed, and added to the RAG system,
            allowing you to ask questions about its contents.
            """
            try:
                # Clean the arxiv ID
                clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")

                # Get paper details
                search = arxiv.Search(id_list=[clean_id])
                paper = next(search.results())

                # Download to temp directory
                temp_dir = Path(tempfile.gettempdir()) / "arxiv_papers"
                temp_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Downloading paper {clean_id}...")
                pdf_path = paper.download_pdf(dirpath=str(temp_dir))

                # Parse metadata
                meta = json.loads(metadata) if metadata != "{}" else {}
                meta.update(
                    {
                        "source": f"ArXiv:{clean_id}",
                        "title": paper.title,
                        "authors": ", ".join([author.name for author in paper.authors]),
                        "published": paper.published.strftime("%Y-%m-%d"),
                        "arxiv_id": clean_id,
                    }
                )

                # Process and add to vector store
                logger.info(f"Processing paper {clean_id}...")
                docs = self.document_processor.process_file(pdf_path)

                # Add metadata to all chunks
                for doc in docs:
                    doc.metadata.update(meta)

                # Store in vector database
                self.vector_store.add_documents(docs)

                logger.info(f"Successfully added paper {clean_id} to knowledge base")

                authors_preview = meta["authors"][:AUTHORS_PREVIEW_LENGTH]
                authors_ellipsis = (
                    "..." if len(meta["authors"]) > AUTHORS_PREVIEW_LENGTH else ""
                )

                result = f"""✓ Successfully downloaded and analyzed '{paper.title}'
   - ArXiv ID: {clean_id}
   - Authors: {authors_preview}{authors_ellipsis}
   - Chunks processed: {len(docs)}
   - PDF saved to: {pdf_path}

💡 You can now use ask_about_papers() to ask questions about this paper!"""

            except StopIteration:
                return f"Paper with ID '{arxiv_id}' not found on ArXiv"
            except Exception as e:
                logger.exception("Error downloading and analyzing paper")
                return f"Error processing paper: {e}"
            else:
                return result

        return download_and_analyze_paper

    def _create_ask_papers_tool(self):
        """Create the ask about papers tool."""

        @tool
        def ask_about_papers(
            query: Annotated[str, "Question about the analyzed papers"],
            num_results: Annotated[int, "Number of relevant chunks to retrieve"] = 5,
        ) -> str:
            """
            Ask questions about papers that have been downloaded and analyzed.

            Use this after downloading papers with download_and_analyze_paper().
            Returns answers with citations to specific papers and sections.
            """
            try:
                result = self.rag_chain.ask(query, k=num_results)
                answer = result["answer"]
                sources = result["num_sources"]

                response = f"{answer}\n\n📚 Sources: {sources} paper section(s)"
            except Exception as e:
                logger.exception("Error querying papers")
                return f"Error querying papers: {e}"
            else:
                return response

        return ask_about_papers

    def _create_list_papers_tool(self):
        """Create the list papers tool."""

        @tool
        def list_analyzed_papers() -> str:
            """
            List all ArXiv papers currently in the knowledge base.

            Shows papers that have been downloaded and are available for analysis.
            """
            try:
                # Get all documents
                all_docs = self.vector_store.similarity_search("", k=100)

                if not all_docs:
                    return "No papers in the knowledge base yet. Use download_and_analyze_paper() to add papers."

                # Organize by ArXiv ID
                papers = {}
                for doc in all_docs:
                    arxiv_id = doc.metadata.get("arxiv_id", "unknown")
                    if arxiv_id not in papers:
                        papers[arxiv_id] = {
                            "title": doc.metadata.get("title", "Unknown"),
                            "authors": doc.metadata.get("authors", "Unknown"),
                            "published": doc.metadata.get("published", "N/A"),
                            "chunks": 0,
                        }
                    papers[arxiv_id]["chunks"] += 1

                # Format output
                result = (
                    f"📚 ArXiv Papers in Knowledge Base ({len(papers)} papers):\n\n"
                )
                for arxiv_id, info in papers.items():
                    authors_preview = info["authors"][:AUTHORS_LIST_LENGTH]
                    authors_ellipsis = (
                        "..." if len(info["authors"]) > AUTHORS_LIST_LENGTH else ""
                    )
                    result += f"• **{info['title']}**\n"
                    result += f"  - ArXiv ID: {arxiv_id}\n"
                    result += f"  - Authors: {authors_preview}{authors_ellipsis}\n"
                    result += f"  - Published: {info['published']}\n"
                    result += f"  - Chunks: {info['chunks']}\n\n"

                total_count = self.vector_store.get_collection_count()
                result += f"Total chunks: {total_count}"
            except Exception as e:
                logger.exception("Error listing papers")
                return f"Error listing papers: {e}"
            else:
                return result

        return list_analyzed_papers

    def _create_clear_memory_tool(self):
        """Create the clear memory tool."""

        @tool
        def clear_conversation_memory() -> str:
            """
            Clear the conversation history for paper analysis.

            Use this when starting a new research topic or when you want to
            reset the conversation context.
            """
            try:
                self.rag_chain.clear_history()
            except Exception as e:
                logger.exception("Error clearing history")
                return f"Error clearing history: {e}"
            else:
                return "✓ Conversation history cleared"

        return clear_conversation_memory

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the ArXiv agent.

        Returns:
            System prompt string
        """
        return ARXIV_AGENT_SYSTEM_PROMPT
