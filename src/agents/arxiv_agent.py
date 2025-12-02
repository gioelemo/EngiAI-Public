"""
ArXiv Research Agent for searching and analyzing research papers.

This agent combines ArXiv search capabilities with MMORE RAG for paper analysis.
"""

import logging
import tempfile
from pathlib import Path
from typing import Annotated

import arxiv
from langchain_core.tools import tool

from src.agents.base_agent import BaseAgent
from src.tools import MMOREClient
from src.utils.prompts import ARXIV_AGENT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

# Constants for formatting
MAX_AUTHORS_DISPLAY = 3
SUMMARY_PREVIEW_LENGTH = 300
AUTHORS_PREVIEW_LENGTH = 100


class ArXivAgent(BaseAgent):
    """Agent for ArXiv paper search and analysis with MMORE RAG integration."""

    def __init__(
        self,
        model_name: str | None = None,
        temperature: float | None = None,
        mmore_url: str | None = None,
    ):
        """Initialize the ArXiv agent with search and MMORE RAG capabilities.

        Args:
            model_name: Name of the LLM model to use (defaults to config.llm_model)
            temperature: Model temperature (defaults to config.llm_temperature)
            mmore_url: URL of MMORE service (defaults to MMORE_RAG_URL env var)
        """
        # Initialize MMORE client for paper analysis
        self.mmore_client = MMOREClient(base_url=mmore_url)

        super().__init__(model_name=model_name, temperature=temperature)

        # Verify MMORE connection
        if self.mmore_client.health_check():
            logger.info("ArXiv Agent initialized with MMORE service")
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
        """Create LangChain tools for the ArXiv agent."""
        return [
            self._create_search_tool(),
            self._create_get_paper_tool(),
            self._create_download_and_analyze_tool(),
            self._create_ask_papers_tool(),
            self._create_list_papers_tool(),
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
        ) -> str:
            """
            Download an ArXiv paper PDF and add it to MMORE knowledge base for analysis.

            The paper will be downloaded, processed, and uploaded to MMORE,
            allowing you to ask questions about its contents with advanced
            multimodal retrieval (text, images, tables).
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

                # Generate file_id for MMORE
                file_id = f"arxiv_{clean_id}"

                # Upload to MMORE
                logger.info(f"Uploading paper {clean_id} to MMORE...")
                self.mmore_client.upload_file(file_path=pdf_path, file_id=file_id)

                # Track uploaded document in database
                authors = ", ".join([author.name for author in paper.authors])
                self.db.add_mmore_document(
                    file_id=file_id,
                    file_name=f"{paper.title}.pdf",
                    file_path=pdf_path,
                )

                logger.info(f"Successfully added paper {clean_id} to MMORE")

                authors_preview = authors[:AUTHORS_PREVIEW_LENGTH]
                authors_ellipsis = (
                    "..." if len(authors) > AUTHORS_PREVIEW_LENGTH else ""
                )

                result = f"""✓ Successfully downloaded and analyzed '{paper.title}'
   - ArXiv ID: {clean_id}
   - Authors: {authors_preview}{authors_ellipsis}
   - File ID: {file_id}
   - PDF saved to: {pdf_path}

💡 MMORE will extract multimodal content (text, images, tables) automatically.
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
            Ask questions about papers that have been downloaded and analyzed using MMORE.

            Use this after downloading papers with download_and_analyze_paper().
            Returns relevant passages with citations to specific papers using
            MMORE's advanced multimodal retrieval.
            """
            try:
                # Retrieve documents from MMORE (filter for arxiv papers)
                docs = self.mmore_client.retrieve(
                    query=query,
                    max_matches=num_results,
                    min_similarity=0.3,  # Filter low-quality matches
                )

                if not docs:
                    return "No relevant information found in the analyzed papers. Try downloading more papers or rephrasing your question."

                # Format results with context
                response_parts = []
                for i, doc in enumerate(docs, 1):
                    file_id = doc.metadata.get("source", "unknown")
                    score = doc.metadata.get("score", 0.0)
                    content = doc.page_content[:500]  # Limit content length

                    response_parts.append(
                        f"**Passage {i}** (relevance: {score:.2f})\n"
                        f"Source: {file_id}\n"
                        f"{content}...\n"
                    )

                response = "\n\n".join(response_parts)
                response += f"\n\n📚 Found {len(docs)} relevant passage(s) from papers"
            except Exception as e:
                logger.exception("Error querying papers with MMORE")
                return f"Error querying papers: {e}"
            else:
                return response

        return ask_about_papers

    def _create_list_papers_tool(self):
        """Create the list papers tool."""

        @tool
        def list_analyzed_papers() -> str:
            """
            List all ArXiv papers currently in the MMORE knowledge base.

            Shows papers that have been downloaded and are available for analysis.
            """
            try:
                # Get all documents from database (filter for arxiv papers)
                all_documents = self.db.get_all_mmore_documents()

                # Filter only arxiv papers
                arxiv_papers = [
                    doc for doc in all_documents if doc["file_id"].startswith("arxiv_")
                ]

                if not arxiv_papers:
                    return "No ArXiv papers in MMORE knowledge base yet.\n\nUse download_and_analyze_paper() to add papers."

                # Format output
                result = f"📚 ArXiv Papers in MMORE Knowledge Base ({len(arxiv_papers)} paper(s)):\n\n"
                for doc in arxiv_papers:
                    file_name = doc["file_name"]
                    file_id = doc["file_id"]
                    arxiv_id = file_id.replace("arxiv_", "")
                    uploaded_at = doc["uploaded_at"].strftime("%Y-%m-%d %H:%M")

                    result += f"• **{file_name}**\n"
                    result += f"  - ArXiv ID: {arxiv_id}\n"
                    result += f"  - File ID: {file_id}\n"
                    result += f"  - Uploaded: {uploaded_at}\n\n"

                result += "\n✓ All papers indexed with multimodal content (text, images, tables)"
            except Exception as e:
                logger.exception("Error listing papers")
                return f"Error listing papers: {e}"
            else:
                return result

        return list_analyzed_papers

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the ArXiv agent.

        Returns:
            System prompt string
        """
        return ARXIV_AGENT_SYSTEM_PROMPT
