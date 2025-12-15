"""
ArXiv tools for searching and retrieving research papers.

Provides tools for:
- Searching ArXiv papers by query
- Fetching paper details by ID
- Downloading and analyzing papers with MMORE RAG
- Listing and querying analyzed papers
"""

import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import arxiv
from langchain_core.tools import tool

if TYPE_CHECKING:
    from src.tools.mmore_client import MMOREClient
    from src.ui.database import DatabaseManager

logger = logging.getLogger(__name__)

# Constants for formatting
MAX_AUTHORS_DISPLAY = 3
SUMMARY_PREVIEW_LENGTH = 300
AUTHORS_PREVIEW_LENGTH = 100


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
        response += (
            "\n\n💡 Use download_and_analyze_paper(arxiv_id) to analyze any paper."
        )
    except Exception as e:
        logger.exception("Error searching ArXiv")
        return f"Error searching ArXiv: {e}"
    else:
        return response


@tool
def get_arxiv_paper(
    arxiv_id: Annotated[
        str, "ArXiv paper ID (e.g., '1605.08386' or 'arxiv:1605.08386')"
    ],
) -> str:
    """
    Get detailed information about a specific ArXiv paper by its ID.

    Returns the full paper details including title, authors, abstract,
    publication date, and PDF link.
    """
    try:
        # Clean the arxiv ID (remove 'arxiv:' prefix if present)
        clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")

        # Search for the specific paper
        search = arxiv.Search(id_list=[clean_id])
        paper = next(search.results())

        # Format authors
        authors = ", ".join([author.name for author in paper.authors])

        # Format categories
        categories = ", ".join(paper.categories)

        response = f"""
**{paper.title}**

**ArXiv ID:** {paper.entry_id.split("/")[-1]}

**Authors:** {authors}

**Published:** {paper.published.strftime("%Y-%m-%d")}
**Updated:** {paper.updated.strftime("%Y-%m-%d")}

**Categories:** {categories}

**Abstract:**
{paper.summary}

**Links:**
- Paper: {paper.entry_id}
- PDF: {paper.pdf_url}

**DOI:** {paper.doi if paper.doi else "N/A"}
"""
    except StopIteration:
        return f"Paper with ID '{arxiv_id}' not found on ArXiv"
    except Exception as e:
        logger.exception("Error fetching ArXiv paper")
        return f"Error fetching paper: {e}"
    else:
        return response


def create_download_and_analyze_tool(
    mmore_client: "MMOREClient", db_manager: "DatabaseManager"
):
    """Create a tool for downloading and analyzing ArXiv papers with MMORE.

    Args:
        mmore_client: MMORE client instance for uploading papers
        db_manager: Database manager for tracking uploaded documents

    Returns:
        LangChain tool for downloading and analyzing papers
    """

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
            mmore_client.upload_file(file_path=pdf_path, file_id=file_id)

            # Track uploaded document in database
            authors = ", ".join([author.name for author in paper.authors])
            db_manager.add_mmore_document(
                file_id=file_id,
                file_name=f"{paper.title}.pdf",
                file_path=pdf_path,
            )

            logger.info(f"Successfully added paper {clean_id} to MMORE")

            authors_preview = authors[:AUTHORS_PREVIEW_LENGTH]
            authors_ellipsis = "..." if len(authors) > AUTHORS_PREVIEW_LENGTH else ""

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


def create_ask_papers_tool(mmore_client: "MMOREClient"):
    """Create a tool for asking questions about analyzed papers.

    Args:
        mmore_client: MMORE client instance for retrieving information

    Returns:
        LangChain tool for querying papers
    """

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
            docs = mmore_client.retrieve(
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


def create_list_papers_tool(db_manager: "DatabaseManager"):
    """Create a tool for listing analyzed papers.

    Args:
        db_manager: Database manager for accessing document records

    Returns:
        LangChain tool for listing papers
    """

    @tool
    def list_analyzed_papers() -> str:
        """
        List all ArXiv papers currently in the MMORE knowledge base.

        Shows papers that have been downloaded and are available for analysis.
        """
        try:
            # Get all documents from database (filter for arxiv papers)
            all_documents = db_manager.get_all_mmore_documents()

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

            result += (
                "\n✓ All papers indexed with multimodal content (text, images, tables)"
            )
        except Exception as e:
            logger.exception("Error listing papers")
            return f"Error listing papers: {e}"
        else:
            return result

    return list_analyzed_papers


def create_arxiv_tools(
    mmore_client: "MMOREClient | None" = None,
    db_manager: "DatabaseManager | None" = None,
) -> list:
    """
    Create a list of ArXiv-related tools.

    Args:
        mmore_client: Optional MMORE client for analysis tools
        db_manager: Optional database manager for tracking papers

    Returns:
        List of LangChain tools for ArXiv operations
    """
    tools = [search_arxiv, get_arxiv_paper]

    # Add MMORE-integrated tools if dependencies provided
    if mmore_client and db_manager:
        tools.extend(
            [
                create_download_and_analyze_tool(mmore_client, db_manager),
                create_ask_papers_tool(mmore_client),
                create_list_papers_tool(db_manager),
            ]
        )

    return tools
