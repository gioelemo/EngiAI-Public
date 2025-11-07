"""
ArXiv tools for searching and retrieving research papers.

Provides tools for:
- Searching ArXiv papers by query
- Fetching paper details by ID
- Downloading paper PDFs
"""

import logging
import tempfile
from pathlib import Path
from typing import Annotated

import arxiv
from langchain_core.tools import tool

logger = logging.getLogger(__name__)

# Constants for formatting
MAX_AUTHORS_DISPLAY = 3
SUMMARY_PREVIEW_LENGTH = 300


@tool
def search_arxiv(
    query: Annotated[str, "Search query for ArXiv papers"],
    max_results: Annotated[int, "Maximum number of results to return"] = 5,
) -> str:
    """
    Search ArXiv for research papers matching a query.

    Returns paper titles, authors, IDs, and summaries.
    Use this to find papers on specific topics or by specific authors.

    Example queries:
    - "quantum computing"
    - "attention mechanisms in transformers"
    - "au:Hinton" (papers by Geoffrey Hinton)
    """
    try:
        # Create search client
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )

        results = []
        for i, paper in enumerate(search.results(), start=1):
            # Format authors
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
   - PDF: {paper.pdf_url}
"""
            results.append(result)

        if not results:
            return f"No papers found for query: '{query}'"

        response = f"Found {len(results)} papers for '{query}':\n"
        response += "\n".join(results)
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


@tool
def download_arxiv_paper(
    arxiv_id: Annotated[str, "ArXiv paper ID to download"],
    download_dir: Annotated[str, "Directory to save the PDF"] = "",
) -> str:
    """
    Download an ArXiv paper PDF to local storage.

    Returns the path to the downloaded PDF file.
    If no download directory is specified, uses a temporary directory.
    """
    try:
        # Clean the arxiv ID
        clean_id = arxiv_id.replace("arxiv:", "").replace("arXiv:", "")

        # Get paper details
        search = arxiv.Search(id_list=[clean_id])
        paper = next(search.results())

        # Determine download directory
        if download_dir:
            save_dir = Path(download_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        else:
            save_dir = Path(tempfile.gettempdir()) / "arxiv_papers"
            save_dir.mkdir(parents=True, exist_ok=True)

        # Download the paper
        filename = paper.download_pdf(dirpath=str(save_dir))

        result = f"✓ Downloaded '{paper.title}' to: {filename}"

    except StopIteration:
        return f"Paper with ID '{arxiv_id}' not found on ArXiv"
    except Exception as e:
        logger.exception("Error downloading ArXiv paper")
        return f"Error downloading paper: {e}"
    else:
        return result


def create_arxiv_tools() -> list:
    """
    Create a list of ArXiv-related tools.

    Returns:
        List of LangChain tools for ArXiv operations
    """
    return [search_arxiv, get_arxiv_paper, download_arxiv_paper]
