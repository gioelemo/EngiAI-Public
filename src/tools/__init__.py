"""
Tools module for the engineer assistant.

This module contains all custom tools that can be used by agents.
"""

from src.tools.arxiv_tools import (
    create_arxiv_tools,
    download_arxiv_paper,
    get_arxiv_paper,
    search_arxiv,
)
from src.tools.document_processor import MultimodalDocumentProcessor
from src.tools.engibench import (
    check_beam_constraints,
    create_beam_problem,
    get_dataset_info,
    get_problem_details,
    get_problem_info,
    optimize_beam_design,
    render_beam_design,
    simulate_beam_design,
)
from src.tools.engiopt import (
    download_wandb_model,
    list_available_algorithms,
    load_wandb_model,
    sample_designs_from_model,
)
from src.tools.rag_chain import EngineeringRAGChain
from src.tools.search import create_search_tool
from src.tools.stl_export import convert_design_to_stl
from src.tools.vector_store import EngineerRAGStore

__all__ = [
    "EngineerRAGStore",
    "EngineeringRAGChain",
    "MultimodalDocumentProcessor",
    "check_beam_constraints",
    "convert_design_to_stl",
    "create_arxiv_tools",
    "create_beam_problem",
    "create_search_tool",
    "download_arxiv_paper",
    "download_wandb_model",
    "get_arxiv_paper",
    "get_dataset_info",
    "get_problem_details",
    "get_problem_info",
    "list_available_algorithms",
    "load_wandb_model",
    "optimize_beam_design",
    "render_beam_design",
    "sample_designs_from_model",
    "search_arxiv",
    "simulate_beam_design",
]
