"""
Tools module for EngiAI.

This module contains all custom tools that can be used by agents.
"""

from src.tools.arxiv_tools import (
    create_arxiv_tools,
    get_arxiv_paper,
    search_arxiv,
)
from src.tools.engibench import (
    create_problem,
    # get_dataset_info,  # Disabled to reduce LLM context
    # get_problem_details,  # Disabled to reduce LLM context
    optimize_design,
    render_design,
    simulate_design,
)
from src.tools.engiopt import (
    download_wandb_model,
    # list_available_algorithms,  # Disabled to reduce LLM context
    load_wandb_model,
    sample_designs_from_model,
)
from src.tools.mmore_client import (
    MMOREClient,
    get_mmore_progress_callback,
    set_mmore_progress_callback,
)
from src.tools.rag_tools import create_rag_tools
from src.tools.search import create_search_tool
from src.tools.stl_export import convert_design_to_stl

__all__ = [
    "MMOREClient",
    "convert_design_to_stl",
    "create_arxiv_tools",
    "create_problem",
    "create_rag_tools",
    "create_search_tool",
    "download_wandb_model",
    "get_arxiv_paper",
    # "get_dataset_info",  # Disabled to reduce LLM context
    "get_mmore_progress_callback",
    # "get_problem_details",  # Disabled to reduce LLM context
    # "list_available_algorithms",  # Disabled to reduce LLM context
    "load_wandb_model",
    "optimize_design",
    "render_design",
    "sample_designs_from_model",
    "search_arxiv",
    "set_mmore_progress_callback",
    "simulate_design",
]
