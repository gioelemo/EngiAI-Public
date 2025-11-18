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
    check_thermoelastic_constraints,
    create_beam_problem,
    create_thermoelastic_problem,
    get_dataset_info,
    get_problem_details,
    get_problem_info,
    get_thermoelastic_dataset_info,
    get_thermoelastic_problem_details,
    optimize_beam_design,
    optimize_thermoelastic_design,
    render_beam_design,
    render_thermoelastic_design,
    simulate_beam_design,
    simulate_thermoelastic_design,
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
    "check_thermoelastic_constraints",
    "convert_design_to_stl",
    "create_arxiv_tools",
    "create_beam_problem",
    "create_search_tool",
    "create_thermoelastic_problem",
    "download_arxiv_paper",
    "download_wandb_model",
    "get_arxiv_paper",
    "get_dataset_info",
    "get_problem_details",
    "get_problem_info",
    "get_thermoelastic_dataset_info",
    "get_thermoelastic_problem_details",
    "list_available_algorithms",
    "load_wandb_model",
    "optimize_beam_design",
    "optimize_thermoelastic_design",
    "render_beam_design",
    "render_thermoelastic_design",
    "sample_designs_from_model",
    "search_arxiv",
    "simulate_beam_design",
    "simulate_thermoelastic_design",
]
