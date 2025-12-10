#!/usr/bin/env python3
"""Generate API documentation from source code automatically."""

from pathlib import Path


def generate_module_doc(module_path: str, title: str, output_file: Path) -> None:
    """Generate RST file for a Python module.

    Args:
        module_path: Python module path (e.g., 'src.agents')
        title: Title for the documentation page
        output_file: Path to output .rst file
    """
    content = f"""{title}
{"=" * len(title)}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
"""

    output_file.write_text(content)
    print(f"Generated: {output_file}")


def main() -> None:
    """Generate all API documentation files."""
    docs_dir = Path(__file__).parent
    api_dir = docs_dir / "source" / "api"
    api_dir.mkdir(exist_ok=True)

    # Define modules to document
    modules = {
        "agents.rst": [
            ("src.agents.supervisor_agent", "Supervisor Agent"),
            ("src.agents.base_agent", "Base Agent"),
            ("src.agents.rag_agent", "RAG Agent"),
            ("src.agents.arxiv_agent", "ArXiv Agent"),
            ("src.agents.hpc_agent", "HPC Agent"),
            ("src.agents.engineering_agent", "Engineering Agent"),
            ("src.agents.prusa_agent", "Prusa Agent"),
            ("src.agents.search_agent", "Search Agent"),
            ("src.agents.cli_agent", "CLI Agent"),
        ],
        "tools.rst": [
            ("src.tools.arxiv_tools", "ArXiv Tools"),
            ("src.tools.mmore_client", "MMORE Client"),
            ("src.tools.hpc", "HPC Tools"),
            ("src.tools.cli", "CLI Tools"),
            ("src.tools.connection", "Connection Tools"),
            ("src.tools.search", "Search Tools"),
            ("src.tools.web_crawler", "Web Crawler"),
            ("src.tools.algorithms", "Algorithms"),
            ("src.tools.engiopt", "EngiOpt Tools"),
            ("src.tools.engibench", "EngiBench Tools"),
            ("src.tools.stl_export", "STL Export"),
            ("src.tools.job_monitor", "Job Monitor"),
            ("src.tools.problems", "Problem Registry"),
        ],
        "utils.rst": [
            ("src.utils.prompts", "Prompts"),
            ("src.utils.api_usage", "API Usage"),
        ],
    }

    # Generate main API index
    index_content = """API Reference
=============

This section contains auto-generated API documentation from the source code.

.. toctree::
   :maxdepth: 2
   :caption: API Documentation

   agents
   tools
   utils
"""

    (api_dir / "index.rst").write_text(index_content)
    print(f"Generated: {api_dir / 'index.rst'}")

    # Generate individual module documentation
    for filename, module_list in modules.items():
        category = filename.replace(".rst", "").title()
        content = f"""{category} API
{"=" * (len(category) + 4)}

Auto-generated API documentation for {category.lower()}.

"""

        for module_path, title in module_list:
            content += f"""
{title}
{"-" * len(title)}

.. automodule:: {module_path}
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__

"""

        output_file = api_dir / filename
        output_file.write_text(content)
        print(f"Generated: {output_file}")

    print("\n✓ API documentation generated successfully!")
    print("Run 'make docs' to build the documentation.")


if __name__ == "__main__":
    main()
