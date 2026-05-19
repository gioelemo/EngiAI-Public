#!/usr/bin/env python3
"""Sync benchmark prompts from EngiAI into the ASME paper appendix.

Extracts prompts from Python modules and injects them into lstlisting blocks
in the paper LaTeX file, following the same pattern as sync_thesis.py.

Usage:
    python scripts/sync_paper.py --prompts
    python scripts/sync_paper.py --prompts --dry-run
"""

import argparse
import importlib
import logging
import re
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "paper-revision" / "paper_revision_sync_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger("sync_paper")


# ---------------------------------------------------------------------------
# Unicode sanitization (same as sync_thesis.py)
# ---------------------------------------------------------------------------

_UNICODE_REPLACEMENTS: list[tuple[str, str]] = [
    ("\u2192", "->"),  # → rightwards arrow
    ("\u2014", "--"),  # — em dash
    ("\u26a0\ufe0f", "[!]"),  # ⚠️  warning sign + variation selector
    ("\u26a0", "[!]"),  # ⚠  warning sign (without VS16)
    ("\ufe0f", ""),  # variation selector-16 (strip)
    ("\u00b0", "deg"),  # ° degree sign
    ("\u03bc", "u"),  # μ micro sign
]


def _sanitize_for_latex(text: str) -> str:
    """Replace unicode characters that break pdflatex verbatim."""
    for char, replacement in _UNICODE_REPLACEMENTS:
        text = text.replace(char, replacement)
    return text


# ---------------------------------------------------------------------------
# LaTeX lstlisting replacement
# ---------------------------------------------------------------------------

_LISTING_PATTERN = re.compile(
    r"(\\begin\{lstlisting\}\[style=prompt\]\n)(.*?)(\n\\end\{lstlisting\})",
    re.DOTALL,
)


def _replace_listing_after_textsc(tex: str, textsc_label: str, new_content: str) -> str:
    r"""Replace lstlisting content after \textbf{\textsc{<label>}} or \paragraph{\textsc{<label>}} in appendix."""
    # Try patterns in order of specificity (most unique to appendix first)
    patterns = [
        re.escape(f"\\begin{{promptbox}}{{\\textsc{{{textsc_label}}}}}"),
        re.escape(f"\\textbf{{\\textsc{{{textsc_label}}}}}"),
        re.escape(f"\\paragraph{{\\textsc{{{textsc_label}}}}}"),
        re.escape(f"\\textsc{{{textsc_label}}}"),
    ]
    label_match = None
    for pat in patterns:
        label_match = re.search(pat, tex)
        if label_match:
            break
    if not label_match:
        log.warning("\\textsc{%s} not found in LaTeX file", textsc_label)
        return tex

    vm = _LISTING_PATTERN.search(tex, label_match.end())
    if not vm:
        log.warning("No lstlisting block found after \\textsc{%s}", textsc_label)
        return tex

    return tex[: vm.start(2)] + new_content + tex[vm.end(2) :]


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

# Maps beams2d prompt style -> \textsc{} label in paper LaTeX
_WORKFLOW_LABELS: dict[str, str] = {
    "full": "Full",
    "natural": "Natural",
    "workflow-random": "W-Rand",
    "workflow-derived-params": "W-Derived",
    "workflow-distractor": "W-Distract",
    "workflow-conditional": "W-Cond",
    "workflow-multi-export": "W-Multi",
}


def _extract_workflow_prompts(cfg: dict) -> dict[str, str] | None:
    """Extract one example prompt per beams2d style using representative params."""
    try:
        root_str = str(PROJECT_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        logging.getLogger("config").setLevel(logging.WARNING)

        mod = importlib.import_module("benchmarks.problems.beams2d.generate_prompts")

        params = cfg.get("prompts", {}).get("workflow_prompts", {}).get("params", {})
        vf = params.get("volfrac", 0.4)
        fd = params.get("forcedist", 0.65)
        rm = params.get("rmin", 4.0)
        seed = params.get("seed", 42)

        results: dict[str, str] = {}
        results["full"] = mod._create_full_prompt(vf, fd, rm)
        results["natural"] = mod._create_natural_prompt(vf, fd)
        results["workflow-random"] = mod._create_workflow_random_prompt(
            vf, fd, rm, 0, seed
        )[0]
        results["workflow-derived-params"] = mod._create_workflow_derived_params_prompt(
            vf, fd, rm
        )[0]
        results["workflow-distractor"] = mod._create_workflow_distractor_prompt(
            vf, fd, rm, 0, seed
        )[0]
        results["workflow-conditional"] = mod._create_workflow_conditional_prompt(
            vf, fd, rm, 0, seed
        )[0]
        results["workflow-multi-export"] = mod._create_workflow_multi_export_prompt(
            vf, fd, rm, 0, seed
        )[0]
    except Exception:
        log.exception("Failed to extract workflow prompts from beams2d")
        return None
    else:
        return results


def _extract_rag_prompts(cfg: dict) -> dict[str, str] | None:
    """Extract selected RAG evaluation prompts."""
    try:
        root_str = str(PROJECT_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        logging.getLogger("config").setLevel(logging.WARNING)

        mod = importlib.import_module(
            "benchmarks.problems.rag_beams2d.generate_prompts"
        )
        rag_prompts = getattr(mod, "RAG_PROMPTS", None)
        if rag_prompts is None:
            log.warning("RAG_PROMPTS not found in rag_beams2d generate_prompts")
            return None

        indices = cfg.get("prompts", {}).get("rag_prompts", {}).get("indices", [0, 3])
        results: dict[str, str] = {}
        for i in indices:
            if i < len(rag_prompts):
                results[f"P{i}"] = rag_prompts[i]["prompt"]
    except Exception:
        log.exception("Failed to extract RAG prompts")
        return None
    else:
        return results


# Maps \textsc{} label -> (algorithm, natural) for HPC training prompts
_HPC_PROMPT_LABELS: dict[str, dict] = {
    "HPC-Train (Explicit)": {"algorithm": "cgan_cnn_2d", "natural": False},
    "HPC-Train (Natural)": {"algorithm": "cgan_cnn_2d", "natural": True},
}

_HPC_EXAMPLE_SEED = 1
_HPC_EXAMPLE_EPOCHS = 100


def _extract_hpc_prompts() -> dict[str, str] | None:
    """Extract HPC training prompt texts using seed=1, epochs=100."""
    try:
        root_str = str(PROJECT_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        logging.getLogger("config").setLevel(logging.WARNING)

        mod = importlib.import_module(
            "benchmarks.problems.hpc_train_beams2d.generate_prompts"
        )

        results: dict[str, str] = {}
        for label, hpc_cfg in _HPC_PROMPT_LABELS.items():
            if hpc_cfg["natural"]:
                text = mod._build_natural_prompt(
                    _HPC_EXAMPLE_SEED, _HPC_EXAMPLE_EPOCHS, hpc_cfg["algorithm"]
                )
            else:
                text = mod._build_prompt(
                    _HPC_EXAMPLE_SEED, _HPC_EXAMPLE_EPOCHS, hpc_cfg["algorithm"]
                )
            results[label] = text
    except Exception:
        log.exception("Failed to extract HPC prompts")
        return None
    else:
        return results


# ---------------------------------------------------------------------------
# Main sync logic
# ---------------------------------------------------------------------------


def sync_prompts(cfg: dict, *, dry_run: bool) -> list[dict]:  # noqa: PLR0912
    """Sync all prompts into the paper LaTeX file."""
    paper_path = PROJECT_ROOT / cfg.get(
        "paper_submodule_path", "paper-revision/asmeconf"
    )
    target_file = cfg.get("target_file", "asmeconf-template.tex")
    tex_path = paper_path / target_file

    if not tex_path.exists():
        log.warning("LaTeX file not found: %s", tex_path)
        return []

    tex = tex_path.read_text()
    actions: list[dict] = []

    # --- Workflow prompts ---
    workflow_texts = _extract_workflow_prompts(cfg)
    if workflow_texts:
        for style, label in _WORKFLOW_LABELS.items():
            text = workflow_texts.get(style)
            if text is None:
                continue
            sanitized = _sanitize_for_latex(text)
            old_tex = tex
            tex = _replace_listing_after_textsc(tex, label, sanitized)
            if tex != old_tex:
                action = "would replace" if dry_run else "replaced"
                actions.append(
                    {
                        "source": f"beams2d/generate_prompts.py:{style}",
                        "target": f"{target_file} (\\textsc{{{label}}})",
                        "action": action,
                    }
                )
                log.info("%s workflow prompt: %s", action.capitalize(), label)

    # --- HPC prompts ---
    hpc_texts = _extract_hpc_prompts()
    if hpc_texts:
        for label, text in hpc_texts.items():
            sanitized = _sanitize_for_latex(text)
            old_tex = tex
            tex = _replace_listing_after_textsc(tex, label, sanitized)
            if tex != old_tex:
                action = "would replace" if dry_run else "replaced"
                actions.append(
                    {
                        "source": f"hpc_train_beams2d/generate_prompts.py:{label}",
                        "target": f"{target_file} (\\textsc{{{label}}})",
                        "action": action,
                    }
                )
                log.info("%s HPC prompt: %s", action.capitalize(), label)

    # --- RAG prompts ---
    rag_texts = _extract_rag_prompts(cfg)
    if rag_texts:
        for label, text in rag_texts.items():
            sanitized = _sanitize_for_latex(text)
            old_tex = tex
            tex = _replace_listing_after_textsc(tex, label, sanitized)
            if tex != old_tex:
                action = "would replace" if dry_run else "replaced"
                actions.append(
                    {
                        "source": f"rag_beams2d/generate_prompts.py:{label}",
                        "target": f"{target_file} (\\textsc{{{label}}})",
                        "action": action,
                    }
                )
                log.info("%s RAG prompt: %s", action.capitalize(), label)

    # --- Write back ---
    if not dry_run and actions:
        tex_path.write_text(tex)
        log.info("Wrote updated %s", tex_path)
    elif dry_run and actions:
        log.info("Dry run — no files modified")

    return actions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Sync benchmark prompts to ASME paper appendix"
    )
    parser.add_argument(
        "--prompts",
        action="store_true",
        help="Sync prompts into paper LaTeX",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Path to config YAML (default: paper_sync_config.yaml)",
    )
    args = parser.parse_args()

    if not args.prompts:
        parser.print_help()
        print("\nUse --prompts to sync benchmark prompts into the paper.")
        return

    cfg = load_config(args.config)
    actions = sync_prompts(cfg, dry_run=args.dry_run)

    if actions:
        print(
            f"\n{'Would sync' if args.dry_run else 'Synced'} {len(actions)} prompt(s):"
        )
        for a in actions:
            print(f"  {a['action']}: {a['source']} -> {a['target']}")
    else:
        print("No changes needed.")


if __name__ == "__main__":
    main()
