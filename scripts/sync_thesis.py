#!/usr/bin/env python3
"""Sync figures, tables, and prompts from engineer-assistant to the thesis submodule.

Usage:
    python scripts/sync_thesis.py --all
    python scripts/sync_thesis.py --figures --tables
    python scripts/sync_thesis.py --prompts --dry-run
    python scripts/sync_thesis.py --all --auto-commit
"""

import argparse
import hashlib
import importlib
import json
import logging
import re
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "sync_config.yaml"

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger("sync_thesis")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _copy_if_changed(src: Path, dst: Path, *, dry_run: bool) -> str | None:
    """Copy *src* to *dst* if content differs.  Returns action taken or None."""
    if not src.exists():
        log.warning("Source not found: %s", src)
        return None

    if dst.exists() and _sha256(src) == _sha256(dst):
        return None  # unchanged

    action = "would copy" if dry_run else "copied"
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return action


def _submodule_is_clean(thesis_path: Path) -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=thesis_path,
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() == ""


# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    return cfg


# ---------------------------------------------------------------------------
# Figure sync
# ---------------------------------------------------------------------------


def sync_figures(cfg: dict, thesis_root: Path, *, dry_run: bool) -> list[dict]:
    actions: list[dict] = []
    fig_cfg = cfg.get("figures", {})
    primary = fig_cfg.get("primary_source", "")

    # Mappings relative to primary_source
    for mapping in fig_cfg.get("mappings", []):
        src_dir = PROJECT_ROOT / primary / mapping["source"]
        dst_dir = thesis_root / mapping["target"]
        extensions = set(mapping.get("extensions", [".pdf"]))

        if not src_dir.is_dir():
            log.warning("Figure source dir not found: %s", src_dir)
            continue

        for f in sorted(src_dir.iterdir()):
            if f.suffix.lower() not in extensions:
                continue
            dst = dst_dir / f.name
            result = _copy_if_changed(f, dst, dry_run=dry_run)
            if result:
                actions.append(
                    {
                        "source": str(f.relative_to(PROJECT_ROOT)),
                        "target": str(dst.relative_to(thesis_root)),
                        "action": result,
                    }
                )

    # Extra sources (absolute paths relative to PROJECT_ROOT)
    for extra in fig_cfg.get("extra_sources", []):
        src_dir = PROJECT_ROOT / extra["source"]
        dst_dir = thesis_root / extra["target"]
        extensions = set(extra.get("extensions", [".pdf"]))
        recursive = extra.get("recursive", False)
        whitelist = set(extra.get("files", []))  # optional filename whitelist

        if not src_dir.is_dir():
            log.warning("Extra figure source dir not found: %s", src_dir)
            continue

        pattern = src_dir.rglob("*") if recursive else src_dir.iterdir()
        for f in sorted(pattern):
            if not f.is_file() or f.suffix.lower() not in extensions:
                continue
            if whitelist and f.name not in whitelist:
                continue
            rel = f.relative_to(src_dir)
            dst = dst_dir / rel
            result = _copy_if_changed(f, dst, dry_run=dry_run)
            if result:
                actions.append(
                    {
                        "source": str(f.relative_to(PROJECT_ROOT)),
                        "target": str(dst.relative_to(thesis_root)),
                        "action": result,
                    }
                )

    return actions


# ---------------------------------------------------------------------------
# Table sync
# ---------------------------------------------------------------------------


def sync_tables(cfg: dict, thesis_root: Path, *, dry_run: bool) -> list[dict]:
    actions: list[dict] = []
    tbl_cfg = cfg.get("tables", {})

    # Static file copies
    for mapping in tbl_cfg.get("mappings", []):
        src = PROJECT_ROOT / mapping["source"]
        dst = thesis_root / mapping["target"]
        result = _copy_if_changed(src, dst, dry_run=dry_run)
        if result:
            actions.append(
                {
                    "source": str(src.relative_to(PROJECT_ROOT)),
                    "target": str(dst.relative_to(thesis_root)),
                    "action": result,
                }
            )

    # Generator commands
    for gen in tbl_cfg.get("generators", []):
        dst = thesis_root / gen["target"]
        cmd = gen["command"].replace("{target}", str(dst))
        if dry_run:
            log.info("Would run: %s", cmd)
            actions.append(
                {
                    "source": cmd,
                    "target": str(dst.relative_to(thesis_root)),
                    "action": "would generate",
                }
            )
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)
            log.info("Running: %s", cmd)
            gen_result = subprocess.run(
                shlex.split(cmd),
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            if gen_result.returncode != 0:
                log.warning("Generator failed: %s\n%s", cmd, gen_result.stderr)
            else:
                actions.append(
                    {
                        "source": cmd,
                        "target": str(dst.relative_to(thesis_root)),
                        "action": "generated",
                    }
                )

    return actions


# ---------------------------------------------------------------------------
# Prompt extraction
# ---------------------------------------------------------------------------

# Unicode chars that break pdflatex lstlisting/verbatim -> ASCII replacements
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


# Maps config label key -> label suffix used in appendix.tex (agent prompts)
_LABEL_MAP = {
    "supervisor": "supervisorsysprompt",
    "engineer": "engineersysprompt",
    "search": "searchsysprompt",
    "rag": "ragsysprompt",
    "arxiv": "arxivsysprompt",
    "hpc": "hpcsysprompt",
    "cli": "clisysprompt",
    "prusa": "prusasysprompt",
}


def _extract_prompt(attr_name: str) -> str | None:
    """Extract a prompt string from src.utils.prompts by attribute or function name."""
    try:
        # Ensure project root is on sys.path for imports
        root_str = str(PROJECT_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        # Suppress noisy config initialization logs during import
        logging.getLogger("config").setLevel(logging.WARNING)

        mod = importlib.import_module("src.utils.prompts")

        obj = getattr(mod, attr_name, None)
        if obj is None:
            log.warning("Attribute %r not found in src.utils.prompts", attr_name)
            return None

        if callable(obj):
            # Try calling with no args first, fall back to common kwargs
            try:
                return str(obj())
            except TypeError:
                try:
                    return str(obj(read_only=False))
                except TypeError:
                    log.warning("Cannot call %s() — unknown signature", attr_name)
                    return None
        return str(obj)
    except Exception:
        log.exception("Failed to import src.utils.prompts")
        return None


def _extract_prompt_regex(attr_name: str) -> str | None:
    """Fallback: regex-extract a string constant from prompts.py."""
    prompts_file = PROJECT_ROOT / "src" / "utils" / "prompts.py"
    if not prompts_file.exists():
        return None

    content = prompts_file.read_text()
    # Match: ATTR_NAME = """..."""  or  ATTR_NAME = "..."
    pattern = rf'{re.escape(attr_name)}\s*=\s*(?:f?"""(.*?)"""|f?"(.*?)")'
    m = re.search(pattern, content, re.DOTALL)
    if m:
        return m.group(1) or m.group(2)
    return None


_LISTING_PATTERN = re.compile(
    r"(\\begin\{lstlisting\}\[style=prompt\]\n)(.*?)(\n\\end\{lstlisting\})",
    re.DOTALL,
)


def _replace_listing_after_label(tex: str, label_suffix: str, new_content: str) -> str:
    """Replace lstlisting content following \\label{subsubsec:<label_suffix>}."""
    label_pattern = re.escape(f"\\label{{subsubsec:{label_suffix}}}")
    label_match = re.search(label_pattern, tex)
    if not label_match:
        log.warning("Label subsubsec:%s not found in appendix.tex", label_suffix)
        return tex

    vm = _LISTING_PATTERN.search(tex, label_match.end())
    if not vm:
        log.warning("No lstlisting block found after label subsubsec:%s", label_suffix)
        return tex

    return tex[: vm.start(2)] + new_content + tex[vm.end(2) :]


def _replace_listing_after_textsc(tex: str, textsc_label: str, new_content: str) -> str:
    """Replace lstlisting content after \\textsc{<label>} in a promptbox."""
    pattern = re.escape(f"\\textsc{{{textsc_label}}}")
    label_match = re.search(pattern, tex)
    if not label_match:
        log.warning("\\textsc{%s} not found in appendix.tex", textsc_label)
        return tex

    vm = _LISTING_PATTERN.search(tex, label_match.end())
    if not vm:
        log.warning("No lstlisting block found after \\textsc{%s}", textsc_label)
        return tex

    return tex[: vm.start(2)] + new_content + tex[vm.end(2) :]


def _extract_benchmark_prompts() -> list[str] | None:
    """Extract RAG_PROMPTS prompt texts from the rag_beams2d generate_prompts module."""
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
            log.warning("RAG_PROMPTS not found in generate_prompts module")
            return None
        return [p["prompt"] for p in rag_prompts]
    except Exception:
        log.exception("Failed to import rag_beams2d generate_prompts")
        return None


# Maps P0-P3 index -> \textsc{} label used in appendix.tex
_BENCHMARK_PROMPT_LABELS = ["P0", "P1", "P2", "P3"]

# Maps beams2d prompt style -> \textsc{} label in appendix.tex
_WORKFLOW_PROMPT_LABELS: dict[str, str] = {
    "full": "Full",
    "natural": "Natural",
    "workflow-random": "Workflow-Random",
    "workflow-derived-params": "Workflow-Derived-Params",
    "workflow-distractor": "Workflow-Distractor",
    "workflow-conditional": "Workflow-Conditional",
    "workflow-multi-export": "Workflow-Multi-Export",
}

# Representative parameters for generating example prompts
_EXAMPLE_PARAMS: dict[str, float] = {
    "volfrac": 0.4,
    "forcedist": 0.65,
    "rmin": 4.0,
}
_EXAMPLE_SEED = 42


def _extract_workflow_prompts() -> dict[str, str] | None:
    """Extract one example prompt per beams2d style using representative params."""
    try:
        root_str = str(PROJECT_ROOT)
        if root_str not in sys.path:
            sys.path.insert(0, root_str)

        logging.getLogger("config").setLevel(logging.WARNING)

        mod = importlib.import_module("benchmarks.problems.beams2d.generate_prompts")

        vf = _EXAMPLE_PARAMS["volfrac"]
        fd = _EXAMPLE_PARAMS["forcedist"]
        rm = _EXAMPLE_PARAMS["rmin"]
        seed = _EXAMPLE_SEED

        results: dict[str, str] = {}

        # Simple styles (return str)
        results["full"] = mod._create_full_prompt(vf, fd, rm)
        results["natural"] = mod._create_natural_prompt(vf, fd)
        # Complex styles (return tuple[str, dict])
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


def _sync_agent_prompts(
    prompt_cfg: dict, target_file: str, tex: str
) -> tuple[str, list[dict]]:
    """Replace agent system prompt verbatim blocks.  Returns (updated_tex, actions)."""
    actions: list[dict] = []
    for agent_key, attr_name in prompt_cfg.get("agent_prompts", {}).items():
        label_suffix = _LABEL_MAP.get(agent_key)
        if not label_suffix:
            log.warning("Unknown agent key %r — add it to _LABEL_MAP", agent_key)
            continue

        prompt_text = _extract_prompt(attr_name)
        if prompt_text is None:
            prompt_text = _extract_prompt_regex(attr_name)
        if prompt_text is None:
            log.warning("Could not extract prompt for %s (%s)", agent_key, attr_name)
            continue

        prompt_text = re.sub(
            r"\n*## Suggested Next Prompts.*", "", prompt_text, flags=re.DOTALL
        )
        prompt_text = _sanitize_for_latex(prompt_text)

        old_tex = tex
        tex = _replace_listing_after_label(tex, label_suffix, prompt_text)
        if tex != old_tex:
            actions.append(
                {
                    "source": f"src/utils/prompts.py:{attr_name}",
                    "target": f"{target_file} (subsubsec:{label_suffix})",
                    "action": "replaced",
                }
            )
    return tex, actions


def _sync_workflow_prompts(
    prompt_cfg: dict, target_file: str, tex: str
) -> tuple[str, list[dict]]:
    """Replace workflow prompt example verbatim blocks.  Returns (updated_tex, actions)."""
    actions: list[dict] = []
    if not prompt_cfg.get("workflow_prompts"):
        return tex, actions

    workflow_texts = _extract_workflow_prompts()
    if workflow_texts is None:
        log.warning("Could not extract workflow prompts — skipping")
        return tex, actions

    for style, label in _WORKFLOW_PROMPT_LABELS.items():
        text = workflow_texts.get(style)
        if text is None:
            continue
        wrapped = _sanitize_for_latex(text)
        old_tex = tex
        tex = _replace_listing_after_textsc(tex, label, wrapped)
        if tex != old_tex:
            actions.append(
                {
                    "source": f"beams2d/generate_prompts.py:{style}",
                    "target": f"{target_file} (\\textsc{{{label}}})",
                    "action": "replaced",
                }
            )
    return tex, actions


# Maps \textsc{} label -> (algorithm, natural) for HPC training prompts
_HPC_PROMPT_LABELS: dict[str, dict[str, str | bool]] = {
    "HPC-Train (Explicit)": {"algorithm": "cgan_cnn_2d", "natural": False},
    "HPC-Train (Natural)": {"algorithm": "cgan_cnn_2d", "natural": True},
    "HPC-Train-Diff (Natural)": {"algorithm": "diffusion_2d_cond", "natural": True},
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
        for label, cfg in _HPC_PROMPT_LABELS.items():
            if cfg["natural"]:
                text = mod._build_natural_prompt(
                    _HPC_EXAMPLE_SEED, _HPC_EXAMPLE_EPOCHS, cfg["algorithm"]
                )
            else:
                text = mod._build_prompt(
                    _HPC_EXAMPLE_SEED, _HPC_EXAMPLE_EPOCHS, cfg["algorithm"]
                )
            results[label] = text
    except Exception:
        log.exception("Failed to extract HPC prompts from hpc_train_beams2d")
        return None
    else:
        return results


def _sync_hpc_prompts(
    prompt_cfg: dict, target_file: str, tex: str
) -> tuple[str, list[dict]]:
    """Replace HPC training prompt verbatim blocks.  Returns (updated_tex, actions)."""
    actions: list[dict] = []
    if not prompt_cfg.get("hpc_prompts"):
        return tex, actions

    hpc_texts = _extract_hpc_prompts()
    if hpc_texts is None:
        log.warning("Could not extract HPC prompts — skipping")
        return tex, actions

    for label, text in hpc_texts.items():
        wrapped = _sanitize_for_latex(text)
        old_tex = tex
        tex = _replace_listing_after_textsc(tex, label, wrapped)
        if tex != old_tex:
            actions.append(
                {
                    "source": f"hpc_train_beams2d/generate_prompts.py:{label}",
                    "target": f"{target_file} (\\textsc{{{label}}})",
                    "action": "replaced",
                }
            )
    return tex, actions


def _sync_benchmark_prompts(
    prompt_cfg: dict, target_file: str, tex: str
) -> tuple[str, list[dict]]:
    """Replace P0-P3 RAG benchmark prompt verbatim blocks.  Returns (updated_tex, actions)."""
    actions: list[dict] = []
    benchmark_cfg = prompt_cfg.get("benchmark_prompts", {})
    if not benchmark_cfg:
        return tex, actions

    prompt_texts = _extract_benchmark_prompts()
    if prompt_texts is None:
        log.warning("Could not extract benchmark prompts — skipping P0-P3")
        return tex, actions

    for i, (label, text) in enumerate(
        zip(_BENCHMARK_PROMPT_LABELS, prompt_texts, strict=False)
    ):
        wrapped = _sanitize_for_latex(text)
        old_tex = tex
        tex = _replace_listing_after_textsc(tex, label, wrapped)
        if tex != old_tex:
            actions.append(
                {
                    "source": f"{benchmark_cfg.get('source', 'rag_beams2d')}:P{i}",
                    "target": f"{target_file} (\\textsc{{{label}}})",
                    "action": "replaced",
                }
            )
    return tex, actions


def sync_prompts(cfg: dict, thesis_root: Path, *, dry_run: bool) -> list[dict]:
    prompt_cfg = cfg.get("prompts", {})
    target_file = prompt_cfg.get("target_file", "appendix.tex")
    appendix_path = thesis_root / target_file

    if not appendix_path.exists():
        log.warning("Appendix file not found: %s", appendix_path)
        return []

    tex = appendix_path.read_text()
    original_tex = tex

    tex, agent_actions = _sync_agent_prompts(prompt_cfg, target_file, tex)
    tex, workflow_actions = _sync_workflow_prompts(prompt_cfg, target_file, tex)
    tex, hpc_actions = _sync_hpc_prompts(prompt_cfg, target_file, tex)
    tex, bench_actions = _sync_benchmark_prompts(prompt_cfg, target_file, tex)
    all_actions = agent_actions + workflow_actions + hpc_actions + bench_actions

    # In dry-run mode, change "replaced" → "would replace"
    if dry_run:
        for a in all_actions:
            a["action"] = "would replace"
    elif tex != original_tex:
        appendix_path.write_text(tex)

    return all_actions


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


def write_manifest(thesis_root: Path, all_actions: list[dict]) -> None:
    manifest = {
        "last_sync": datetime.now(tz=UTC).isoformat(),
        "synced_files": [a for a in all_actions if "would" not in a.get("action", "")],
    }
    manifest_path = thesis_root / ".sync_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Auto-commit
# ---------------------------------------------------------------------------


def auto_commit(thesis_root: Path) -> None:
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M")
    # Only stage directories the sync pipeline writes to (avoid temp/OS files).
    for p in ("figures", "tables", "appendices", ".sync_manifest.json"):
        target = thesis_root / p
        if target.exists():
            subprocess.run(["git", "add", str(p)], cwd=thesis_root, check=True)

    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=thesis_root,
        capture_output=True,
        text=True,
        check=False,
    )
    if not status.stdout.strip():
        log.info("Nothing to commit in thesis submodule.")
        return

    subprocess.run(
        ["git", "commit", "-m", f"sync: update from engineer-assistant ({ts})"],
        cwd=thesis_root,
        check=True,
    )
    log.info("Committed changes in thesis submodule.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sync content from engineer-assistant to the thesis submodule.",
    )
    parser.add_argument("--figures", action="store_true", help="Sync figures")
    parser.add_argument("--tables", action="store_true", help="Sync tables")
    parser.add_argument("--prompts", action="store_true", help="Sync prompts")
    parser.add_argument("--all", action="store_true", help="Sync everything")
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview without making changes"
    )
    parser.add_argument(
        "--auto-commit",
        action="store_true",
        help="Commit in thesis submodule after sync",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Config file path"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    if not any([args.figures, args.tables, args.prompts, args.all]):
        parser.error("Specify at least one of: --figures, --tables, --prompts, --all")

    cfg = load_config(args.config)
    thesis_root = PROJECT_ROOT / cfg["thesis_submodule_path"]

    if not thesis_root.is_dir():
        log.error(
            "Thesis submodule not found at %s. Run: git submodule update --init",
            thesis_root,
        )
        sys.exit(1)

    # Safety check
    if not args.dry_run and not _submodule_is_clean(thesis_root):
        log.warning(
            "Thesis submodule has uncommitted changes. "
            "Consider committing or stashing them first."
        )

    do_figures = args.all or args.figures
    do_tables = args.all or args.tables
    do_prompts = args.all or args.prompts

    all_actions: list[dict] = []
    prefix = "[DRY RUN] " if args.dry_run else ""

    if do_figures:
        log.info("%sSyncing figures...", prefix)
        all_actions.extend(sync_figures(cfg, thesis_root, dry_run=args.dry_run))

    if do_tables:
        log.info("%sSyncing tables...", prefix)
        all_actions.extend(sync_tables(cfg, thesis_root, dry_run=args.dry_run))

    if do_prompts:
        log.info("%sSyncing prompts...", prefix)
        all_actions.extend(sync_prompts(cfg, thesis_root, dry_run=args.dry_run))

    # Summary
    if all_actions:
        print(f"\n{'=' * 60}")
        print(f"{prefix}Sync summary: {len(all_actions)} action(s)")
        print(f"{'=' * 60}")
        for a in all_actions:
            print(f"  {a['action']:>15}  {a['source']}")
            print(f"{'':>15}  -> {a['target']}")
    else:
        print(f"\n{prefix}Nothing to sync — everything is up to date.")

    # Manifest & auto-commit
    if not args.dry_run and all_actions:
        write_manifest(thesis_root, all_actions)
        if args.auto_commit:
            auto_commit(thesis_root)


if __name__ == "__main__":
    main()
