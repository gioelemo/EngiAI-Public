#!/usr/bin/env python3
"""Build the arXiv version of the EngiAI paper from the IDETC revision.

Reads the hand-maintained arXiv template (preamble + title block + keywords)
and injects two pieces of auto-extracted content from the IDETC source:
  - the abstract body (between \\begin{abstract}...\\end{abstract})
  - the paper body (from \\acresetall through the line before \\end{document})

The source of truth for content is paper-revision/asmeconf/asmeconf-template.tex.
The template file (EngiAI---arXiv/template.tex) is hand-maintained — edit it
for title, author, affiliation, keywords, package list, or macro shims.

Usage:
    python scripts/build_arxiv.py
    python scripts/build_arxiv.py --dry-run
"""

import argparse
import logging
import re
import shutil
import sys
from pathlib import Path

import yaml  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "EngiAI---arXiv" / "arxiv_build_config.yaml"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger("build_arxiv")


def load_config(config_path: Path) -> dict:
    with config_path.open() as f:
        return yaml.safe_load(f)


def _extract(tex: str, start_marker: str, end_marker: str, *, inclusive: bool) -> str:
    """Return the substring delimited by start_marker / end_marker.

    With inclusive=True, the markers themselves are included.
    With inclusive=False, only the content between them is returned.
    """
    start = tex.find(start_marker)
    if start == -1:
        raise RuntimeError(f"start_marker {start_marker!r} not found in source")
    # For end_marker that is unique near EOF (e.g. \end{document}), use rfind.
    end = tex.rfind(end_marker)
    if end == -1 or end < start:
        raise RuntimeError(f"end_marker {end_marker!r} not found after start_marker")
    if inclusive:
        return tex[start:end]
    return tex[start + len(start_marker) : end]


def _apply_path_rewrites(text: str, rewrites: list[dict]) -> str:
    for r in rewrites or []:
        text = text.replace(r["from"], r["to"])
    return text


def _scale_single_column_table_files(tables_dir: Path, scale: float) -> int:
    """For each .tex file in tables_dir that is a single-column `\\begin{table}`
    (not `table*`) AND uses `\\resizebox{\\columnwidth}{!}{...}`, rewrite the
    width to `<scale>\\columnwidth`. Returns count of files modified.
    """
    n = 0
    if not tables_dir.exists():
        return 0
    for tex_path in tables_dir.glob("*.tex"):
        text = tex_path.read_text()
        if "\\begin{table*}" in text or "\\resizebox{\\columnwidth}" not in text:
            continue
        if "\\begin{table}" not in text:
            continue
        new_text = text.replace(
            "\\resizebox{\\columnwidth}",
            f"\\resizebox{{{scale:.3g}\\columnwidth}}",
        )
        if new_text != text:
            tex_path.write_text(new_text)
            n += 1
            log.info("scaled resizebox in %s by %s", tex_path.name, scale)
    return n


def _scale_single_column_figures(body: str, scale: float) -> tuple[str, int]:
    """Scale every `width=N\\linewidth` inside a `\\begin{figure}` env (not
    `figure*`) by `scale`. IDETC's `figure` spans one ~3.25" column; arXiv's
    `\\linewidth` is ~6.5", so figures meant for one column render 2x too large.
    `figure*` is the two-column-spanning variant and is left alone.
    Returns (rewritten_body, count_of_widths_scaled).
    """
    count = 0

    def _scale_w(wm: re.Match) -> str:
        nonlocal count
        num = wm.group(1)
        try:
            factor = float(num) if num else 1.0
        except ValueError:
            return wm.group(0)
        count += 1
        return f"width={factor * scale:.3g}\\linewidth"

    def _scale_env(env_match: re.Match) -> str:
        head, content, tail = env_match.group(1), env_match.group(2), env_match.group(3)
        new_content = re.sub(r"width=([0-9.]*)\\linewidth", _scale_w, content)
        return head + new_content + tail

    body = re.sub(
        r"(\\begin\{figure\}(?:\[[^\]]*\])?)(.*?)(\\end\{figure\})",
        _scale_env,
        body,
        flags=re.DOTALL,
    )
    return body, count


def _flatten_multicolumn_math(body: str) -> tuple[str, int]:
    """Flatten ASME two-column math envs (multline, equation+split) into
    single-line equation envs for arXiv's single-column layout.

    Returns (rewritten_body, count_of_envs_flattened).
    """
    count = 0

    def _from_multline(m: re.Match) -> str:
        nonlocal count
        count += 1
        content = m.group(1)
        # IDETC continues onto next line with "\\\n+ \;" or "\\\n- \;"
        content = re.sub(r"\\\\\s*\+\s*\\;\s*", " + ", content)
        content = re.sub(r"\\\\\s*-\s*\\;\s*", " - ", content)
        content = re.sub(r"\\\\\s*", " ", content)
        return r"\begin{equation}" + content + r"\end{equation}"

    body = re.sub(
        r"\\begin\{multline\}(.*?)\\end\{multline\}",
        _from_multline,
        body,
        flags=re.DOTALL,
    )

    def _from_eq_split(m: re.Match) -> str:
        nonlocal count
        count += 1
        content = m.group(1)
        content = content.replace("={}", "=")
        content = re.sub(r"\\\\\s*&", " ", content)  # `\\\n  & ...`
        content = re.sub(r"\\\\\s*", " ", content)  # any remaining `\\`
        content = re.sub(r"\s*&\s*", " ", content)  # alignment marker `&`
        return r"\begin{equation}" + content + r"\end{equation}"

    body = re.sub(
        r"\\begin\{equation\}\s*\\begin\{split\}(.*?)\\end\{split\}\s*\\end\{equation\}",
        _from_eq_split,
        body,
        flags=re.DOTALL,
    )
    return body, count


def _sync_assets(cfg: dict, *, dry_run: bool) -> list[dict]:
    actions: list[dict] = []
    assets = cfg.get("assets", {}) or {}

    for d in assets.get("copy_dirs", []) or []:
        src = PROJECT_ROOT / d["from"]
        dst = PROJECT_ROOT / d["to"]
        if not src.exists():
            log.warning("Asset dir missing: %s", src)
            continue
        n_files = sum(1 for _ in src.rglob("*") if _.is_file())
        action = "would copy" if dry_run else "copied"
        if not dry_run:
            shutil.copytree(src, dst, dirs_exist_ok=True)
        actions.append(
            {
                "source": str(src.relative_to(PROJECT_ROOT)),
                "target": str(dst.relative_to(PROJECT_ROOT)),
                "action": f"{action} dir ({n_files} files)",
            }
        )
        log.info("%s %s -> %s (%d files)", action, src.name, dst.name, n_files)

    max_authors = cfg.get("truncate_bib_authors_to")

    for f in assets.get("copy_files", []) or []:
        src = PROJECT_ROOT / f["from"]
        dst = PROJECT_ROOT / f["to"]
        if not src.exists():
            log.warning("Asset file missing: %s", src)
            continue
        action = "would copy" if dry_run else "copied"
        note = ""
        if not dry_run:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if max_authors and src.suffix == ".bib":
                truncated_text, n = _truncate_bib_authors(src.read_text(), max_authors)
                dst.write_text(truncated_text)
                shutil.copystat(src, dst)
                if n:
                    note = f" (truncated {n} author lists to {max_authors})"
            else:
                shutil.copy2(src, dst)
        actions.append(
            {
                "source": str(src.relative_to(PROJECT_ROOT)),
                "target": str(dst.relative_to(PROJECT_ROOT)),
                "action": f"{action} file{note}",
            }
        )
        log.info("%s %s -> %s%s", action, src.name, dst.relative_to(PROJECT_ROOT), note)

    return actions


_BIB_AUTHOR_START_RE = re.compile(r"author\s*=\s*\{", re.IGNORECASE)


def _truncate_bib_authors(bib_text: str, max_authors: int) -> tuple[str, int]:
    """Truncate every `author = {A and B and ...}` field to first `max_authors`
    entries followed by `and others` (BibTeX renders as 'et al.').

    Walks brace-balanced to find each author field's closing `}` so it
    tolerates protected letters like `Wanderman-{M}ilne` or `Rockt\\"{a}schel`.

    Returns the rewritten bib text and the count of entries truncated.
    """
    truncated = 0
    out: list[str] = []
    pos = 0

    while True:
        m = _BIB_AUTHOR_START_RE.search(bib_text, pos)
        if m is None:
            out.append(bib_text[pos:])
            break

        # Append everything up to and including the opening `{`.
        out.append(bib_text[pos : m.end()])

        # Walk to the matching closing `}` (brace-balanced).
        depth = 1
        i = m.end()
        while i < len(bib_text) and depth > 0:
            c = bib_text[i]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        if depth != 0:
            # Unbalanced — give up on this entry, copy verbatim and stop.
            out.append(bib_text[m.end() :])
            break

        authors_raw = bib_text[m.end() : i]
        parts = [p.strip() for p in re.split(r"\s+and\s+", authors_raw) if p.strip()]
        if len(parts) > max_authors and parts[-1].lower() != "others":
            kept = parts[:max_authors]
            out.append(f"{' and '.join(kept)} and others")
            truncated += 1
        else:
            out.append(authors_raw)

        out.append("}")  # closing brace
        pos = i + 1

    return "".join(out), truncated


_INPUT_RE = re.compile(r"\\input\{([^}]+)\}")
_INCLUDEGRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_SECTION_RE = re.compile(r"^\\section\{", re.MULTILINE)


def _validate(output_tex: str, target_dir: Path, cfg: dict) -> list[str]:
    warnings: list[str] = []

    for m in _INPUT_RE.finditer(output_tex):
        ref = m.group(1)
        candidate = target_dir / ref
        if not candidate.exists() and not candidate.with_suffix(".tex").exists():
            warnings.append(f"\\input{{{ref}}} -> file not found under {target_dir}")

    for m in _INCLUDEGRAPHICS_RE.finditer(output_tex):
        ref = m.group(1)
        candidate = target_dir / ref
        if not candidate.exists() and not any(
            candidate.with_suffix(ext).exists()
            for ext in (".pdf", ".png", ".jpg", ".jpeg")
        ):
            warnings.append(
                f"\\includegraphics{{{ref}}} -> file not found under {target_dir}"
            )

    expected = cfg.get("expected_section_count")
    if expected is not None:
        n = len(_SECTION_RE.findall(output_tex))
        if n != expected:
            warnings.append(f"Expected {expected} \\section{{...}} blocks, found {n}")

    warnings.extend(
        f"Placeholder {ph!r} still present in output"
        for ph in (cfg.get("placeholders") or {}).values()
        if ph and ph in output_tex
    )

    return warnings


def _assemble_body(src_tex: str, cfg: dict) -> tuple[str, str]:
    """Extract abstract + body from the IDETC source and apply rewrites.
    Returns (abstract_text, body_text).
    """
    extract = cfg["extract"]

    abstract_text = _extract(
        src_tex,
        extract["abstract"]["start_marker"],
        extract["abstract"]["end_marker"],
        inclusive=extract["abstract"].get("inclusive", False),
    ).strip()

    body_text = _extract(
        src_tex,
        extract["body"]["start_marker"],
        extract["body"]["end_marker"],
        inclusive=extract["body"].get("inclusive", True),
    )
    body_text = _apply_path_rewrites(body_text, cfg.get("path_rewrites", []))

    if cfg.get("flatten_multicolumn_math", True):
        body_text, n_flat = _flatten_multicolumn_math(body_text)
        if n_flat:
            log.info(
                "flattened %d multi-column math env(s) to single-line equation", n_flat
            )

    fig_scale = cfg.get("scale_single_column_figures")
    if fig_scale:
        body_text, n_scaled = _scale_single_column_figures(body_text, float(fig_scale))
        if n_scaled:
            log.info(
                "scaled %d single-column figure width(s) by %s", n_scaled, fig_scale
            )

    for ref in cfg.get("wrap_inputs_landscape", []) or []:
        target = f"\\input{{{ref}}}"
        wrapped = f"\\begin{{landscape}}\n{target}\n\\end{{landscape}}"
        if target in body_text:
            body_text = body_text.replace(target, wrapped)
            log.info("wrapped %s in landscape env", target)
        else:
            log.warning("wrap_inputs_landscape: %r not found in body", target)

    return abstract_text, body_text


def build_arxiv(cfg: dict, *, dry_run: bool) -> list[dict]:
    src_path = PROJECT_ROOT / cfg["source_paper"]
    template_path = PROJECT_ROOT / cfg["template_file"]
    target_path = PROJECT_ROOT / cfg["target_tex"]

    if not src_path.exists():
        raise RuntimeError(f"Source paper not found: {src_path}")
    if not template_path.exists():
        raise RuntimeError(f"Template file not found: {template_path}")

    src_tex = src_path.read_text()
    template = template_path.read_text()
    placeholders = cfg["placeholders"]

    abstract_text, body_text = _assemble_body(src_tex, cfg)

    if placeholders["abstract"] not in template:
        raise RuntimeError(
            f"Abstract placeholder {placeholders['abstract']!r} not in template"
        )
    if placeholders["body"] not in template:
        raise RuntimeError(f"Body placeholder {placeholders['body']!r} not in template")

    output = template.replace(placeholders["abstract"], abstract_text)
    output = output.replace(placeholders["body"], body_text)

    actions: list[dict] = []
    action = "would write" if dry_run else "wrote"
    if not dry_run:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(output)
    actions.append(
        {
            "source": str(src_path.relative_to(PROJECT_ROOT)),
            "target": str(target_path.relative_to(PROJECT_ROOT)),
            "action": f"{action} main.tex ({len(output):,} bytes)",
        }
    )
    log.info(
        "%s %s (%d bytes)", action, target_path.relative_to(PROJECT_ROOT), len(output)
    )

    actions.extend(_sync_assets(cfg, dry_run=dry_run))

    tbl_scale = cfg.get(
        "scale_single_column_tables", cfg.get("scale_single_column_figures")
    )
    if tbl_scale and not dry_run:
        tables_dir = target_path.parent / "tables"
        n_tables = _scale_single_column_table_files(tables_dir, float(tbl_scale))
        if n_tables:
            actions.append(
                {
                    "source": "tables/*.tex",
                    "target": str(tables_dir.relative_to(PROJECT_ROOT)),
                    "action": f"scaled {n_tables} single-column table(s) by {tbl_scale}",
                }
            )

    warnings = _validate(output, target_path.parent, cfg)
    for w in warnings:
        log.warning(w)
    if warnings:
        actions.append(
            {
                "source": "validation",
                "target": "main.tex",
                "action": f"{len(warnings)} warning(s)",
            }
        )

    return actions


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build arXiv version of the EngiAI paper from IDETC revision"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Preview actions without writing files"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to YAML config (default: {DEFAULT_CONFIG})",
    )
    args = parser.parse_args()

    if not args.config.exists():
        log.error("Config not found: %s", args.config)
        return 1

    cfg = load_config(args.config)
    try:
        actions = build_arxiv(cfg, dry_run=args.dry_run)
    except RuntimeError:
        log.exception("build_arxiv failed")
        return 1

    print(
        f"\n{'Would perform' if args.dry_run else 'Performed'} {len(actions)} action(s):"
    )
    for a in actions:
        print(f"  {a['action']}: {a['source']} -> {a['target']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
