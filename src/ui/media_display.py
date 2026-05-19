"""
Media display functions for the Streamlit UI.

Handles display of images, STL files, and log files in chat messages.
"""

import contextlib
import re
from pathlib import Path

import streamlit as st
from PIL import Image
from streamlit_stl import stl_from_file  # type: ignore[import-untyped]

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent


def find_images_in_text(text: str) -> list[Path]:
    """Find image file paths mentioned in text.

    Args:
        text: Text that may contain file paths

    Returns:
        List of valid image file paths
    """
    image_extensions = {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg"}
    image_paths: list[Path] = []

    # Regex patterns to catch:
    # - explicit outputs/ or assets/ or artifacts/ paths
    # - bare filenames like image.png
    # - markdown image syntax: ![alt](path.png)
    # - HTML <img src="path.png">
    patterns = [
        r"outputs/[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"assets/[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"artifacts/[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"/[^\s)\"]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"[\w\-_.]+\.(?:png|jpg|jpeg|gif|bmp|svg)",
        r"!\[[^\]]*\]\(([^)]+\.(?:png|jpg|jpeg|gif|bmp|svg))\)",
        r"<img[^>]+src=[\"']([^\"']+\.(?:png|jpg|jpeg|gif|bmp|svg))[\"']",
    ]

    seen: set[str] = set()

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            # re groups may return tuples for some patterns, normalize
            candidate_str = m[0] if isinstance(m, tuple) else m

            path = Path(candidate_str)
            # Expand user home if present
            with contextlib.suppress(Exception):
                path = Path(str(path).replace("~", str(Path.home())))

            # If not absolute, try sensible locations: project root, outputs, assets, artifacts
            candidates = [path]
            if not path.is_absolute():
                candidates.extend(
                    [
                        project_root / candidate_str,
                        project_root / "outputs" / candidate_str,
                        project_root / "assets" / candidate_str,
                        project_root / "artifacts" / candidate_str,
                    ]
                )

            for candidate in candidates:
                try:
                    if (
                        candidate.exists()
                        and candidate.suffix.lower() in image_extensions
                    ):
                        key = str(candidate.resolve())
                        if key not in seen:
                            image_paths.append(candidate)
                            seen.add(key)
                        break
                except Exception:
                    # ignore resolution errors and continue
                    continue

    return image_paths


def find_stl_files_in_text(text: str) -> list[Path]:
    """Find STL file paths mentioned in text.

    Args:
        text: Text that may contain file paths

    Returns:
        List of valid STL file paths (only files that currently exist)
    """
    stl_paths = []

    # Look for STL file patterns
    patterns = [
        r"outputs/[\w\-_.]+\.stl",
        r"[\w\-_.]+\.stl",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            # Try as absolute path first
            path = Path(match)
            if not path.is_absolute():
                # Try relative to project root
                path = project_root / match

            # Only add if file exists and is valid STL
            if path.exists() and path.suffix.lower() == ".stl":
                stl_paths.append(path)

    return list(set(stl_paths))  # Remove duplicates


def find_log_files_in_text(text: str) -> list[Path]:
    """Find .err and .out log files mentioned in text or associated with downloaded jobs.

    Args:
        text: Text that may contain file paths or job IDs

    Returns:
        List of valid log file paths (.err and .out files)
    """
    log_paths = []
    seen: set[str] = set()

    # Look for explicit .err and .out file patterns
    patterns = [
        r"outputs/[\w\-_.]+\.(?:err|out)",
        r"[\w\-_.]+\.(?:err|out)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            path = Path(match)
            if not path.is_absolute():
                path = project_root / match

            if path.exists() and path.suffix.lower() in [".err", ".out"]:
                key = str(path.resolve())
                if key not in seen:
                    log_paths.append(path)
                    seen.add(key)

    # Also look for job IDs and find their corresponding log files
    job_id_pattern = (
        r"job[_ ](?:id|ID)[:\s]*(\d+)|Job ID: (\d+)|job_id[\"']?\s*:\s*[\"']?(\d+)"
    )
    job_matches = re.findall(job_id_pattern, text)

    for match_groups in job_matches:
        # Extract the actual job ID from the groups
        job_id = next((m for m in match_groups if m), None)
        if job_id:
            # Look for log files matching this job ID in outputs directory
            outputs_dir = project_root / "outputs"
            if outputs_dir.exists():
                err_files = list(outputs_dir.glob(f"*{job_id}.err"))
                out_files = list(outputs_dir.glob(f"*{job_id}.out"))
                for log_file in err_files + out_files:
                    key = str(log_file.resolve())
                    if key not in seen:
                        log_paths.append(log_file)
                        seen.add(key)

    return log_paths


def find_slurm_files_in_text(text: str) -> list[Path]:
    """Find SLURM script files (.slurm) mentioned in text.

    Args:
        text: Text that may contain file paths

    Returns:
        List of valid SLURM file paths
    """
    slurm_paths = []
    seen: set[str] = set()

    # Look for .slurm file patterns
    patterns = [
        r"outputs/[\w\-_.]+\.slurm",
        r"[\w\-_.]+\.slurm",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            path = Path(match)
            if not path.is_absolute():
                path = project_root / match

            if path.exists() and path.suffix.lower() == ".slurm":
                key = str(path.resolve())
                if key not in seen:
                    slurm_paths.append(path)
                    seen.add(key)

    return slurm_paths


def save_file_to_server(file_path: Path, save_dir_path: Path) -> bool:
    """Save a file to the server directory.

    Args:
        file_path: Source file path
        save_dir_path: Destination directory path

    Returns:
        True if successful, False otherwise
    """
    try:
        save_dir_path.mkdir(parents=True, exist_ok=True)
        dest = save_dir_path / file_path.name
        dest.write_bytes(file_path.read_bytes())
    except Exception:
        return False
    else:
        return True


def auto_save_if_enabled(file_path: Path) -> None:
    """Automatically save file if auto-save is enabled in session state.

    Args:
        file_path: File to save
    """
    if st.session_state.get("media_auto_save", False):
        save_dir = Path(st.session_state.media_save_dir)
        save_file_to_server(file_path, save_dir)


def render_download_save_controls(file_path: Path, button_key_prefix: str) -> None:
    """Render download and save buttons for a media file.

    Args:
        file_path: Path to the file
        button_key_prefix: Unique prefix for button keys
    """
    # Create unique key based on file path hash and prefix
    unique_key_base = f"{button_key_prefix}_{abs(hash(str(file_path)))}"

    # Determine MIME type based on file extension
    mime_types = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".stl": "application/sla",
        ".slurm": "text/plain",
        ".sh": "text/plain",
    }
    mime = mime_types.get(file_path.suffix.lower(), "application/octet-stream")

    # Create two columns with equal width for buttons
    cols = st.columns(2)

    # Download button
    with cols[0]:
        st.download_button(
            label="📥 Download",
            data=file_path.read_bytes(),
            file_name=file_path.name,
            mime=mime,
            key=f"{unique_key_base}_download",
            use_container_width=True,
        )

    # Save to server button
    with cols[1]:
        if st.button(
            "💾 Save to server",
            key=f"{unique_key_base}_save",
            use_container_width=True,
        ):
            save_dir = Path(st.session_state.media_save_dir)
            if save_file_to_server(file_path, save_dir):
                st.success(f"✅ Saved to {save_dir / file_path.name}")
            else:
                st.error(f"❌ Could not save {file_path.name}")

    # Auto-save
    auto_save_if_enabled(file_path)


def display_image(img_path: Path, button_key_prefix: str) -> None:
    """Display an image with download/save controls.

    Args:
        img_path: Path to image file
        button_key_prefix: Unique prefix for button keys
    """
    # Skip if file doesn't exist (e.g., from previous sessions)
    if not img_path.exists():
        return

    try:
        image = Image.open(img_path)
        st.image(image, caption=img_path.name, width=500)
        render_download_save_controls(img_path, button_key_prefix)
    except FileNotFoundError:
        # Silently skip - file was deleted
        pass
    except Exception as e:
        # Only show warning for unexpected errors
        if "MediaFileStorageError" not in str(type(e).__name__):
            st.warning(f"Could not display image {img_path.name}: {e}")


def display_stl(stl_path: Path, idx: int, button_key_prefix: str) -> None:
    """Display an STL file with viewer and download/save controls.

    Args:
        stl_path: Path to STL file
        idx: Index for unique key generation
        button_key_prefix: Unique prefix for button keys
    """
    if not stl_path.exists():
        # Silently skip missing STL files (e.g., from previous sessions)
        return

    try:
        st.markdown(f"**3D Model: {stl_path.name}**")
        # Include button_key_prefix to make key unique across messages
        stable_key = f"{button_key_prefix}_stl_{abs(hash(str(stl_path)))}_{idx}"
        stl_from_file(
            file_path=str(stl_path),
            color=st.session_state.stl_color,
            material=st.session_state.stl_material,
            auto_rotate=st.session_state.stl_auto_rotate,
            height=st.session_state.stl_height,
            opacity=st.session_state.stl_opacity,
            shininess=st.session_state.stl_shininess,
            key=stable_key,
        )
        render_download_save_controls(stl_path, button_key_prefix)
    except FileNotFoundError:
        # Silently skip - file was deleted
        pass
    except Exception as e:
        st.warning(f"Could not display 3D model {stl_path.name}: {e}")


def display_log_file(log_path: Path, button_key_prefix: str) -> None:
    """Display a log file (.err or .out) with expandable content and download controls.

    Args:
        log_path: Path to log file
        button_key_prefix: Unique prefix for button keys
    """
    if not log_path.exists():
        st.info(f"Log file not found: {log_path.name}")
        return

    try:
        # Read log file content
        content = log_path.read_text(encoding="utf-8", errors="replace")

        # Determine file type for styling
        file_type = log_path.suffix.lower()
        icon = "❌" if file_type == ".err" else "📄"
        label = "Error Log" if file_type == ".err" else "Output Log"

        # Display log file in an expander
        with st.expander(f"{icon} **{label}: {log_path.name}**", expanded=False):
            if content.strip():
                # Show first 100 lines by default, full content in code block
                lines = content.split("\n")
                preview_lines = 100

                if len(lines) > preview_lines:
                    st.caption(
                        f"Showing first {preview_lines} lines of {len(lines)} total lines"
                    )
                    st.code("\n".join(lines[:preview_lines]), language="text")

                    # Option to show full content
                    if st.button(
                        "Show full content",
                        key=f"{button_key_prefix}_{abs(hash(str(log_path)))}_full",
                    ):
                        st.code(content, language="text")
                else:
                    st.code(content, language="text")
            else:
                st.info("Log file is empty")

            # Download button for log file
            cols = st.columns([1, 3])
            with cols[0]:
                st.download_button(
                    label=f"Download {file_type} file",
                    data=content,
                    file_name=log_path.name,
                    mime="text/plain",
                    key=f"{button_key_prefix}_{abs(hash(str(log_path)))}_download",
                )

        # Auto-save if enabled
        auto_save_if_enabled(log_path)

    except Exception as e:
        st.warning(f"Could not display log file {log_path.name}: {e}")


def display_slurm_file(slurm_path: Path, button_key_prefix: str) -> None:
    """Display a SLURM script file with syntax highlighting and download controls.

    Args:
        slurm_path: Path to SLURM script file
        button_key_prefix: Unique prefix for button keys
    """
    if not slurm_path.exists():
        st.info(f"SLURM script not found: {slurm_path.name}")
        return

    try:
        # Read SLURM script content
        content = slurm_path.read_text(encoding="utf-8", errors="replace")

        # Display SLURM script header
        st.markdown(f"### 📋 SLURM Script: `{slurm_path.name}`")

        # Display the full script content
        if content.strip():
            st.code(content, language="bash")
        else:
            st.info("SLURM script is empty")

        # Download and save controls
        render_download_save_controls(slurm_path, button_key_prefix)

        # Auto-save if enabled
        auto_save_if_enabled(slurm_path)

    except Exception as e:
        st.warning(f"Could not display SLURM script {slurm_path.name}: {e}")


def display_response_media(response_text: str) -> None:
    """Display images, STL files, SLURM scripts, and log files found in response text.

    Args:
        response_text: The response text to scan for media files
    """
    # Display images
    images = find_images_in_text(response_text)
    for img_path in images:
        display_image(img_path, button_key_prefix="resp_img")

    # Display STL files
    stl_files = find_stl_files_in_text(response_text)
    for idx, stl_path in enumerate(stl_files):
        display_stl(stl_path, idx, button_key_prefix="resp_stl")

    # Display SLURM scripts
    slurm_files = find_slurm_files_in_text(response_text)
    for slurm_path in slurm_files:
        display_slurm_file(slurm_path, button_key_prefix="resp_slurm")

    # Display log files (.err and .out)
    log_files = find_log_files_in_text(response_text)
    for log_path in log_files:
        display_log_file(log_path, button_key_prefix="resp_log")
