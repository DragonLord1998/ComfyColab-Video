from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .catalog_h3 import h3_filenames_by_role, normalize_h3_variant, selected_h3_assets
from .download import download_file
from .models import _ComfyProgress


def _first_model_path(folder_paths: Any, key: str) -> Path:
    paths = folder_paths.get_folder_paths(key)
    if not paths:
        raise RuntimeError(f"ComfyUI has no configured model folder for '{key}'.")
    destination = Path(paths[0])
    destination.mkdir(parents=True, exist_ok=True)
    return destination


def ensure_h3_model_assets(
    variant: str,
    *,
    force_redownload: bool = False,
) -> dict[str, str]:
    folder_paths = importlib.import_module("folder_paths")
    assets = selected_h3_assets(normalize_h3_variant(variant))
    progress = _ComfyProgress()
    filenames: dict[str, str] = {}
    for role, specification in assets.items():
        destination = (
            _first_model_path(folder_paths, specification["folder_key"])
            / specification["filename"]
        )
        download_file(
            url=specification["url"],
            destination=destination,
            expected_sha256=specification["sha256"],
            expected_size=specification["size_bytes"],
            force=force_redownload,
            progress=progress,
        )
        filenames[role] = specification["filename"]
    return filenames


def h3_model_filenames(variant: str) -> dict[str, str]:
    return h3_filenames_by_role(variant)
