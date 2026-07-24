"""
@author: DragonLord1998
@title: ComfyColab Video
@nickname: ComfyColab Video
@description: Standalone LTX-2.3 text and image to video generation.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def _load_internal_package():
    name = f"{__name__}._ltxvideo"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    package_dir = ROOT / "custom_nodes" / "ComfyColab-LTXVideo"
    spec = importlib.util.spec_from_file_location(
        name,
        package_dir / "__init__.py",
        submodule_search_locations=[str(package_dir)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load internal node package: {package_dir}")
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    try:
        spec.loader.exec_module(package)
    except BaseException:
        sys.modules.pop(name, None)
        raise
    return package


async def comfy_entrypoint():
    package = _load_internal_package()
    extension = package.comfy_entrypoint()
    if inspect.isawaitable(extension):
        extension = await extension
    return extension


__all__ = ["comfy_entrypoint"]
