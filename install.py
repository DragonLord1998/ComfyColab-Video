#!/usr/bin/env python3
"""Install pinned standalone ComfyUI dependencies for ComfyColab Video."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEPENDENCIES = (
    (
        "ComfyUI-GGUF",
        "https://github.com/city96/ComfyUI-GGUF.git",
        "6ea2651e7df66d7585f6ffee804b20e92fb38b8a",
    ),
    (
        "ComfyUI-LTXVideo",
        "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
        "aceeae9635f6d493f2893ba3c411a1c36031788a",
    ),
)


def _run(*argv: str, cwd: Path | None = None) -> None:
    subprocess.run(
        list(argv),
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def _custom_nodes_root() -> Path:
    parent = ROOT.parent
    if parent.name != "custom_nodes":
        raise RuntimeError(
            "ComfyColab-Video must be directly inside ComfyUI/custom_nodes "
            "before install.py is run."
        )
    return parent


def _commit(path: Path) -> str | None:
    if not (path / ".git").is_dir():
        return None
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _install_dependency(name: str, repository: str, revision: str) -> Path:
    target = _custom_nodes_root() / name
    if target.exists():
        actual = _commit(target)
        if actual != revision:
            raise RuntimeError(
                f"Existing {name} is not the audited revision required by "
                f"ComfyColab Video. Expected {revision}, found "
                f"{actual or 'a non-git installation'}. Move or update that "
                "checkout explicitly, then rerun this installer."
            )
        print(
            f"[ComfyColab Video] Existing pinned {name} reused "
            f"(revision: {actual})."
        )
        return target
    _run("git", "clone", "--filter=blob:none", repository, str(target))
    _run("git", "fetch", "origin", revision, "--depth", "1", cwd=target)
    _run("git", "checkout", "--detach", "FETCH_HEAD", cwd=target)
    requirements = target / "requirements.txt"
    if requirements.is_file():
        _run(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "-r",
            str(requirements),
        )
    return target


def _install_ltx_compatibility(target: Path) -> None:
    if _commit(target) != DEPENDENCIES[1][2]:
        return
    # The pinned LTXVideo revision imports ``pad`` from Kornia's pyramid module.
    # Kornia 0.8.2 removed that re-export; 0.8.1 remains compatible.
    _run(
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        "kornia==0.8.1",
    )


def _check() -> None:
    custom_nodes = _custom_nodes_root()
    checks = {
        name: (
            (custom_nodes / name / "__init__.py").is_file()
            and _commit(custom_nodes / name) == revision
        )
        for name, _repository, revision in DEPENDENCIES
    }
    checks["video_entrypoint"] = (ROOT / "__init__.py").is_file()
    print(json.dumps(checks, indent=2, sort_keys=True))
    if not all(checks.values()):
        raise SystemExit(1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not args.check:
        installed = {
            name: _install_dependency(name, repository, revision)
            for name, repository, revision in DEPENDENCIES
        }
        _install_ltx_compatibility(installed["ComfyUI-LTXVideo"])
    _check()
    if not args.check:
        print("[ComfyColab Video] Standalone installation complete. Restart ComfyUI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
