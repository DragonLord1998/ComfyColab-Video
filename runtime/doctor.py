#!/usr/bin/env python3
"""Read-only health inspection for ComfyColab Video."""

from __future__ import annotations

import json
from pathlib import Path


PUBLIC_NODE_ID = "ComfyColabLTX23Video"


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    node_root = root / "custom_nodes" / "ComfyColab-LTXVideo"
    node_source = (node_root / "nodes.py").read_text(encoding="utf-8")
    checks = {
        "node_root": node_root.is_dir(),
        "public_node_id": PUBLIC_NODE_ID in node_source,
        "catalog": (node_root / "catalog" / "ltx_2_3.json").is_file(),
        "workflow": (
            root / "workflows" / "comfycolab_ltx23_text_image_to_video.json"
        ).is_file(),
    }
    status = "ok" if all(checks.values()) else "error"
    print(
        json.dumps(
            {"schema": 1, "hook": "doctor", "status": status, "checks": checks},
            sort_keys=True,
        )
    )
    return 0 if status == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
