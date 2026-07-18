#!/usr/bin/env python3
"""Offline configure hook for the declarative Video pack."""

from __future__ import annotations

import json


def main() -> int:
    print(
        json.dumps(
            {
                "schema": 1,
                "hook": "configure",
                "status": "ok",
                "changes": [],
                "readiness": {"configured": True},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
