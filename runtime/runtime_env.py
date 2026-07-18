#!/usr/bin/env python3
"""Return the Video pack's contributed runtime environment."""

from __future__ import annotations

import json


def main() -> int:
    print(
        json.dumps(
            {
                "schema": 1,
                "hook": "runtime_env",
                "status": "ok",
                "environment": {},
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
