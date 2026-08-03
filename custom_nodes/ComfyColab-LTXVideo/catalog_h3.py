from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


CATALOG_PATH = Path(__file__).with_name("catalog") / "minimax_h3.json"
H3_REVISION = "0543966fbdce5ba05709a8f2031c94bdba629b4a"
FL2VA = "FL2VA"
REF2VA = "Ref2VA"
VARIANT_LABELS = {
    FL2VA: "FL2VA - Text / First / Last Frame",
    REF2VA: "Ref2VA - Reference Images / Video / Audio",
}
LABEL_TO_VARIANT = {label: variant for variant, label in VARIANT_LABELS.items()}


class H3CatalogError(RuntimeError):
    pass


def _validate_file(name: str, value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise H3CatalogError(f"H3 catalog field '{name}' must be an object.")
    required = {"filename", "folder_key", "url", "sha256", "size_bytes"}
    missing = sorted(required - value.keys())
    if missing:
        raise H3CatalogError(
            f"H3 catalog field '{name}' is missing: {', '.join(missing)}"
        )
    url = str(value["url"])
    if not url.startswith("https://huggingface.co/") or "/resolve/main/" in url:
        raise H3CatalogError(
            f"H3 catalog field '{name}' must use a revision-pinned Hugging Face URL."
        )
    if f"/resolve/{H3_REVISION}/" not in url:
        raise H3CatalogError(f"H3 catalog field '{name}' uses the wrong revision.")
    sha256 = str(value["sha256"])
    if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
        raise H3CatalogError(f"H3 catalog field '{name}' has an invalid SHA-256.")
    if not isinstance(value["size_bytes"], int) or value["size_bytes"] <= 0:
        raise H3CatalogError(f"H3 catalog field '{name}' has an invalid byte size.")
    if not str(value["filename"]) or not str(value["folder_key"]):
        raise H3CatalogError(f"H3 catalog field '{name}' has an empty filename/folder.")
    return value


@lru_cache(maxsize=1)
def load_h3_catalog() -> dict[str, Any]:
    try:
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise H3CatalogError(f"Unable to read the MiniMax H3 catalog: {error}") from error
    if catalog.get("schema_version") != 1:
        raise H3CatalogError("Unsupported MiniMax H3 catalog schema version.")
    if catalog.get("repository") != "Comfy-Org/MiniMax-H3":
        raise H3CatalogError("MiniMax H3 catalog must use Comfy-Org/MiniMax-H3.")
    if catalog.get("revision") != H3_REVISION:
        raise H3CatalogError("MiniMax H3 catalog revision is not the approved pin.")
    variants = catalog.get("variants")
    shared = catalog.get("shared")
    if not isinstance(variants, dict) or set(variants) != {FL2VA, REF2VA}:
        raise H3CatalogError("MiniMax H3 catalog must define FL2VA and Ref2VA.")
    if not isinstance(shared, dict) or set(shared) != {
        "text_encoder",
        "video_vae",
        "audio_vae",
    }:
        raise H3CatalogError("MiniMax H3 catalog has the wrong shared assets.")
    for variant, value in variants.items():
        if not isinstance(value, dict):
            raise H3CatalogError(f"MiniMax H3 variant '{variant}' must be an object.")
        _validate_file(f"variants.{variant}.model", value.get("model"))
    for role, value in shared.items():
        _validate_file(f"shared.{role}", value)
    return catalog


def h3_variant_labels() -> list[str]:
    return [VARIANT_LABELS[FL2VA], VARIANT_LABELS[REF2VA]]


def normalize_h3_variant(value: str) -> str:
    candidate = str(value)
    if candidate in VARIANT_LABELS:
        return candidate
    if candidate in LABEL_TO_VARIANT:
        return LABEL_TO_VARIANT[candidate]
    raise H3CatalogError(f"Unknown MiniMax H3 variant: {value}")


def selected_h3_assets(variant: str) -> dict[str, dict[str, Any]]:
    normalized = normalize_h3_variant(variant)
    catalog = load_h3_catalog()
    return {"model": catalog["variants"][normalized]["model"], **catalog["shared"]}


def h3_filenames_by_role(variant: str) -> dict[str, str]:
    return {
        role: specification["filename"]
        for role, specification in selected_h3_assets(variant).items()
    }


def h3_variant_size_bytes(variant: str) -> int:
    return sum(item["size_bytes"] for item in selected_h3_assets(variant).values())


def h3_shared_size_bytes() -> int:
    return sum(item["size_bytes"] for item in load_h3_catalog()["shared"].values())


def h3_both_variants_size_bytes() -> int:
    catalog = load_h3_catalog()
    return h3_shared_size_bytes() + sum(
        item["model"]["size_bytes"] for item in catalog["variants"].values()
    )
