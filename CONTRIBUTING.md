# Contributing to ComfyColab Video

ComfyColab Video owns video-generation nodes, model catalogs, workflows,
licenses, and video-specific optimizations. Changes to the generic
ComfyUI-on-Colab installer or pack resolver belong in the ComfyColab core
repository.

## Development contract

- Preserve the public `ComfyColabLTX23Video` node ID unless a documented
  breaking release intentionally changes it.
- Keep every Git dependency on an immutable 40-character commit.
- Keep downloadable model assets on immutable revisions with expected byte
  sizes and SHA-256 checksums.
- Declare dependencies and Python or system requirements in
  `comfycolab-pack.json`. Runtime code must not discover or install undeclared
  upstream requirements.
- Keep bootstrap hooks offline (`network: "none"`) and within their declared
  write roots.
- Update the example workflow when the public node schema changes.
- Update `THIRD_PARTY_NOTICES.md` whenever code, model, or data licensing
  changes.
- Keep the version in `comfycolab-pack.json` equal to `[project].version` in
  `pyproject.toml`.

## Local validation

Run the offline suite from this repository's root:

```bash
PYTHON=python3 bash scripts/check.sh
```

This validates Python syntax, the manifest and hook contracts, model selection,
download behavior, graph branches, the public node schema, and workflow wiring.
It does not load CUDA models and does not establish that the pack is installable
by the ComfyColab runtime.

The catalog metadata verifier is a separate, networked check:

```bash
python3 scripts/verify_catalogs.py
```

Record its result separately from the offline suite because remote metadata can
change or become unavailable.

## Live validation

A release candidate needs a bounded Colab run using the exact generated lock,
not a mutable branch or an ad hoc install. Record at least:

- core, ComfyUI, pack, and dependency commits plus the lock SHA-256;
- accelerator and Python/CUDA environment;
- clean install and ComfyUI startup results;
- `ComfyColabLTX23Video` present in `/object_info`;
- representative text-to-video and image-conditioned runs;
- output frame rate, dimensions, duration, audio presence and synchronization;
- peak memory, wall-clock time, produced artifacts, and any unproven path.

Local green tests and a live Colab run are different evidence tiers. Do not
describe local checks as live-runtime proof.

## Current release blocker

`0.1.0-dev1` now declares each pinned Git dependency's own `requirements.txt`,
so the generic core runtime owns those installations. Before publishing a
runtime-installable version, publish and register an immutable daughter commit,
generate a new lock, and run the live validation above. Do not add a second
imperative `pip install -r` path to a hook or node.

## Pull-request checklist

- Manifest and `pyproject.toml` versions agree.
- New dependencies, assets, and licenses are immutable and documented.
- `PYTHON=python3 bash scripts/check.sh` passes.
- Network verification is reported separately when run.
- Live Colab evidence is attached for runtime-affecting changes, or the missing
  live gate is stated explicitly.
- `CHANGELOG.md` records user-visible behavior and remaining limitations.
