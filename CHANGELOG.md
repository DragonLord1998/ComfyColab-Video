# Changelog

All notable changes to ComfyColab Video are recorded here.

This project follows semantic versioning. Development versions are staging
artifacts and are not runtime-installable releases unless the release entry says
otherwise.

## Unreleased

### Added

- Added a repository-root V3 entrypoint for direct installation under
  `ComfyUI/custom_nodes`.
- Added a Manager-compatible installer for pinned ComfyUI-GGUF and the official
  ComfyUI-LTXVideo dependency, plus node inventory and standalone checks.
- Pinned Kornia 0.8.1 during standalone installation because the pinned
  upstream LTXVideo extension imports a helper removed in Kornia 0.8.2.
- Existing sibling dependencies must match their audited revisions before the
  installer reports success.

### Changed

- Declared each pinned Git dependency's own `requirements.txt` for
  dependency-owned installation by the generic core runtime.

### Validation

- Discovered the public video facade on stock ComfyUI 0.28.0 and completed a
  live G4 LTX-2.3 dancing-cat MP4 generation.

### Required before the first runtime-installable release

- Publish an immutable daughter commit and add it to the official core registry.
- Generate an exact ComfyColab lock and complete a live Colab smoke run against
  that lock.
- Record accelerator, peak-memory, runtime, audiovisual synchronization, and
  representative output evidence separately from local test results.

## 0.1.0-dev1 - 2026-07-18

### Added

- Extracted the legacy `ComfyColab-LTXVideo` node root into an independent Video
  pack while preserving the public `ComfyColabLTX23Video` facade.
- Added pinned ComfyUI-GGUF and ComfyUI-LTXVideo dependencies, a
  checksum-verified LTX-2.3 model catalog, offline hooks, health checks, and
  pack probes.
- Preserved the text/image-to-video example workflow and added local graph,
  catalog, download, node-schema, and workflow tests.

### Known limitations

- Local validation does not prove Colab memory fit, runtime, audiovisual
  synchronization, or output quality.
