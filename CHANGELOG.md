# Changelog

All notable changes to ComfyColab Video are recorded here.

This project follows semantic versioning. Development versions are staging
artifacts and are not runtime-installable releases unless the release entry says
otherwise.

## Unreleased

### Required before the first runtime-installable release

- Resolve Python requirements from the pinned ComfyUI-GGUF and
  ComfyUI-LTXVideo checkouts into the manifest's main `environments`
  declaration.
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

- The pack manifest does not yet declare normalized main-environment Python
  requirements from the pinned ComfyUI-GGUF and ComfyUI-LTXVideo checkouts.
  ComfyColab must not infer or install undeclared upstream `requirements.txt`
  files; this development version is therefore not a runtime-installable
  release.
- Local validation does not prove Colab memory fit, runtime, audiovisual
  synchronization, or output quality.
