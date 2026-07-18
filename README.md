# ComfyColab Video

ComfyColab Video is the optional video-generation pack for
[ComfyColab](https://github.com/DragonLord1998/ComfyColab). Its first release
preserves the existing `custom_nodes/ComfyColab-LTXVideo` node root and the
public `ComfyColabLTX23Video` facade.

> Development staging status: the source, manifest, hooks, workflow, and
> offline contract suite are complete, but the v1 manifest does not yet contain
> normalized main-environment Python requirements from the pinned
> ComfyUI-GGUF and ComfyUI-LTXVideo checkouts. Core correctly refuses to
> auto-discover undeclared `requirements.txt` files. Add the resolved
> requirements to `environments` before publishing a runtime-installable
> release.

The facade supports text-to-video and optional first-frame image conditioning,
selectable LTX-2.3 Distilled 1.1 GGUF variants, native synchronized audio,
24/48 FPS output, and optional 1.5x or 2x latent spatial upscaling. It returns
native ComfyUI `VIDEO`, decoded `IMAGE` frames, and `AUDIO`.

## Installation after publication

After normalized upstream requirements are declared and an immutable daughter
commit is added to the official core registry, use:

```bash
comfycolab start --pack video
```

`comfycolab-pack.json` declares exact revisions of both ComfyUI-GGUF and the
official Lighttricks ComfyUI-LTXVideo extension. When Image and Video are
selected together, the resolver can deduplicate their identical GGUF
dependency rather than installing it twice.

During development, core integration uses an explicit authenticated
`--pack-ref` file rather than the currently unpublished `video` alias.

## Workflow

`workflows/comfycolab_ltx23_text_image_to_video.json` is the preserved example
workflow. It contains one public facade, an optional disconnected `LoadImage`,
and a native `SaveVideo` output.

## Model provisioning

`custom_nodes/ComfyColab-LTXVideo/catalog/ltx_2_3.json` records immutable
revisions, byte sizes, and SHA-256 values for selected GGUFs, the Gemma text
encoder, connector, video/audio VAEs, and spatial/temporal upscalers. Only
assets required by the visible node settings are downloaded.

## Validation

```bash
PYTHON=python3 bash scripts/check.sh
python3 scripts/verify_catalogs.py
```

The offline check covers manifest conformance, hook output, catalog selection,
graph branches, optional image conditioning, dependency diagnosis, public
schema, and workflow wiring. The explicit verifier checks pinned Hugging Face
metadata over the network. Live Colab execution remains a separate gate for
VRAM use, audiovisual synchronization, runtime, and output quality.
