# ComfyColab Video

ComfyColab Video is a standalone ComfyUI custom node for LTX-2.3 text-to-video
and image-to-video generation. It also remains compatible with the managed
[ComfyColab](https://github.com/DragonLord1998/ComfyColab) pack runtime.

The facade supports text-to-video and optional first-frame image conditioning,
selectable LTX-2.3 Distilled 1.1 GGUF variants, native synchronized audio,
24/48 FPS output, and optional 1.5x or 2x latent spatial upscaling. It returns
native ComfyUI `VIDEO`, decoded `IMAGE` frames, and `AUDIO`.

## Standalone installation

Install the Git repository through ComfyUI Manager, or install manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/DragonLord1998/ComfyColab-Video.git
cd ComfyColab-Video
python install.py
```

Use the Python executable that starts ComfyUI, then restart ComfyUI. The
installer adds exact pinned sibling checkouts of ComfyUI-GGUF and the official
Lighttricks ComfyUI-LTXVideo extension only when they are absent. Existing
exact-pinned checkouts are reused; different or non-git installations produce
an actionable error and are never overwritten. Model weights remain lazy,
checksum-verified downloads.

## Managed ComfyColab installation

After an immutable daughter commit is added to the official core registry and
the resulting lock passes live validation, use:

```bash
comfycolab start --pack video
```

`comfycolab-pack.json` declares exact revisions of both ComfyUI-GGUF and the
official Lighttricks ComfyUI-LTXVideo extension. When Image and Video are
selected together, the resolver can deduplicate their identical GGUF
dependency rather than installing it twice.

The managed resolver deduplicates ComfyUI-GGUF when Image and Video are enabled
together.

## Workflow

`workflows/comfycolab_ltx23_text_image_to_video.json` is the preserved example
workflow. It contains one public facade, an optional disconnected `LoadImage`,
and a native `SaveVideo` output.

## Model provisioning

`custom_nodes/ComfyColab-LTXVideo/catalog/ltx_2_3.json` records immutable
revisions, byte sizes, and SHA-256 values for selected GGUFs, the Gemma text
encoder, connector, video/audio VAEs, and spatial/temporal upscalers. Only
assets required by the visible node settings are downloaded.

Model downloads use authenticated Hugging Face Hub with `hf-xet` as the primary
transport and enable `HF_XET_HIGH_PERFORMANCE=1` automatically. Public models
still work anonymously when no `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` is set.
If Hub/Xet is unavailable or fails, the pack falls back to its checksum-verified
`urllib` downloader with resumable `.part` files.

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
