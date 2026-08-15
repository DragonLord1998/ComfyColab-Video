# ComfyColab Video

ComfyColab Video is a standalone ComfyUI custom node for LTX-2.3 text-to-video
and image-to-video generation plus MiniMax H3 text/image-to-audio-video and
reference-to-audio-video generation. It also remains compatible with the managed
[ComfyColab](https://github.com/DragonLord1998/ComfyColab) pack runtime.

The facade supports text-to-video and optional first-frame image conditioning,
selectable LTX-2.3 Distilled 1.1 GGUF variants, native synchronized audio,
24/48 FPS output, and optional 1.5x or 2x latent spatial upscaling. It returns
native ComfyUI `VIDEO`, decoded `IMAGE` frames, and `AUDIO`.

The MiniMax H3 facades expose:

- `MiniMax H3 Bundle Loader`, which downloads one optimized FL2VA or Ref2VA
  transformer plus the three shared H3 components and returns one typed
  `MINIMAX_H3_BUNDLE` cable plus raw `MODEL`, `CLIP`, `VAE`, and `VAE` outputs.
- `ComfyColab MiniMax H3 - Prompt Enhancer`, which rewrites a plain user prompt
  into the official MiniMax H3 Base or Ref2VA prompt structure before the prompt
  is connected to an H3 generation node.
- `ComfyColab MiniMax H3 - Text/Image to Video`, which accepts an FL2VA bundle,
  a prompt, and optional first/last frames.
- `ComfyColab MiniMax H3 - Reference to Video`, which accepts a Ref2VA bundle
  and ordered reference images, videos, paired video soundtracks, and standalone
  audio.

H3 output is 24 FPS video with native synchronized 32 kHz stereo audio. The
local Base path is 768p-class and capped at `768 x 1344` pixels of area; this
pack does not claim local 2K regeneration, hosted Context-IR, sparse attention,
unrestricted geography, or OSI-open-source weights.

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

After the immutable daughter commit is added to the official core registry, use:

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

`workflows/comfycolab_ltx23_text_image_to_video.json` is the preserved LTX
example workflow. It contains one public facade, an optional disconnected
`LoadImage`, and a native `SaveVideo` output.

`workflows/comfycolab_minimax_h3_text_image_to_video.json` contains one H3
FL2VA loader, one H3 Text/Image facade, disconnected first/last `LoadImage`
examples, and one native `SaveVideo`.

`workflows/comfycolab_minimax_h3_reference_to_video.json` contains one H3
Ref2VA loader, one H3 Reference facade, a reference image branch, a reference
video branch with paired audio, a standalone audio branch, and one native
`SaveVideo`. Prompt tags are one-based and connection-ordered: `<Picture 1>`,
`<Video 1>`, and `<Audio 1>`.

`workflows/comfycolab_minimax_h3_fl2va_to_ref2va_chain.json` is a proof
workflow that uses separate FL2VA and Ref2VA loaders, then routes the FL2VA
facade's decoded `frames` and `audio` outputs into the Ref2VA autogrow
`ref_videos.ref_video_0` and `ref_video_audios.ref_video_audio_0` inputs before
saving the Ref2VA video.

## Model provisioning

`custom_nodes/ComfyColab-LTXVideo/catalog/ltx_2_3.json` records immutable
revisions, byte sizes, and SHA-256 values for selected GGUFs, the Gemma text
encoder, connector, video/audio VAEs, and spatial/temporal upscalers. Only
assets required by the visible node settings are downloaded.

`custom_nodes/ComfyColab-LTXVideo/catalog/minimax_h3.json` records the
immutable `Comfy-Org/MiniMax-H3@0543966fbdce5ba05709a8f2031c94bdba629b4a`
optimized Base assets. First use of either FL2VA or Ref2VA downloads
`42,470,585,471` bytes. The shared Qwen3-VL text encoder and video/audio VAEs
are `21,500,205,855` bytes, so installing the other H3 variant later downloads
only the second `20,970,379,616` byte transformer. Keeping both variants uses
`63,440,965,087` bytes.

The H3 loader requires explicit acknowledgement that the user reviewed the
MiniMax H3 Community License before any model folder creation, download, or
model loading. It does not perform country, IP, geolocation, or
regional-availability checks.

The `ComfyColab MiniMax H3 - Prompt Enhancer` node sits before the prompt input
in every included H3 workflow. It uses thinking-enabled `Qwen/Qwen3.8-27B` with
the pinned 17.1 GB `Q4_K_M` GGUF from Unsloth, an official MiniMax H3 prompting
policy pinned to MiniMax's H3 prompt-writing guide, and an isolated llama.cpp
server that exits before H3 sampling.

## MiniMax H3 reference limits

Ref2VA supports up to 9 reference images, 3 reference videos, 3 paired
reference-video soundtracks, and 3 standalone audio clips. At least one image
or video reference is required; audio alone is rejected. Each video/audio clip
must be 2-15 seconds, total reference-video duration is capped at 15 seconds,
total reference-audio duration is capped at 15 seconds, and the combined count
of image, video, paired-audio, and standalone-audio reference files is capped
at 12.

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
