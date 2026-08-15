# Third-party notices

ComfyColab Video downloads and connects third-party projects at runtime. It
does not store model weights in this repository. Review the upstream terms
before redistribution, commercial use, or hosted-service deployment.

## LTX-2.3 video

- Official model: <https://huggingface.co/Lightricks/LTX-2.3>
- Pinned model revision: `4229404625088d21c4f112eb640fb04a0900ee25`
- Official direct-distilled inference reference: <https://github.com/Lightricks/LTX-2>
- Pinned inference-reference revision: `9377758131b1ffde4b7f766804590a6617bf2ab9`
- LTX-2 Community License: <https://github.com/Lightricks/LTX-2/blob/main/LICENSE>
- Official ComfyUI extension: <https://github.com/Lightricks/ComfyUI-LTXVideo>
- Pinned extension revision: `aceeae9635f6d493f2893ba3c411a1c36031788a`
- Community GGUF repository: <https://huggingface.co/unsloth/LTX-2.3-GGUF>
- Pinned GGUF revision: `96e8ed4925ead3db9ff4d0084f165ef6a74f28d0`
- GGUF loader: <https://github.com/city96/ComfyUI-GGUF>
- Pinned GGUF-loader revision: `6ea2651e7df66d7585f6ffee804b20e92fb38b8a`
- GGUF-loader license: Apache-2.0
- Gemma text encoder: <https://huggingface.co/unsloth/gemma-3-12b-it-qat-GGUF>
- Pinned Gemma revision: `858acec7ec0541a46c39985c95d3b52d8f3ab183`
- Gemma terms: <https://ai.google.dev/gemma/terms>

The LTX model, official ComfyUI extension, upscalers, and derivatives are
subject to the LTX-2 Community License Agreement and its acceptable-use and
commercial-use terms. The current agreement requires entities with at least
USD 10 million in annual revenue to obtain a paid commercial-use license. The
selected Gemma text encoder is separately subject to the Gemma terms, including
their distribution and hosted-service obligations. The GGUF files are
community conversions and derivatives rather than official Lighttricks
releases; they remain subject to the LTX-2 license. ComfyColab Video
checksum-pins them but does not independently certify their numeric fidelity.

Only the selected GGUF, shared encoder/connector/VAEs, and selected spatial or
temporal upscalers are downloaded. Their exact artifact revisions, sizes, and
SHA-256 values are recorded in `catalog/ltx_2_3.json`.

## MiniMax H3 Base

- Official model and license source: <https://huggingface.co/MiniMaxAI/MiniMax-H3>
- MiniMax H3 Community License: <https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/main/LICENSE>
- Optimized Comfy bundle: <https://huggingface.co/Comfy-Org/MiniMax-H3>
- Pinned optimized bundle revision: `0543966fbdce5ba05709a8f2031c94bdba629b4a`
- Native ComfyUI implementation PR: <https://github.com/Comfy-Org/ComfyUI/pull/15224>
- Minimum H3-capable ComfyUI merge commit: `57500fc5bc92566a63f2046824f522cd55c335ca`
- Qwen3-VL text encoder family: <https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct>
- Qwen3-VL license: Apache-2.0
- Prompt enhancer GGUF repository: <https://huggingface.co/unsloth/Qwen3.8-27B-GGUF>
- Pinned prompt enhancer GGUF revision: `f1bfb127c64f7072bdd2cad55f258b9c8b2910fe`
- Prompt enhancer GGUF file: `Qwen3.8-27B-Q4_K_M.gguf`
- llama.cpp server: <https://github.com/ggml-org/llama.cpp>
- Pinned llama.cpp revision: `16d222fc5ead59d20039501a37251c9ed457a454`
- llama.cpp license: MIT

MiniMax H3 weights are governed by the MiniMax H3 Community License Agreement,
not an OSI open-source license. This pack requires the user to acknowledge that
they reviewed the license before any H3 asset is downloaded or loaded. It does
not perform country, IP, geolocation, or regional-availability checks; users
remain responsible for verifying regional availability themselves.

The local open-weight Base release supports 24 FPS audio-video generation in
the 4-15 second range with a 768-pixel short-edge class canvas. This pack uses
the optimized FL2VA and Ref2VA Base checkpoints and does not claim local 2K
regeneration, hosted H3-Context-IR, sparse attention, fine-tuning support, or
model-training workflows.

Exact MiniMax H3 artifact revisions, byte sizes, and SHA-256 values are
recorded in `catalog/minimax_h3.json`. ComfyColab Video downloads weights on
first use; it does not redistribute them in this repository.

The prompt enhancer downloads the pinned Q4_K_M GGUF on first use and runs it
through a checksum-pinned llama.cpp server. On G4/SM120 runtimes it can restore
a precompiled cache from ComfyColab's GitHub Releases; otherwise it builds from
the pinned llama.cpp source revision.
