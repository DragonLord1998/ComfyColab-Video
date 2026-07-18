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
