from __future__ import annotations

import importlib
from typing import Any

from .catalog import gguf_names, spatial_upscaler_names
from .catalog_h3 import FL2VA, REF2VA, h3_variant_labels, normalize_h3_variant
from .graph import build_ltx23_graph, required_nodes
from .graph_h3 import (
    FL2VA_REQUIRED_NODES,
    REF2VA_REQUIRED_NODES,
    build_h3_reference_graph,
    build_h3_video_graph,
)
from .models import ensure_model_assets
from .models_h3 import ensure_h3_model_assets


MAX_SEED = (2**63) - 1
FPS_OPTIONS = ["24", "48"]
H3_SCHEDULERS = ["beta", "normal", "simple"]
H3_REF_IMAGE_SIZES = ["match", "max"]


def _collect_autogrow(values) -> dict[str, Any]:
    if values is None:
        return {}
    if isinstance(values, dict):
        return {str(key): value for key, value in values.items() if value is not None}
    if isinstance(values, (list, tuple)):
        return {
            str(index): value
            for index, value in enumerate(values)
            if value is not None
        }
    raise ValueError("MiniMax H3 reference inputs must be expandable input groups.")


def _io():
    return importlib.import_module("comfy_api.latest").io


def _require_upstream_nodes(node_ids: set[str]) -> None:
    try:
        registry = importlib.import_module("nodes").NODE_CLASS_MAPPINGS
    except (ModuleNotFoundError, AttributeError):
        return
    missing = sorted(node_ids - set(registry))
    if missing:
        raise RuntimeError(
            "ComfyColab LTX-2.3 requires current ComfyUI plus ComfyUI-GGUF and "
            "Lighttricks ComfyUI-LTXVideo. Missing node IDs: "
            f"{', '.join(missing)}. Run this repository's `install.py`, update "
            "ComfyUI, and restart. Managed ComfyColab users can instead run "
            "`comfycolab start --refresh`."
        )


def _loader(comfy_nodes: Any, name: str) -> Any:
    loader_class = comfy_nodes.NODE_CLASS_MAPPINGS.get(name)
    if loader_class is None:
        raise RuntimeError(
            f"Required loader '{name}' is unavailable. Restart with "
            "`comfycolab start --refresh`."
        )
    return loader_class()


def _video_output(io: Any):
    return io.Video.Output("video")


def _h3_bundle_output(io: Any):
    return io.Custom("MINIMAX_H3_BUNDLE").Output("bundle")


def _h3_bundle_input(io: Any):
    return io.Custom("MINIMAX_H3_BUNDLE").Input(
        "bundle",
        tooltip="Connect the one-cable output from MiniMax H3 Bundle Loader.",
    )


class ComfyColabLTX23Video:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabLTX23Video",
            display_name="ComfyColab LTX-2.3 — Text/Image to Video",
            category="ComfyColab/Video",
            description=(
                "Generates synchronized video and audio with the latest LTX-2.3 "
                "Distilled 1.1 GGUF. Width and height are the base generation size "
                "before the optional spatial upscaler. 48 FPS uses the official "
                "temporal x2 latent upscaler after 24 FPS generation."
            ),
            enable_expand=True,
            inputs=[
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describe the shot, motion, dialogue, and ambient audio.",
                ),
                io.Image.Input(
                    "image",
                    optional=True,
                    tooltip="Optional first-frame reference. Leave disconnected for text-to-video.",
                ),
                io.Combo.Input(
                    "gguf_model",
                    options=gguf_names(),
                    default="Q3_K_S",
                    tooltip="Q3_K_S is the safest low-memory default; Q4 variants improve fidelity.",
                ),
                io.Combo.Input(
                    "fps",
                    options=FPS_OPTIONS,
                    default="24",
                    tooltip="48 applies real temporal x2 latent upscaling.",
                ),
                io.Combo.Input(
                    "spatial_upscaler",
                    options=spatial_upscaler_names(),
                    default="2x",
                    tooltip=(
                        "2x uses the latest v1.1 hotfix; 1.5x uses the latest "
                        "available v1.0 and requires 64-divisible base dimensions."
                    ),
                ),
                io.Int.Input("width", default=960, min=256, max=1536, step=32),
                io.Int.Input("height", default=544, min=256, max=1536, step=32),
                io.Int.Input(
                    "frame_count",
                    default=121,
                    min=9,
                    max=257,
                    step=8,
                    tooltip="Must satisfy (frame_count - 1) % 8 == 0.",
                ),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Float.Input(
                    "image_strength",
                    default=0.7,
                    min=0.0,
                    max=1.0,
                    step=0.05,
                    advanced=True,
                ),
                io.Boolean.Input(
                    "force_redownload",
                    default=False,
                    advanced=True,
                    tooltip="Discard resumable cached files and download selected assets again.",
                ),
            ],
            outputs=[
                _video_output(io),
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        prompt,
        image=None,
        gguf_model="Q3_K_S",
        fps="24",
        spatial_upscaler="2x",
        width=960,
        height=544,
        frame_count=121,
        seed=0,
        image_strength=0.7,
        force_redownload=False,
    ):
        fps_value = int(fps)
        width = int(width)
        height = int(height)
        frame_count = int(frame_count)
        seed = int(seed)
        image_strength = float(image_strength)
        if not str(prompt).strip():
            raise ValueError("LTX-2.3 requires a non-empty prompt.")
        if fps_value not in {24, 48}:
            raise ValueError("LTX-2.3 FPS must be 24 or 48.")
        if width % 32 or height % 32:
            raise ValueError("LTX-2.3 width and height must be divisible by 32.")
        if spatial_upscaler == "1.5x" and (width % 64 or height % 64):
            raise ValueError(
                "LTX-2.3 1.5x spatial upscaling requires width and height "
                "to be divisible by 64."
            )
        if (frame_count - 1) % 8:
            raise ValueError("LTX-2.3 frame_count must satisfy (frame_count - 1) % 8 == 0.")
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}.")
        if not 0.0 <= image_strength <= 1.0:
            raise ValueError("image_strength must be between 0 and 1.")
        _require_upstream_nodes(
            required_nodes(
                image_to_video=image is not None,
                spatial=spatial_upscaler != "None",
                temporal=fps_value == 48,
            )
        )
        model_names = ensure_model_assets(
            gguf_model,
            spatial_upscaler,
            fps_value,
            force_redownload=bool(force_redownload),
        )
        return build_ltx23_graph(
            prompt=str(prompt),
            image=image,
            fps=fps_value,
            spatial_upscaler=spatial_upscaler,
            width=width,
            height=height,
            frame_count=frame_count,
            seed=seed,
            image_strength=image_strength,
            model_names=model_names,
        )


class ComfyColabMiniMaxH3BundleLoader:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabMiniMaxH3BundleLoader",
            display_name="MiniMax H3 Bundle Loader",
            category="ComfyColab/loaders",
            description=(
                "Downloads, verifies, and loads the optimized MiniMax H3 Base "
                "FL2VA or Ref2VA transformer plus the shared Qwen3-VL text encoder "
                "and separate video/audio VAEs. The acknowledgement is required "
                "before any filesystem or network side effect."
            ),
            inputs=[
                io.Combo.Input(
                    "model_variant",
                    options=h3_variant_labels(),
                    default=h3_variant_labels()[0],
                    tooltip="FL2VA is for text, first-frame, last-frame, or both-frame generation. Ref2VA is for reference media.",
                ),
                io.Boolean.Input(
                    "accept_h3_license_and_territory",
                    default=False,
                    tooltip=(
                        "Required acknowledgement that you reviewed the MiniMax H3 "
                        "Community License and are authorized to use the weights in "
                        "your location."
                    ),
                ),
                io.Boolean.Input(
                    "force_redownload",
                    default=False,
                    advanced=True,
                    tooltip="Discard resumable cached H3 files and download selected assets again.",
                ),
            ],
            outputs=[
                _h3_bundle_output(io),
                io.Model.Output("model"),
                io.Clip.Output("text_encoder"),
                io.Vae.Output("video_vae"),
                io.Vae.Output("audio_vae"),
            ],
        )

    @classmethod
    def execute(
        cls,
        model_variant=h3_variant_labels()[0],
        accept_h3_license_and_territory=False,
        force_redownload=False,
    ):
        if not bool(accept_h3_license_and_territory):
            raise PermissionError(
                "MiniMax H3 download is blocked until you acknowledge the H3 "
                "Community License and territory restrictions."
            )
        variant = normalize_h3_variant(model_variant)
        _require_upstream_nodes(FL2VA_REQUIRED_NODES | REF2VA_REQUIRED_NODES)
        model_names = ensure_h3_model_assets(
            variant,
            force_redownload=bool(force_redownload),
        )
        comfy_nodes = importlib.import_module("nodes")
        model = _loader(comfy_nodes, "UNETLoader").load_unet(
            model_names["model"],
            weight_dtype="default",
        )[0]
        text_encoder = _loader(comfy_nodes, "CLIPLoader").load_clip(
            model_names["text_encoder"],
            type="minimax",
        )[0]
        video_vae = _loader(comfy_nodes, "VAELoader").load_vae(
            model_names["video_vae"]
        )[0]
        audio_vae = _loader(comfy_nodes, "VAELoader").load_vae(
            model_names["audio_vae"]
        )[0]
        bundle = {
            "family": "minimax_h3",
            "variant": variant,
            "model": model,
            "text_encoder": text_encoder,
            "video_vae": video_vae,
            "audio_vae": audio_vae,
            "filenames": dict(model_names),
        }
        return bundle, model, text_encoder, video_vae, audio_vae


class ComfyColabMiniMaxH3Video:
    @classmethod
    def define_schema(cls):
        io = _io()
        return io.Schema(
            node_id="ComfyColabMiniMaxH3Video",
            display_name="ComfyColab MiniMax H3 - Text/Image to Video",
            category="ComfyColab/Video",
            description=(
                "Generates MiniMax H3 Base 24 FPS video with native 32 kHz stereo "
                "audio from a prompt plus optional first and last frames. Requires "
                "an FL2VA bundle from the MiniMax H3 Bundle Loader."
            ),
            enable_expand=True,
            inputs=[
                _h3_bundle_input(io),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Describe the shot, camera motion, dialogue, sound effects, and music.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=4.0,
                    max=15.0,
                    step=0.25,
                    tooltip="H3 runs at 24 FPS and snaps upward to the valid 17k + 5 frame grid.",
                ),
                io.Int.Input("width", default=864, min=256, max=1344, step=32),
                io.Int.Input("height", default=480, min=256, max=768, step=32),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Image.Input(
                    "first_frame",
                    optional=True,
                    tooltip="Optional starting image. Leave disconnected for text-only generation.",
                ),
                io.Image.Input(
                    "last_frame",
                    optional=True,
                    tooltip="Optional ending image. May be used alone or with first_frame.",
                ),
            ],
            outputs=[
                _video_output(io),
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        bundle,
        prompt,
        duration_seconds=5.0,
        width=864,
        height=480,
        seed=0,
        first_frame=None,
        last_frame=None,
    ):
        seed = int(seed)
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}.")
        return build_h3_video_graph(
            bundle=bundle,
            prompt=str(prompt),
            duration_seconds=float(duration_seconds),
            width=int(width),
            height=int(height),
            seed=seed,
            first_frame=first_frame,
            last_frame=last_frame,
        )


class ComfyColabMiniMaxH3ReferenceVideo:
    @classmethod
    def define_schema(cls):
        io = _io()
        ref_images = io.Autogrow.Input(
            "ref_images",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("ref_image"),
                prefix="ref_image_",
                min=0,
                max=9,
            ),
        )
        ref_videos = io.Autogrow.Input(
            "ref_videos",
            template=io.Autogrow.TemplatePrefix(
                input=io.Image.Input("ref_video"),
                prefix="ref_video_",
                min=0,
                max=3,
            ),
        )
        ref_video_audios = io.Autogrow.Input(
            "ref_video_audios",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("ref_video_audio"),
                prefix="ref_video_audio_",
                min=0,
                max=3,
            ),
        )
        ref_audios = io.Autogrow.Input(
            "ref_audios",
            template=io.Autogrow.TemplatePrefix(
                input=io.Audio.Input("ref_audio"),
                prefix="ref_audio_",
                min=0,
                max=3,
            ),
        )
        return io.Schema(
            node_id="ComfyColabMiniMaxH3ReferenceVideo",
            display_name="ComfyColab MiniMax H3 - Reference to Video",
            category="ComfyColab/Video",
            description=(
                "Generates MiniMax H3 Ref2VA video and native stereo audio from "
                "ordered reference images, videos, paired video soundtracks, and "
                "standalone audio. Use prompt tags such as <Picture 1>, <Video 1>, "
                "and <Audio 1>."
            ),
            enable_expand=True,
            inputs=[
                _h3_bundle_input(io),
                io.String.Input(
                    "prompt",
                    multiline=True,
                    default="",
                    tooltip="Use one-based tags such as <Picture 1>, <Video 1>, and <Audio 1>.",
                ),
                io.Float.Input(
                    "duration_seconds",
                    default=5.0,
                    min=4.0,
                    max=15.0,
                    step=0.25,
                ),
                io.Int.Input("width", default=864, min=256, max=1344, step=32),
                io.Int.Input("height", default=480, min=256, max=768, step=32),
                io.Int.Input("seed", default=0, min=0, max=MAX_SEED),
                io.Combo.Input(
                    "ref_image_size",
                    options=H3_REF_IMAGE_SIZES,
                    default="match",
                    advanced=True,
                    tooltip="match is the low-memory default; max can improve identity fidelity at higher cost.",
                ),
                io.Combo.Input(
                    "scheduler",
                    options=H3_SCHEDULERS,
                    default="beta",
                    advanced=True,
                    tooltip="All choices use res_multistep and 20 steps; beta is the reference-heavy default.",
                ),
                ref_images,
                ref_videos,
                ref_video_audios,
                ref_audios,
            ],
            outputs=[
                _video_output(io),
                io.Image.Output("frames"),
                io.Audio.Output("audio"),
            ],
        )

    @classmethod
    def execute(
        cls,
        bundle,
        prompt,
        duration_seconds=5.0,
        width=864,
        height=480,
        seed=0,
        ref_image_size="match",
        scheduler="beta",
        ref_images=None,
        ref_videos=None,
        ref_video_audios=None,
        ref_audios=None,
    ):
        seed = int(seed)
        if seed < 0 or seed > MAX_SEED:
            raise ValueError(f"seed must be between 0 and {MAX_SEED}.")
        ref_images = _collect_autogrow(ref_images)
        ref_videos = _collect_autogrow(ref_videos)
        ref_video_audios = _collect_autogrow(ref_video_audios)
        ref_audios = _collect_autogrow(ref_audios)
        return build_h3_reference_graph(
            bundle=bundle,
            prompt=str(prompt),
            duration_seconds=float(duration_seconds),
            width=int(width),
            height=int(height),
            seed=seed,
            ref_image_size=str(ref_image_size),
            scheduler=str(scheduler),
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios=ref_audios,
        )


PUBLIC_NODE_CLASS_MAPPINGS = {
    "ComfyColabLTX23Video": ComfyColabLTX23Video,
    "ComfyColabMiniMaxH3BundleLoader": ComfyColabMiniMaxH3BundleLoader,
    "ComfyColabMiniMaxH3Video": ComfyColabMiniMaxH3Video,
    "ComfyColabMiniMaxH3ReferenceVideo": ComfyColabMiniMaxH3ReferenceVideo,
}

NODE_CLASS_MAPPINGS = dict(PUBLIC_NODE_CLASS_MAPPINGS)

NODE_DISPLAY_NAME_MAPPINGS = {
    "ComfyColabLTX23Video": "ComfyColab LTX-2.3 — Text/Image to Video",
    "ComfyColabMiniMaxH3BundleLoader": "MiniMax H3 Bundle Loader",
    "ComfyColabMiniMaxH3Video": "ComfyColab MiniMax H3 - Text/Image to Video",
    "ComfyColabMiniMaxH3ReferenceVideo": "ComfyColab MiniMax H3 - Reference to Video",
}
