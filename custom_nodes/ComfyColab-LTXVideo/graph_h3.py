from __future__ import annotations

import importlib
from typing import Any

from .catalog_h3 import FL2VA, REF2VA


FPS = 24
STEPS = 20
AREA_CAP = 768 * 1344
BASE_REQUIRED_NODES = frozenset(
    {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "RandomNoise",
        "BasicGuider",
        "KSamplerSelect",
        "BasicScheduler",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
    }
)
FL2VA_REQUIRED_NODES = BASE_REQUIRED_NODES | {"MiniMaxH3ImageToVideo"}
REF2VA_REQUIRED_NODES = BASE_REQUIRED_NODES | {"MiniMaxH3ReferenceToVideo"}


def _builder():
    return importlib.import_module("comfy_execution.graph_utils").GraphBuilder()


def _finish(graph, video, images, audio):
    io = importlib.import_module("comfy_api.latest").io
    return io.NodeOutput(video, images, audio, expand=graph.finalize())


def snap_h3_frame_count(duration_seconds: float) -> int:
    requested = max(1, round(float(duration_seconds) * FPS))
    return requested + ((5 - requested) % 17)


def validate_h3_prompt_and_canvas(prompt: str, width: int, height: int) -> None:
    if not str(prompt).strip():
        raise ValueError("MiniMax H3 requires a non-empty prompt.")
    if width % 32 or height % 32:
        raise ValueError("MiniMax H3 width and height must be divisible by 32.")
    if width * height > AREA_CAP:
        raise ValueError(
            "MiniMax H3 Base local output is capped at 768 x 1344 pixels of area."
        )


def validate_h3_duration(duration_seconds: float) -> None:
    if not 4.0 <= float(duration_seconds) <= 15.0:
        raise ValueError("MiniMax H3 duration_seconds must be between 4 and 15.")


def _component(bundle: dict[str, Any], name: str):
    try:
        return bundle[name]
    except KeyError as error:
        raise ValueError("MiniMax H3 bundle is missing loaded components.") from error


def validate_h3_bundle(bundle: dict[str, Any], expected_variant: str) -> None:
    if not isinstance(bundle, dict) or bundle.get("family") != "minimax_h3":
        raise ValueError("Connect a bundle from MiniMax H3 Bundle Loader.")
    variant = bundle.get("variant")
    if variant != expected_variant:
        if expected_variant == FL2VA:
            raise ValueError("Use the Text/Image Video node with an FL2VA bundle.")
        raise ValueError("Use the Reference Video node with a Ref2VA bundle.")


def build_h3_video_graph(
    *,
    bundle: dict[str, Any],
    prompt: str,
    duration_seconds: float,
    width: int,
    height: int,
    seed: int,
    first_frame: Any | None = None,
    last_frame: Any | None = None,
):
    validate_h3_bundle(bundle, FL2VA)
    validate_h3_duration(duration_seconds)
    validate_h3_prompt_and_canvas(prompt, width, height)
    length = snap_h3_frame_count(duration_seconds)
    graph = _builder()
    conditioning = graph.node(
        "MiniMaxH3ImageToVideo",
        clip=_component(bundle, "text_encoder"),
        vae=_component(bundle, "video_vae"),
        first_frame=first_frame,
        last_frame=last_frame,
        prompt=str(prompt),
        width=width,
        height=height,
        length=length,
    )
    return _build_sampling_tail(
        graph=graph,
        bundle=bundle,
        conditioning=conditioning.out(0),
        latent=conditioning.out(1),
        seed=seed,
        scheduler="simple",
    )


def validate_h3_references(
    *,
    ref_images: dict[str, Any],
    ref_videos: dict[str, Any],
    ref_video_audios: dict[str, Any],
    ref_audios: dict[str, Any],
) -> None:
    image_count = len(ref_images)
    video_count = len(ref_videos)
    standalone_audio_count = len(ref_audios)
    paired_audio_count = len(ref_video_audios)
    if image_count == 0 and video_count == 0:
        raise ValueError("MiniMax H3 Ref2VA requires at least one image or video reference.")
    if image_count > 9 or video_count > 3 or standalone_audio_count > 3:
        raise ValueError("MiniMax H3 Ref2VA reference counts exceed native limits.")
    for key in ref_video_audios:
        index = _reference_index(key)
        if _matching_reference_key(ref_videos, index) is None:
            raise ValueError("MiniMax H3 paired video audio must match a reference video.")
    if image_count + video_count + standalone_audio_count + paired_audio_count > 12:
        raise ValueError("MiniMax H3 Ref2VA accepts at most 12 reference files total.")
    video_durations = [_video_duration_seconds(value) for value in ref_videos.values()]
    audio_durations = [
        _audio_duration_seconds(value)
        for value in [*ref_video_audios.values(), *ref_audios.values()]
    ]
    for duration in video_durations:
        if not 2.0 <= duration <= 15.0:
            raise ValueError("MiniMax H3 reference videos must be 2 to 15 seconds long.")
    for duration in audio_durations:
        if not 2.0 <= duration <= 15.0:
            raise ValueError("MiniMax H3 reference audio clips must be 2 to 15 seconds long.")
    if sum(video_durations) > 15.0:
        raise ValueError("MiniMax H3 total reference-video duration must be at most 15 seconds.")
    if sum(audio_durations) > 15.0:
        raise ValueError("MiniMax H3 total reference-audio duration must be at most 15 seconds.")


def _video_duration_seconds(value: Any) -> float:
    if isinstance(value, dict):
        if "frames" in value:
            return _frame_count(value["frames"]) / FPS
        raise ValueError("MiniMax H3 reference video duration is unavailable.")
    return _frame_count(value) / FPS


def _frame_count(frames: Any) -> int:
    if hasattr(frames, "shape") and frames.shape:
        return int(frames.shape[0])
    try:
        return len(frames)
    except TypeError as error:
        raise ValueError("MiniMax H3 reference video duration is unavailable.") from error


def _audio_duration_seconds(value: Any) -> float:
    if not isinstance(value, dict):
        raise ValueError("MiniMax H3 reference audio must include waveform and sample_rate.")
    waveform = value.get("waveform")
    sample_rate = int(value.get("sample_rate") or 0)
    if waveform is None or sample_rate <= 0:
        raise ValueError("MiniMax H3 reference audio must include waveform and sample_rate.")
    return _sample_count(waveform) / sample_rate


def _sample_count(waveform: Any) -> int:
    if hasattr(waveform, "shape") and waveform.shape:
        return int(waveform.shape[-1])
    try:
        return len(waveform)
    except TypeError as error:
        raise ValueError("MiniMax H3 reference audio duration is unavailable.") from error


def _reference_index(key: str) -> str:
    digits = "".join(character for character in str(key) if character.isdigit())
    return digits or str(key)


def _matching_reference_key(values: dict[str, Any], index: str) -> str | None:
    for key in values:
        if _reference_index(key) == index:
            return key
    return None


def build_h3_reference_graph(
    *,
    bundle: dict[str, Any],
    prompt: str,
    duration_seconds: float,
    width: int,
    height: int,
    seed: int,
    ref_image_size: str,
    scheduler: str,
    ref_images: dict[str, Any],
    ref_videos: dict[str, Any],
    ref_video_audios: dict[str, Any],
    ref_audios: dict[str, Any],
):
    validate_h3_bundle(bundle, REF2VA)
    validate_h3_duration(duration_seconds)
    validate_h3_prompt_and_canvas(prompt, width, height)
    if ref_image_size not in {"match", "max"}:
        raise ValueError("MiniMax H3 ref_image_size must be match or max.")
    if scheduler not in {"beta", "normal", "simple"}:
        raise ValueError("MiniMax H3 scheduler must be beta, normal, or simple.")
    validate_h3_references(
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_video_audios=ref_video_audios,
        ref_audios=ref_audios,
    )
    length = snap_h3_frame_count(duration_seconds)
    graph = _builder()
    inputs = {
        "clip": _component(bundle, "text_encoder"),
        "vae": _component(bundle, "video_vae"),
        "audio_vae": _component(bundle, "audio_vae"),
        "prompt": str(prompt),
        "width": width,
        "height": height,
        "length": length,
        "ref_image_size": ref_image_size,
        "ref_images": ref_images,
        "ref_videos": ref_videos,
        "ref_video_audios": ref_video_audios,
        "ref_audios": ref_audios,
    }
    conditioning = graph.node("MiniMaxH3ReferenceToVideo", **inputs)
    return _build_sampling_tail(
        graph=graph,
        bundle=bundle,
        conditioning=conditioning.out(0),
        latent=conditioning.out(1),
        seed=seed,
        scheduler=scheduler,
    )


def _build_sampling_tail(
    *,
    graph,
    bundle: dict[str, Any],
    conditioning,
    latent,
    seed: int,
    scheduler: str,
):
    noise = graph.node("RandomNoise", noise_seed=seed)
    guider = graph.node(
        "BasicGuider",
        model=_component(bundle, "model"),
        conditioning=conditioning,
    )
    sampler = graph.node("KSamplerSelect", sampler_name="res_multistep")
    sigmas = graph.node(
        "BasicScheduler",
        model=_component(bundle, "model"),
        scheduler=scheduler,
        steps=STEPS,
        denoise=1.0,
    )
    sampled = graph.node(
        "SamplerCustomAdvanced",
        noise=noise.out(0),
        guider=guider.out(0),
        sampler=sampler.out(0),
        sigmas=sigmas.out(0),
        latent_image=latent,
    )
    frames = graph.node(
        "VAEDecode",
        samples=sampled.out(0),
        vae=_component(bundle, "video_vae"),
    )
    audio = graph.node(
        "VAEDecodeAudio",
        samples=sampled.out(0),
        vae=_component(bundle, "audio_vae"),
    )
    video = graph.node(
        "CreateVideo",
        images=frames.out(0),
        audio=audio.out(0),
        fps=float(FPS),
    )
    return _finish(graph, video.out(0), frames.out(0), audio.out(0))
