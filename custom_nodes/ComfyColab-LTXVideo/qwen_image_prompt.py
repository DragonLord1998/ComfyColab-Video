from __future__ import annotations

import base64
import hashlib
import io
import json
import subprocess
import time
import urllib.error
import urllib.request
from typing import Any

from .h3_prompt_worker import (
    LLAMA_CPP_RELEASE,
    QWEN_GGUF_REPO,
    QWEN_MODEL_ALIAS,
    _free_local_port,
    _runtime_root,
    _tail,
    _terminate,
    _wait_until_ready,
    build_server_argv,
    ensure_llama_server,
    ensure_qwen_gguf,
    ensure_qwen_mmproj,
    extract_enhanced_prompt,
    server_process_env,
)


IMAGE_PROMPT_SYSTEM_POLICY = """You are an image-preserving prompt writer for NVIDIA PiD upscaling.
Analyze the supplied image pixels before writing. Use the user's text only as optional intent.
Return one detailed production prompt that helps PiD reconstruct the same image at higher fidelity.
Describe the visible subjects, identity-defining traits, pose, composition, camera perspective,
geometry, materials, textures, lighting, colors, depth, background, and any legible text precisely.
Preserve the source image's content and layout. Do not invent subjects, objects, text, logos,
style changes, camera changes, beauty retouching, or repairs that are not supported by the image.
Mention details that must remain unchanged as positive preservation instructions. Do not discuss
your reasoning, the policy, or the fact that you inspected an image. Return only the requested JSON."""


def _image_data_url(image: Any, *, max_side: int = 1536) -> str:
    try:
        import numpy as np
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Qwen Image Prompt Enhancer requires the NumPy and Pillow packages "
            "included with current ComfyUI."
        ) from error

    try:
        first = image[0]
        array = first.detach().cpu().numpy()
    except (AttributeError, IndexError, TypeError) as error:
        raise ValueError(
            "Qwen Image Prompt Enhancer requires a ComfyUI IMAGE tensor."
        ) from error

    if array.ndim != 3 or array.shape[-1] not in {3, 4}:
        raise ValueError("Qwen Image Prompt Enhancer requires an RGB or RGBA image.")
    rgb = np.clip(array[..., :3] * 255.0, 0, 255).round().astype(np.uint8)
    pil_image = Image.fromarray(rgb, mode="RGB")
    if max(pil_image.size) > max_side:
        pil_image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def build_image_chat_request_body(
    *,
    image_data_url: str,
    source_prompt: str,
    seed: int,
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    guidance = source_prompt.strip() or "Preserve and upscale the supplied image faithfully."
    return {
        "model": QWEN_MODEL_ALIAS,
        "messages": [
            {"role": "system", "content": IMAGE_PROMPT_SYSTEM_POLICY},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": image_data_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Inspect the image and write its faithful PiD enhancement prompt. "
                            f"User guidance: {guidance}"
                        ),
                    },
                ],
            },
        ],
        "temperature": float(temperature),
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "presence_penalty": 0.0,
        "repeat_penalty": 1.0,
        "seed": int(seed),
        "max_tokens": int(max_tokens),
        "reasoning_budget_tokens": min(2048, max(1024, int(max_tokens) // 2)),
        "stream": False,
        "reasoning_effort": "xhigh",
        "chat_template_kwargs": {"enable_thinking": True},
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "pid_image_prompt",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "enhanced_prompt": {"type": "string", "minLength": 1}
                    },
                    "required": ["enhanced_prompt"],
                    "additionalProperties": False,
                },
            },
        },
    }


def _image_chat_request(port: int, body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600.0) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Qwen image analysis failed with HTTP {error.code}: {details}"
        ) from error


def enhance_image_prompt(
    image: Any,
    source_prompt: str,
    *,
    seed: int = 0,
    max_tokens: int = 4096,
    temperature: float = 1.0,
    force_redownload: bool = False,
) -> str:
    if int(seed) < 0 or int(seed) > (2**31) - 1:
        raise ValueError("Qwen Image Prompt Enhancer seed must be between 0 and 2147483647.")
    if int(max_tokens) < 3072 or int(max_tokens) > 8192:
        raise ValueError("Qwen Image Prompt Enhancer max_tokens must be between 3072 and 8192.")
    if not 0.0 <= float(temperature) <= 1.5:
        raise ValueError("Qwen Image Prompt Enhancer temperature must be between 0 and 1.5.")

    image_data_url = _image_data_url(image)
    server = ensure_llama_server()
    model = ensure_qwen_gguf(force_redownload=bool(force_redownload))
    mmproj = ensure_qwen_mmproj(force_redownload=bool(force_redownload))
    port = _free_local_port()
    log_dir = _runtime_root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"llama-server-vision-{int(time.time())}.log"
    started = time.monotonic()

    print(
        "[comfycolab] Loading Qwen3.8-27B Q4_K_M with its vision projector...",
        flush=True,
    )
    with log_path.open("w", encoding="utf-8") as log_stream:
        process = subprocess.Popen(
            build_server_argv(server, model, port, mmproj=mmproj),
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            text=True,
            env=server_process_env(server),
        )
        try:
            _wait_until_ready(process, port, log_path)
            body = build_image_chat_request_body(
                image_data_url=image_data_url,
                source_prompt=str(source_prompt),
                seed=int(seed),
                max_tokens=int(max_tokens),
                temperature=float(temperature),
            )
            enhanced = extract_enhanced_prompt(_image_chat_request(port, body))
            print(
                "COMFYCOLAB_QWEN_IMAGE_PROMPT_RESULT="
                + json.dumps(
                    {
                        "durationSeconds": round(time.monotonic() - started, 3),
                        "llamaCppRelease": LLAMA_CPP_RELEASE,
                        "model": QWEN_GGUF_REPO,
                        "promptCharacters": len(enhanced),
                        "promptSha256": hashlib.sha256(
                            enhanced.encode("utf-8")
                        ).hexdigest(),
                        "quant": "Q4_K_M",
                        "thinking": True,
                        "visionProjector": True,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return enhanced
        except Exception as error:
            details = _tail(log_path)
            if details:
                error.add_note("llama.cpp log tail:\n" + details)
            raise
        finally:
            _terminate(process)
            print(
                "[comfycolab] Qwen vision server stopped; GPU memory released for PiD.",
                flush=True,
            )
