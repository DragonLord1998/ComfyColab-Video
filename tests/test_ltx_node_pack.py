from __future__ import annotations

import asyncio
import contextlib
import importlib
import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = ROOT / "custom_nodes" / "ComfyColab-LTXVideo"

PUBLIC_NODE_ID = "ComfyColabLTX23Video"
DISPLAY_NAME = "ComfyColab LTX-2.3 — Text/Image to Video"
H3_LOADER_ID = "ComfyColabMiniMaxH3BundleLoader"
H3_PROMPT_ID = "ComfyColabMiniMaxH3PromptEnhancer"
H3_VIDEO_ID = "ComfyColabMiniMaxH3Video"
H3_REFERENCE_ID = "ComfyColabMiniMaxH3ReferenceVideo"
GGUF_OPTIONS = ["Q3_K_S", "Q4_K_S", "Q4_K_M"]
FPS_OPTIONS = ["24", "48"]
SPATIAL_OPTIONS = ["None", "1.5x", "2x"]

GGUF_FILENAMES = {
    "Q3_K_S": "ltx-2.3-22b-distilled-1.1-Q3_K_S.gguf",
    "Q4_K_S": "ltx-2.3-22b-distilled-1.1-Q4_K_S.gguf",
    "Q4_K_M": "ltx-2.3-22b-distilled-1.1-Q4_K_M.gguf",
}
SPATIAL_FILENAMES = {
    "1.5x": "ltx-2.3-spatial-upscaler-x1.5-1.0.safetensors",
    "2x": "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
}
TEMPORAL_FILENAME = "ltx-2.3-temporal-upscaler-x2-1.0.safetensors"


def load_package():
    name = "comfycolab_ltx_test"
    for module_name in list(sys.modules):
        if module_name == name or module_name.startswith(name + "."):
            del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        name,
        PACKAGE_DIR / "__init__.py",
        submodule_search_locations=[str(PACKAGE_DIR)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    assert spec.loader
    spec.loader.exec_module(package)
    return package


class PortFactory:
    def __init__(self, io_type=None):
        self.io_type = io_type

    def Input(self, name, **kwargs):
        return {
            "direction": "input",
            "name": name,
            "io_type": self.io_type,
            **kwargs,
        }

    def Output(self, name=None, **kwargs):
        return {
            "direction": "output",
            "name": name,
            "io_type": self.io_type,
            **kwargs,
        }


class FakeIO:
    class ComfyNode:
        pass

    String = PortFactory("STRING")
    Combo = PortFactory("COMBO")
    Int = PortFactory("INT")
    Float = PortFactory("FLOAT")
    Boolean = PortFactory("BOOLEAN")
    Image = PortFactory("IMAGE")
    Video = PortFactory("VIDEO")
    Audio = PortFactory("AUDIO")
    Latent = PortFactory("LATENT")
    Conditioning = PortFactory("CONDITIONING")
    Model = PortFactory("MODEL")
    Clip = PortFactory("CLIP")
    Vae = PortFactory("VAE")
    AnyType = PortFactory("ANY")

    class Hidden:
        unique_id = "UNIQUE_ID"
        prompt = "PROMPT"

    class Autogrow:
        Type = dict

        class TemplatePrefix:
            def __init__(self, input, prefix, min=0, max=100):
                self.input = input
                self.prefix = prefix
                self.min = min
                self.max = max

        @staticmethod
        def Input(name, template):
            return {
                "direction": "input",
                "name": name,
                "io_type": "COMFY_AUTOGROW_V3",
                "template": template,
            }

    @staticmethod
    def Custom(name):
        return PortFactory(name)

    @staticmethod
    def Schema(**kwargs):
        return types.SimpleNamespace(**kwargs)

    @staticmethod
    def NodeOutput(*values, **kwargs):
        return types.SimpleNamespace(values=values, **kwargs)


class Link:
    def __init__(self, node_id, index):
        self.node_id = node_id
        self.index = index

    def __eq__(self, other):
        return (
            isinstance(other, Link)
            and self.node_id == other.node_id
            and self.index == other.index
        )

    def __repr__(self):
        return f"Link({self.node_id!r}, {self.index!r})"


class GraphNode:
    def __init__(self, index, class_type, inputs):
        self.index = index
        self.class_type = class_type
        self.inputs = inputs
        self.override_display_id = None

    def out(self, index):
        return Link(self.index, index)

    def set_override_display_id(self, node_id):
        self.override_display_id = node_id


class GraphBuilder:
    last = None

    def __init__(self):
        self.nodes = []
        GraphBuilder.last = self

    def node(self, class_type, **inputs):
        node = GraphNode(len(self.nodes), class_type, inputs)
        self.nodes.append(node)
        return node

    def finalize(self):
        result = []
        for node in self.nodes:
            item = {"class_type": node.class_type, "inputs": node.inputs}
            if node.override_display_id is not None:
                item["override_display_id"] = node.override_display_id
            result.append(item)
        return result


REQUIRED_NATIVE_NODES = {
    "UnetLoaderGGUF",
    "DualCLIPLoaderGGUF",
    "VAELoader",
    "LTXVAudioVAELoader",
    "LTXAVTextEncoderLoader",
    "CLIPTextEncode",
    "LTXVConditioning",
    "EmptyLTXVLatentVideo",
    "LTXVEmptyLatentAudio",
    "LTXVConcatAVLatent",
    "LTXVSeparateAVLatent",
    "LTXVPreprocess",
    "LTXVImgToVideoConditionOnly",
    "RandomNoise",
    "BasicGuider",
    "KSamplerSelect",
    "ManualSigmas",
    "SamplerCustomAdvanced",
    "LatentUpscaleModelLoader",
    "LTXVLatentUpsampler",
    "LTXVTiledVAEDecode",
    "LTXVAudioVAEDecode",
    "CreateVideo",
    "UNETLoader",
    "CLIPLoader",
    "BasicScheduler",
    "VAEDecode",
    "VAEDecodeAudio",
    "MiniMaxH3ImageToVideo",
    "MiniMaxH3ReferenceToVideo",
}


class LTXNodePackTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in (
                "comfy_api",
                "comfy_api.latest",
                "comfy",
                "comfy.ldm",
                "comfy.ldm.modules",
                "comfy.ldm.modules.attention",
                "comfy.model_management",
                "comfy_execution",
                "comfy_execution.graph_utils",
                "folder_paths",
                "nodes",
            )
        }
        latest = types.ModuleType("comfy_api.latest")
        latest.io = FakeIO
        latest.ComfyExtension = type("ComfyExtension", (), {})
        api = types.ModuleType("comfy_api")
        api.latest = latest
        execution = types.ModuleType("comfy_execution")
        graph_utils = types.ModuleType("comfy_execution.graph_utils")
        graph_utils.GraphBuilder = GraphBuilder
        comfy = types.ModuleType("comfy")
        comfy_ldm = types.ModuleType("comfy.ldm")
        comfy_ldm_modules = types.ModuleType("comfy.ldm.modules")
        attention = types.ModuleType("comfy.ldm.modules.attention")
        attention.get_attention_function = lambda name, _default=None: (
            (lambda *args, **kwargs: ("sage", args, kwargs)) if name == "sage" else None
        )
        model_management = types.ModuleType("comfy.model_management")
        model_management.unload_all_models = lambda: None
        model_management.soft_empty_cache = lambda: None
        comfy_nodes = types.ModuleType("nodes")
        comfy_nodes.NODE_CLASS_MAPPINGS = {
            node_id: object for node_id in REQUIRED_NATIVE_NODES
        }
        folder_paths = types.ModuleType("folder_paths")
        folder_paths.get_filename_list = lambda _key: []
        folder_paths.get_folder_paths = lambda key: [f"/tmp/comfy-models/{key}"]
        folder_paths.get_full_path = (
            lambda key, filename: f"/tmp/comfy-models/{key}/{filename}"
        )
        sys.modules.update(
            {
                "comfy_api": api,
                "comfy_api.latest": latest,
                "comfy": comfy,
                "comfy.ldm": comfy_ldm,
                "comfy.ldm.modules": comfy_ldm_modules,
                "comfy.ldm.modules.attention": attention,
                "comfy.model_management": model_management,
                "comfy_execution": execution,
                "comfy_execution.graph_utils": graph_utils,
                "folder_paths": folder_paths,
                "nodes": comfy_nodes,
            }
        )

    def tearDown(self):
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def _modules(self):
        package = load_package()
        nodes = importlib.import_module("comfycolab_ltx_test.nodes")
        graph = importlib.import_module("comfycolab_ltx_test.graph")
        models = importlib.import_module("comfycolab_ltx_test.models")
        return package, nodes, graph, models

    @staticmethod
    def _filenames(
        gguf_model: str,
        spatial_upscaler: str,
        fps: str | int,
    ) -> dict[str, str | None]:
        fps = str(fps)
        return {
            "model": GGUF_FILENAMES[gguf_model],
            "gguf_model": GGUF_FILENAMES[gguf_model],
            "connector": (
                "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
            ),
            "embedding_connector": (
                "ltx-2.3-22b-distilled_embeddings_connectors.safetensors"
            ),
            "text_encoder": "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
            "gemma": "gemma-3-12b-it-qat-UD-Q4_K_XL.gguf",
            "video_vae": "ltx-2.3-22b-distilled_video_vae.safetensors",
            "audio_vae": "ltx-2.3-22b-distilled_audio_vae.safetensors",
            "spatial_upscaler": SPATIAL_FILENAMES.get(spatial_upscaler),
            "temporal_2x": TEMPORAL_FILENAME if fps == "48" else None,
            "temporal_upscaler": TEMPORAL_FILENAME if fps == "48" else None,
        }

    @contextlib.contextmanager
    def _mock_model_downloads(self, modules):
        def ensure(
            gguf_model,
            spatial_upscaler,
            fps,
            force_redownload=False,
        ):
            return self._filenames(gguf_model, spatial_upscaler, fps)

        with contextlib.ExitStack() as stack:
            patched = False
            for module in modules:
                for name in (
                    "ensure_model_assets",
                    "ensure_ltx23_model_assets",
                    "ensure_runtime_assets",
                ):
                    if hasattr(module, name):
                        stack.enter_context(
                            mock.patch.object(module, name, side_effect=ensure)
                        )
                        patched = True
            self.assertTrue(
                patched,
                "The LTX pack must expose a mockable model-provisioning boundary.",
            )
            yield

    @staticmethod
    def _execute(
        node,
        *,
        gguf_model="Q3_K_S",
        fps="24",
        spatial_upscaler="2x",
        image=None,
        width=960,
        height=544,
    ):
        return node.execute(
            prompt="A quiet tea ceremony with synchronized room ambience.",
            gguf_model=gguf_model,
            fps=fps,
            spatial_upscaler=spatial_upscaler,
            width=width,
            height=height,
            frame_count=121,
            seed=0,
            image_strength=0.7,
            image=image,
        )

    def test_import_is_lazy_and_exposes_public_facades_with_exact_ltx_schema(self):
        before = set(sys.modules)
        package = load_package()
        imported = set(sys.modules) - before
        self.assertFalse(
            {"torch", "transformers", "diffusers", "numpy", "PIL"} & imported
        )

        extension = asyncio.run(package.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        schemas = [node.define_schema() for node in node_classes]
        public = [
            schema.node_id
            for schema in schemas
            if not getattr(schema, "is_dev_only", False)
        ]
        self.assertEqual(
            public,
            [PUBLIC_NODE_ID, H3_LOADER_ID, H3_PROMPT_ID, H3_VIDEO_ID, H3_REFERENCE_ID],
        )

        schema = next(item for item in schemas if item.node_id == PUBLIC_NODE_ID)
        inputs = {item["name"]: item for item in schema.inputs}
        self.assertEqual(schema.display_name, DISPLAY_NAME)
        self.assertEqual(schema.category, "ComfyColab/Video")
        self.assertTrue(schema.enable_expand)
        self.assertEqual(
            [item["name"] for item in schema.outputs],
            ["video", "frames", "audio"],
        )
        self.assertEqual(
            [item["io_type"] for item in schema.outputs],
            ["VIDEO", "IMAGE", "AUDIO"],
        )

        self.assertEqual(inputs["gguf_model"]["options"], GGUF_OPTIONS)
        self.assertEqual(inputs["gguf_model"]["default"], "Q3_K_S")
        self.assertEqual(inputs["fps"]["options"], FPS_OPTIONS)
        self.assertEqual(str(inputs["fps"]["default"]), "24")
        self.assertEqual(inputs["spatial_upscaler"]["options"], SPATIAL_OPTIONS)
        self.assertEqual(inputs["spatial_upscaler"]["default"], "2x")
        self.assertEqual(inputs["width"]["default"], 960)
        self.assertEqual(inputs["height"]["default"], 544)
        self.assertEqual(inputs["frame_count"]["default"], 121)
        self.assertEqual(inputs["seed"]["default"], 0)
        self.assertEqual(inputs["image_strength"]["default"], 0.7)
        self.assertTrue(inputs["image"]["optional"])
        self.assertIn("prompt", inputs)
        self.assertNotIn("negative_prompt", inputs)

        h3_schema = next(item for item in schemas if item.node_id == H3_VIDEO_ID)
        h3_inputs = {item["name"]: item for item in h3_schema.inputs}
        self.assertEqual(h3_schema.category, "ComfyColab/Video")
        self.assertTrue(h3_schema.enable_expand)
        self.assertEqual(
            [item["name"] for item in h3_schema.outputs],
            ["video", "frames", "audio"],
        )
        self.assertEqual(h3_inputs["bundle"]["io_type"], "MINIMAX_H3_BUNDLE")
        self.assertTrue(h3_inputs["first_frame"]["optional"])
        self.assertTrue(h3_inputs["last_frame"]["optional"])
        self.assertEqual(h3_inputs["duration_seconds"]["default"], 5.0)

        ref_schema = next(item for item in schemas if item.node_id == H3_REFERENCE_ID)
        ref_inputs = {item["name"]: item for item in ref_schema.inputs}
        self.assertEqual(ref_inputs["scheduler"]["default"], "beta")
        self.assertEqual(ref_inputs["scheduler"]["options"], ["beta", "normal", "simple"])
        self.assertEqual(ref_inputs["ref_image_size"]["options"], ["match", "max"])
        self.assertEqual(ref_inputs["ref_images"]["io_type"], "COMFY_AUTOGROW_V3")
        self.assertEqual(ref_inputs["ref_images"]["template"].max, 9)
        self.assertEqual(ref_inputs["ref_images"]["template"].prefix, "ref_image_")
        self.assertEqual(ref_inputs["ref_images"]["template"].input["name"], "ref_image")
        self.assertEqual(ref_inputs["ref_videos"]["template"].max, 3)
        self.assertEqual(ref_inputs["ref_videos"]["template"].prefix, "ref_video_")
        self.assertEqual(ref_inputs["ref_videos"]["template"].input["name"], "ref_video")
        self.assertEqual(ref_inputs["ref_video_audios"]["template"].max, 3)
        self.assertEqual(
            ref_inputs["ref_video_audios"]["template"].prefix,
            "ref_video_audio_",
        )
        self.assertEqual(
            ref_inputs["ref_video_audios"]["template"].input["name"],
            "ref_video_audio",
        )
        self.assertEqual(ref_inputs["ref_audios"]["template"].max, 3)
        self.assertEqual(ref_inputs["ref_audios"]["template"].prefix, "ref_audio_")
        self.assertEqual(ref_inputs["ref_audios"]["template"].input["name"], "ref_audio")

        loader_schema = next(item for item in schemas if item.node_id == H3_LOADER_ID)
        loader_inputs = {item["name"]: item for item in loader_schema.inputs}
        self.assertEqual(loader_schema.category, "ComfyColab/loaders")
        self.assertEqual(
            [item["io_type"] for item in loader_schema.outputs],
            ["MINIMAX_H3_BUNDLE", "MODEL", "CLIP", "VAE", "VAE"],
        )
        self.assertIs(loader_inputs["accept_h3_license"]["default"], False)

        enhancer_schema = next(item for item in schemas if item.node_id == H3_PROMPT_ID)
        enhancer_inputs = {item["name"]: item for item in enhancer_schema.inputs}
        self.assertEqual(enhancer_schema.category, "ComfyColab/prompt")
        self.assertEqual([item["name"] for item in enhancer_schema.outputs], ["enhanced_prompt"])
        self.assertEqual(
            enhancer_inputs["prompt_mode"]["options"],
            [
                "T2VA — Text only",
                "I2VA — First frame",
                "FL2VA — First + last frame",
                "L2VA — Last frame",
                "Ref2VA — Full references",
            ],
        )
        self.assertEqual(enhancer_inputs["max_tokens"]["default"], 8192)

    def test_graph_matrix_honors_gguf_fps_and_spatial_choices(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        cases = (
            ("Q3_K_S", "24", "None"),
            ("Q4_K_S", "48", "None"),
            ("Q4_K_M", "24", "1.5x"),
            ("Q3_K_S", "48", "2x"),
        )
        with self._mock_model_downloads((nodes, graph, models)):
            for gguf_model, fps, spatial_upscaler in cases:
                with self.subTest(
                    gguf_model=gguf_model,
                    fps=fps,
                    spatial_upscaler=spatial_upscaler,
                ):
                    result = self._execute(
                        facade,
                        gguf_model=gguf_model,
                        fps=fps,
                        spatial_upscaler=spatial_upscaler,
                        height=576 if spatial_upscaler == "1.5x" else 544,
                    )
                    expanded = result.expand
                    node_types = [item["class_type"] for item in expanded]

                    self.assertNotIn("CheckpointLoaderSimple", node_types)
                    self.assertNotIn("LoraLoaderModelOnly", node_types)
                    gguf_loader = next(
                        item
                        for item in expanded
                        if item["class_type"] == "UnetLoaderGGUF"
                    )
                    self.assertEqual(
                        gguf_loader["inputs"]["unet_name"],
                        GGUF_FILENAMES[gguf_model],
                    )

                    latent_video = next(
                        item
                        for item in expanded
                        if item["class_type"] == "EmptyLTXVLatentVideo"
                    )
                    latent_audio = next(
                        item
                        for item in expanded
                        if item["class_type"] == "LTXVEmptyLatentAudio"
                    )
                    conditioning = next(
                        item
                        for item in expanded
                        if item["class_type"] == "LTXVConditioning"
                    )
                    encoded_prompts = [
                        item["inputs"]["text"]
                        for item in expanded
                        if item["class_type"] == "CLIPTextEncode"
                    ]
                    create_video = next(
                        item
                        for item in expanded
                        if item["class_type"] == "CreateVideo"
                    )
                    self.assertEqual(latent_video["inputs"]["length"], 121)
                    self.assertEqual(latent_audio["inputs"]["frames_number"], 121)
                    self.assertEqual(int(latent_audio["inputs"]["frame_rate"]), 24)
                    self.assertEqual(int(conditioning["inputs"]["frame_rate"]), 24)
                    self.assertEqual(
                        encoded_prompts,
                        [
                            "A quiet tea ceremony with synchronized room ambience."
                        ],
                    )
                    self.assertEqual(int(create_video["inputs"]["fps"]), int(fps))

                    loaders = [
                        (index, item)
                        for index, item in enumerate(expanded)
                        if item["class_type"]
                        in {
                            "LatentUpscaleModelLoader",
                            "LowVRAMLatentUpscaleModelLoader",
                        }
                    ]
                    loaded_names = {
                        item["inputs"].get("model_name")
                        or item["inputs"].get("upscale_model")
                        for _, item in loaders
                    }
                    expected_spatial = SPATIAL_FILENAMES.get(spatial_upscaler)
                    if expected_spatial is None:
                        self.assertFalse(
                            loaded_names & set(SPATIAL_FILENAMES.values())
                        )
                    else:
                        self.assertIn(expected_spatial, loaded_names)

                    if fps == "48":
                        self.assertIn(TEMPORAL_FILENAME, loaded_names)
                    else:
                        self.assertNotIn(TEMPORAL_FILENAME, loaded_names)

                    expected_sampler_count = 1 if spatial_upscaler == "None" else 2
                    self.assertEqual(
                        node_types.count("SamplerCustomAdvanced"),
                        expected_sampler_count,
                    )
                    self.assertEqual(
                        node_types.count("BasicGuider"),
                        expected_sampler_count,
                    )
                    self.assertNotIn("CFGGuider", node_types)
                    sampler_names = [
                        item["inputs"]["sampler_name"]
                        for item in expanded
                        if item["class_type"] == "KSamplerSelect"
                    ]
                    self.assertEqual(
                        sampler_names,
                        ["euler"] * expected_sampler_count,
                    )
                    sigma_schedules = [
                        item["inputs"]["sigmas"]
                        for item in expanded
                        if item["class_type"] == "ManualSigmas"
                    ]
                    self.assertEqual(
                        sigma_schedules[0],
                        (
                            "1.0, 0.99375, 0.9875, 0.98125, 0.975, "
                            "0.909375, 0.725, 0.421875, 0.0"
                        ),
                    )
                    if spatial_upscaler == "None":
                        self.assertEqual(len(sigma_schedules), 1)
                    else:
                        self.assertEqual(
                            sigma_schedules[1],
                            "0.909375, 0.725, 0.421875, 0.0",
                        )

                    for loader_index, loader in loaders:
                        model_name = (
                            loader["inputs"].get("model_name")
                            or loader["inputs"].get("upscale_model")
                        )
                        if model_name not in {
                            *SPATIAL_FILENAMES.values(),
                            TEMPORAL_FILENAME,
                        }:
                            continue
                        upsampler = next(
                            item
                            for item in expanded
                            if item["class_type"] == "LTXVLatentUpsampler"
                            and item["inputs"]["upscale_model"]
                            == Link(loader_index, 0)
                        )
                        source = upsampler["inputs"]["samples"]
                        self.assertIsInstance(source, Link)
                        self.assertEqual(
                            expanded[source.node_id]["class_type"],
                            "LTXVSeparateAVLatent",
                        )
                        self.assertEqual(
                            source.index,
                            0,
                            "spatial and temporal upscalers must consume video only",
                        )

                    self.assertEqual(len(result.values), 3)
                    self.assertEqual(
                        expanded[result.values[0].node_id]["class_type"],
                        "CreateVideo",
                    )
                    self.assertIn(
                        expanded[result.values[1].node_id]["class_type"],
                        {"LTXVTiledVAEDecode", "VAEDecode"},
                    )
                    self.assertEqual(
                        expanded[result.values[2].node_id]["class_type"],
                        "LTXVAudioVAEDecode",
                    )

    def test_optional_image_adds_image_conditioning_with_public_strength(self):
        _, nodes, graph, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        image = object()
        with self._mock_model_downloads((nodes, graph, models)):
            text_result = self._execute(facade, image=None)
            image_result = self._execute(facade, image=image)

        self.assertNotIn(
            "LTXVImgToVideoConditionOnly",
            [item["class_type"] for item in text_result.expand],
        )
        preprocess = next(
            item
            for item in image_result.expand
            if item["class_type"] == "LTXVPreprocess"
        )
        self.assertIs(preprocess["inputs"]["image"], image)
        preprocess_index = image_result.expand.index(preprocess)
        conditioners = [
            item
            for item in image_result.expand
            if item["class_type"] == "LTXVImgToVideoConditionOnly"
        ]
        self.assertTrue(conditioners)
        self.assertTrue(
            all(
                conditioner["inputs"]["image"] == Link(preprocess_index, 0)
                for conditioner in conditioners
            )
        )
        self.assertIn(0.7, {item["inputs"]["strength"] for item in conditioners})

    def test_missing_required_ltx_node_fails_before_any_download(self):
        _, nodes, _, models = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]
        sys.modules["nodes"].NODE_CLASS_MAPPINGS.pop("LTXVLatentUpsampler")

        with mock.patch.object(
            models,
            "ensure_model_assets",
            side_effect=AssertionError("download ran before compatibility check"),
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "LTXVLatentUpsampler.*(?:bootstrap|refresh)|"
                "(?:bootstrap|refresh).*LTXVLatentUpsampler",
            ):
                self._execute(facade, spatial_upscaler="2x")

    def test_one_point_five_spatial_requires_exactly_scalable_dimensions(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[PUBLIC_NODE_ID]

        with mock.patch.object(nodes, "ensure_model_assets") as ensure:
            with self.assertRaisesRegex(ValueError, "divisible by 64"):
                facade.execute(
                    prompt="A slow camera move through a quiet greenhouse.",
                    gguf_model="Q3_K_S",
                    fps="24",
                    spatial_upscaler="1.5x",
                    width=960,
                    height=544,
                    frame_count=121,
                    seed=0,
                )
        ensure.assert_not_called()

    def test_h3_loader_license_gate_blocks_all_side_effects(self):
        _, nodes, graph, models = self._modules()
        loader = nodes.NODE_CLASS_MAPPINGS[H3_LOADER_ID]
        with mock.patch.object(
            nodes,
            "ensure_h3_model_assets",
            side_effect=AssertionError("download ran before license gate"),
        ):
            with self.assertRaisesRegex(PermissionError, "MiniMax H3 Community License"):
                loader.execute(accept_h3_license=False)

        with mock.patch.object(
            nodes,
            "ensure_h3_model_assets",
            return_value={
                "model": "minimax_h3_fl2va_pruned_int8_convrot.safetensors",
                "text_encoder": "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "video_vae": "minimax_h3_video_vae_fp16.safetensors",
                "audio_vae": "minimax_h3_audio_vae_fp32.safetensors",
            },
        ) as ensure:
            calls = []

            class Model:
                def __init__(self):
                    self.model_options = {}

                def clone(self):
                    cloned = Model()
                    cloned.model_options = dict(self.model_options)
                    return cloned

            class Unet:
                def load_unet(self, filename, weight_dtype="default"):
                    calls.append(("unet", filename, weight_dtype))
                    return (Model(),)

            class Clip:
                def load_clip(self, filename, type):
                    calls.append(("clip", filename, type))
                    return ("CLIP_OBJECT",)

            class Vae:
                def load_vae(self, filename):
                    calls.append(("vae", filename))
                    return (f"VAE_OBJECT:{filename}",)

            sys.modules["nodes"].NODE_CLASS_MAPPINGS.update(
                {"UNETLoader": Unet, "CLIPLoader": Clip, "VAELoader": Vae}
            )
            result = loader.execute(accept_h3_license=True)

        ensure.assert_called_once()
        self.assertFalse(hasattr(result, "expand"))
        bundle, model, clip, video_vae, audio_vae = result
        self.assertEqual(bundle["variant"], "FL2VA")
        self.assertEqual(bundle["family"], "minimax_h3")
        self.assertIs(bundle["model"], model)
        self.assertEqual(bundle["attention_backend"], "sage")
        self.assertIn("optimized_attention_override", model.model_options["transformer_options"])
        self.assertEqual(clip, "CLIP_OBJECT")
        self.assertEqual(video_vae, "VAE_OBJECT:minimax_h3_video_vae_fp16.safetensors")
        self.assertEqual(audio_vae, "VAE_OBJECT:minimax_h3_audio_vae_fp32.safetensors")
        self.assertIn(
            (
                "clip",
                "qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
                "minimax",
            ),
            calls,
        )

    def test_h3_fl2va_graph_uses_native_audio_video_path_and_frame_grid(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[H3_VIDEO_ID]
        bundle = self._h3_bundle("FL2VA")
        first_frame = object()
        last_frame = object()

        result = facade.execute(
            bundle=bundle,
            prompt="A rainy street scene with synchronized footsteps.",
            duration_seconds=5,
            width=864,
            height=480,
            seed=123,
            first_frame=first_frame,
            last_frame=last_frame,
        )
        expanded = result.expand
        node_types = [item["class_type"] for item in expanded]
        self.assertEqual(node_types.count("MiniMaxH3ImageToVideo"), 1)
        self.assertNotIn("EmptyMiniMaxH3LatentAV", node_types)
        self.assertIn("VAEDecodeAudio", node_types)
        self.assertIn("CreateVideo", node_types)
        self.assertEqual(node_types.count("SamplerCustomAdvanced"), 1)
        self.assertEqual(node_types.count("BasicScheduler"), 1)
        conditioning = next(
            item for item in expanded if item["class_type"] == "MiniMaxH3ImageToVideo"
        )
        self.assertIs(conditioning["inputs"]["first_frame"], first_frame)
        self.assertIs(conditioning["inputs"]["last_frame"], last_frame)
        self.assertEqual(conditioning["inputs"]["length"], 124)
        self.assertEqual(conditioning["inputs"]["width"], 864)
        self.assertEqual(conditioning["inputs"]["height"], 480)
        sampler = next(item for item in expanded if item["class_type"] == "KSamplerSelect")
        scheduler = next(item for item in expanded if item["class_type"] == "BasicScheduler")
        create_video = next(item for item in expanded if item["class_type"] == "CreateVideo")
        sampled_index = next(
            index
            for index, item in enumerate(expanded)
            if item["class_type"] == "SamplerCustomAdvanced"
        )
        video_decode = next(item for item in expanded if item["class_type"] == "VAEDecode")
        audio_decode = next(item for item in expanded if item["class_type"] == "VAEDecodeAudio")
        self.assertEqual(sampler["inputs"]["sampler_name"], "res_multistep")
        self.assertEqual(scheduler["inputs"]["scheduler"], "simple")
        self.assertEqual(scheduler["inputs"]["steps"], 20)
        self.assertEqual(float(create_video["inputs"]["fps"]), 24.0)
        self.assertEqual(video_decode["inputs"]["samples"], Link(sampled_index, 0))
        self.assertEqual(audio_decode["inputs"]["samples"], Link(sampled_index, 0))
        self.assertEqual(
            expanded[result.values[0].node_id]["class_type"],
            "CreateVideo",
        )
        self.assertEqual(
            expanded[result.values[2].node_id]["class_type"],
            "VAEDecodeAudio",
        )

    def test_h3_ref2va_graph_preserves_reference_order_and_limits(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[H3_REFERENCE_ID]
        bundle = self._h3_bundle("Ref2VA")
        ref_image = object()
        ref_video = {"frames": [object()] * 72}
        paired_audio = {"waveform": [0.0] * 96000, "sample_rate": 32000}
        standalone_audio = {"waveform": [0.0] * 80000, "sample_rate": 32000}

        result = facade.execute(
            bundle=bundle,
            prompt="Use <Picture 1> for identity, <Video 1> for motion, and <Audio 1> for voice.",
            duration_seconds=4,
            width=864,
            height=480,
            seed=7,
            ref_images={"ref_image_0": ref_image},
            ref_videos={"ref_video_0": ref_video},
            ref_video_audios={"ref_video_audio_0": paired_audio},
            ref_audios={"ref_audio_0": standalone_audio},
        )
        expanded = result.expand
        self.assertNotIn(
            "EmptyMiniMaxH3LatentAV",
            [item["class_type"] for item in expanded],
        )
        reference = next(
            item for item in expanded if item["class_type"] == "MiniMaxH3ReferenceToVideo"
        )
        self.assertEqual(reference["inputs"]["ref_images"], {"ref_image_0": ref_image})
        self.assertEqual(reference["inputs"]["ref_videos"], {"ref_video_0": ref_video})
        self.assertIs(
            reference["inputs"]["ref_video_audios"]["ref_video_audio_0"],
            paired_audio,
        )
        self.assertIs(reference["inputs"]["ref_audios"]["ref_audio_0"], standalone_audio)
        self.assertEqual(reference["inputs"]["length"], 107)
        scheduler = next(item for item in expanded if item["class_type"] == "BasicScheduler")
        self.assertEqual(scheduler["inputs"]["scheduler"], "beta")

        with self.assertRaisesRegex(ValueError, "at least one image or video"):
            facade.execute(
                bundle=bundle,
                prompt="Audio only should fail.",
                ref_audios={
                    "ref_audio_0": {
                        "waveform": [0.0] * 96000,
                        "sample_rate": 32000,
                    }
                },
            )
        with self.assertRaisesRegex(ValueError, "12 reference files"):
            facade.execute(
                bundle=bundle,
                prompt="Too many files.",
                ref_images={f"ref_image_{index}": object() for index in range(9)},
                ref_videos={"ref_video_0": {"frames": [object()] * 48}},
                ref_video_audios={
                    "ref_video_audio_0": {
                        "waveform": [0.0] * 64000,
                        "sample_rate": 32000,
                    }
                },
                ref_audios={
                    "ref_audio_0": {
                        "waveform": [0.0] * 64000,
                        "sample_rate": 32000,
                    },
                    "ref_audio_1": {
                        "waveform": [0.0] * 64000,
                        "sample_rate": 32000,
                    },
                },
            )
        with self.assertRaisesRegex(ValueError, "Ref2VA bundle"):
            facade.execute(
                bundle=self._h3_bundle("FL2VA"),
                prompt="Wrong family.",
                ref_images={"ref_image_0": object()},
            )

    def test_h3_ref2va_malformed_autogrow_group_fails_closed(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[H3_REFERENCE_ID]
        with self.assertRaisesRegex(ValueError, "expandable input groups"):
            facade.execute(
                bundle=self._h3_bundle("Ref2VA"),
                prompt="Malformed autogrow bucket must not be ignored.",
                ref_images={"ref_image_0": object()},
                ref_videos="not-an-autogrow-group",
            )

    def test_h3_ref2va_reference_duration_derives_from_carriers_and_fails_closed(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[H3_REFERENCE_ID]
        bundle = self._h3_bundle("Ref2VA")
        facade.execute(
            bundle=bundle,
            prompt="Use <Video 1> and <Audio 1>.",
            ref_videos={"ref_video_0": {"frames": [object()] * 48}},
            ref_video_audios={
                "ref_video_audio_0": {
                    "waveform": [0.0] * 64000,
                    "sample_rate": 32000,
                }
            },
        )
        with self.assertRaisesRegex(ValueError, "reference videos"):
            facade.execute(
                bundle=bundle,
                prompt="Too short video.",
                ref_videos={"ref_video_0": {"frames": [object()] * 47}},
            )
        with self.assertRaisesRegex(ValueError, "reference audio clips"):
            facade.execute(
                bundle=bundle,
                prompt="Too short audio.",
                ref_images={"ref_image_0": object()},
                ref_audios={
                    "ref_audio_0": {
                        "waveform": [0.0] * 63999,
                        "sample_rate": 32000,
                    }
                },
            )
        with self.assertRaisesRegex(ValueError, "video duration is unavailable"):
            facade.execute(
                bundle=bundle,
                prompt="Unknown video carrier.",
                ref_videos={"ref_video_0": object()},
            )
        with self.assertRaisesRegex(ValueError, "video duration is unavailable"):
            facade.execute(
                bundle=bundle,
                prompt="Metadata-only video carrier.",
                ref_videos={
                    "ref_video_0": {
                        "duration_seconds": 3.0,
                        **{f"metadata_{index}": index for index in range(48)},
                    }
                },
            )
        with self.assertRaisesRegex(ValueError, "waveform and sample_rate"):
            facade.execute(
                bundle=bundle,
                prompt="Unknown audio carrier.",
                ref_images={"ref_image_0": object()},
                ref_audios={"ref_audio_0": {"duration_seconds": 3.0}},
            )
        with self.assertRaisesRegex(ValueError, "total reference-video duration"):
            facade.execute(
                bundle=bundle,
                prompt="Too much video.",
                ref_videos={
                    "ref_video_0": {"frames": [object()] * 120},
                    "ref_video_1": {"frames": [object()] * 120},
                    "ref_video_2": {"frames": [object()] * 121},
                },
            )
        with self.assertRaisesRegex(ValueError, "total reference-audio duration"):
            facade.execute(
                bundle=bundle,
                prompt="Too much audio.",
                ref_images={"ref_image_0": object()},
                ref_audios={
                    "ref_audio_0": {
                        "waveform": [0.0] * 160000,
                        "sample_rate": 32000,
                    },
                    "ref_audio_1": {
                        "waveform": [0.0] * 160000,
                        "sample_rate": 32000,
                    },
                    "ref_audio_2": {
                        "waveform": [0.0] * 160001,
                        "sample_rate": 32000,
                    },
                },
            )

    def test_h3_validation_rejects_invalid_prompt_canvas_and_duration(self):
        _, nodes, _, _ = self._modules()
        facade = nodes.NODE_CLASS_MAPPINGS[H3_VIDEO_ID]
        bundle = self._h3_bundle("FL2VA")
        cases = (
            {"prompt": "", "duration_seconds": 5, "width": 864, "height": 480},
            {"prompt": "ok", "duration_seconds": 3, "width": 864, "height": 480},
            {"prompt": "ok", "duration_seconds": 5, "width": 865, "height": 480},
            {"prompt": "ok", "duration_seconds": 5, "width": 1376, "height": 768},
        )
        for kwargs in cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    facade.execute(bundle=bundle, seed=1, **kwargs)

    @staticmethod
    def _h3_bundle(variant):
        return {
            "family": "minimax_h3",
            "variant": variant,
            "model": Link(100, 0),
            "text_encoder": Link(101, 0),
            "video_vae": Link(102, 0),
            "audio_vae": Link(103, 0),
            "filenames": {},
        }


if __name__ == "__main__":
    unittest.main()
