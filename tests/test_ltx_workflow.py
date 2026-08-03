from __future__ import annotations

import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / "workflows" / "comfycolab_ltx23_text_image_to_video.json"
H3_FL2VA_WORKFLOW = (
    ROOT / "workflows" / "comfycolab_minimax_h3_text_image_to_video.json"
)
H3_REF2VA_WORKFLOW = (
    ROOT / "workflows" / "comfycolab_minimax_h3_reference_to_video.json"
)
H3_CHAIN_WORKFLOW = (
    ROOT / "workflows" / "comfycolab_minimax_h3_fl2va_to_ref2va_chain.json"
)
PUBLIC_NODE_ID = "ComfyColabLTX23Video"
H3_LOADER_ID = "ComfyColabMiniMaxH3BundleLoader"
H3_VIDEO_ID = "ComfyColabMiniMaxH3Video"
H3_REFERENCE_ID = "ComfyColabMiniMaxH3ReferenceVideo"


class LTXWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.workflow = json.loads(WORKFLOW.read_text(encoding="utf-8"))
        self.nodes = {node["id"]: node for node in self.workflow["nodes"]}
        self.links = {link[0]: link for link in self.workflow["links"]}

    def test_workflow_uses_one_public_facade_and_valid_native_video_links(self):
        types = [node["type"] for node in self.nodes.values()]
        self.assertEqual(types.count(PUBLIC_NODE_ID), 1)
        self.assertEqual(types.count("LoadImage"), 1)
        self.assertEqual(types.count("SaveVideo"), 1)

        for link in self.workflow["links"]:
            link_id, source_id, source_slot, target_id, target_slot, socket_type = link
            self.assertGreater(link_id, 0)
            self.assertIn(source_id, self.nodes)
            self.assertIn(target_id, self.nodes)
            self.assertGreaterEqual(source_slot, 0)
            self.assertGreaterEqual(target_slot, 0)
            self.assertIsInstance(socket_type, str)
            self.assertEqual(self.links[link_id], link)

        facade = next(
            node for node in self.nodes.values() if node["type"] == PUBLIC_NODE_ID
        )
        load_image = next(
            node for node in self.nodes.values() if node["type"] == "LoadImage"
        )
        save_video = next(
            node for node in self.nodes.values() if node["type"] == "SaveVideo"
        )
        self.assertEqual(
            [output["type"] for output in facade["outputs"]],
            ["VIDEO", "IMAGE", "AUDIO"],
        )

        image_input = next(
            item for item in facade["inputs"] if item["name"] == "image"
        )
        video_input = next(
            item for item in save_video["inputs"] if item["name"] == "video"
        )
        self.assertTrue(
            {"video", "filename_prefix", "format", "codec"}
            <= {item["name"] for item in save_video["inputs"]}
        )
        self.assertEqual(
            save_video["widgets_values"],
            ["video/ComfyColab_LTX23", "auto", "auto"],
        )
        self.assertIsNone(image_input.get("link"))
        self.assertIsNotNone(video_input.get("link"))
        video_link = self.links[video_input["link"]]
        self.assertIsNone(load_image["outputs"][0]["links"])
        self.assertEqual(video_link[1:4], [facade["id"], 0, save_video["id"]])
        self.assertEqual(video_link[5], "VIDEO")

    def test_workflow_defaults_match_the_public_ltx23_contract(self):
        facade = next(
            node for node in self.nodes.values() if node["type"] == PUBLIC_NODE_ID
        )
        input_names = {item["name"] for item in facade["inputs"]}
        self.assertTrue(
            {
                "prompt",
                "gguf_model",
                "fps",
                "spatial_upscaler",
                "width",
                "height",
                "frame_count",
                "seed",
                "image_strength",
                "image",
            }
            <= input_names
        )

        values = facade["widgets_values"]
        normalized = {str(value) for value in values}
        self.assertIn("Q3_K_S", normalized)
        self.assertIn("24", normalized)
        self.assertIn("2x", normalized)
        self.assertIn("960", normalized)
        self.assertIn("544", normalized)
        self.assertIn("121", normalized)
        self.assertIn("0", normalized)
        self.assertTrue(
            any(abs(float(value) - 0.7) < 1e-9 for value in values if isinstance(value, float))
        )
        self.assertTrue(
            any(
                isinstance(value, str)
                and len(value.strip()) >= 20
                and value not in {"Q3_K_S", "24", "2x"}
                for value in values
            ),
            "The bundled workflow should include a useful example prompt.",
        )

    def test_h3_fl2va_workflow_uses_one_loader_one_facade_and_savevideo(self):
        workflow = json.loads(H3_FL2VA_WORKFLOW.read_text(encoding="utf-8"))
        self.assertEqual(workflow["last_link_id"], 2)
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        types = [node["type"] for node in nodes.values()]
        self.assertEqual(types.count(H3_LOADER_ID), 1)
        self.assertEqual(types.count(H3_VIDEO_ID), 1)
        self.assertEqual(types.count("SaveVideo"), 1)
        self.assertEqual(types.count("LoadImage"), 2)

        loader = next(node for node in nodes.values() if node["type"] == H3_LOADER_ID)
        facade = next(node for node in nodes.values() if node["type"] == H3_VIDEO_ID)
        save = next(node for node in nodes.values() if node["type"] == "SaveVideo")
        self.assertEqual(loader["widgets_values"][0], "FL2VA - Text / First / Last Frame")
        self.assertIs(loader["widgets_values"][1], False)
        self.assertEqual(
            [output["type"] for output in loader["outputs"]],
            ["MINIMAX_H3_BUNDLE", "MODEL", "CLIP", "VAE", "VAE"],
        )
        bundle_input = next(item for item in facade["inputs"] if item["name"] == "bundle")
        first_input = next(item for item in facade["inputs"] if item["name"] == "first_frame")
        last_input = next(item for item in facade["inputs"] if item["name"] == "last_frame")
        video_input = next(item for item in save["inputs"] if item["name"] == "video")
        self.assertEqual(links[bundle_input["link"]][1:4], [loader["id"], 0, facade["id"]])
        self.assertIsNone(first_input["link"])
        self.assertIsNone(last_input["link"])
        self.assertEqual(links[video_input["link"]][1:4], [facade["id"], 0, save["id"]])
        self.assertIn("864", {str(value) for value in facade["widgets_values"]})
        self.assertIn("480", {str(value) for value in facade["widgets_values"]})

    def test_h3_ref2va_workflow_has_reference_media_branches_and_tags(self):
        workflow = json.loads(H3_REF2VA_WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        types = [node["type"] for node in nodes.values()]
        self.assertEqual(types.count(H3_LOADER_ID), 1)
        self.assertEqual(types.count(H3_REFERENCE_ID), 1)
        self.assertEqual(types.count("SaveVideo"), 1)
        self.assertEqual(types.count("LoadImage"), 1)
        self.assertEqual(types.count("LoadVideo"), 1)
        self.assertEqual(types.count("GetVideoComponents"), 1)
        self.assertEqual(types.count("LoadAudio"), 1)

        loader = next(node for node in nodes.values() if node["type"] == H3_LOADER_ID)
        facade = next(node for node in nodes.values() if node["type"] == H3_REFERENCE_ID)
        load_video = next(node for node in nodes.values() if node["type"] == "LoadVideo")
        components = next(node for node in nodes.values() if node["type"] == "GetVideoComponents")
        self.assertEqual(
            loader["widgets_values"][0],
            "Ref2VA - Reference Images / Video / Audio",
        )
        self.assertEqual(load_video["inputs"][0]["name"], "file")
        self.assertEqual(load_video["widgets_values"], ["reference_motion.mp4"])
        self.assertEqual([output["type"] for output in load_video["outputs"]], ["VIDEO"])
        self.assertEqual(
            [output["type"] for output in components["outputs"]],
            ["IMAGE", "AUDIO", "FLOAT", "INT"],
        )
        prompt = facade["widgets_values"][0]
        self.assertIn("<Picture 1>", prompt)
        self.assertIn("<Video 1>", prompt)
        self.assertIn("<Audio 1>", prompt)

        linked_inputs = {
            item["name"]: item["link"]
            for item in facade["inputs"]
            if item.get("link") is not None
        }
        self.assertEqual(
            {
                "bundle",
                "ref_images.ref_image_0",
                "ref_videos.ref_video_0",
                "ref_video_audios.ref_video_audio_0",
                "ref_audios.ref_audio_0",
            }
            <= set(linked_inputs),
            True,
        )
        self.assertEqual(links[linked_inputs["bundle"]][5], "MINIMAX_H3_BUNDLE")
        self.assertEqual(links[linked_inputs["ref_images.ref_image_0"]][5], "IMAGE")
        self.assertEqual(links[7][1:4], [load_video["id"], 0, components["id"]])
        self.assertEqual(links[7][5], "VIDEO")
        self.assertEqual(
            links[linked_inputs["ref_videos.ref_video_0"]][1:4],
            [components["id"], 0, facade["id"]],
        )
        self.assertEqual(links[linked_inputs["ref_videos.ref_video_0"]][5], "IMAGE")
        self.assertEqual(
            links[linked_inputs["ref_video_audios.ref_video_audio_0"]][1:4],
            [components["id"], 1, facade["id"]],
        )
        self.assertEqual(
            links[linked_inputs["ref_video_audios.ref_video_audio_0"]][5],
            "AUDIO",
        )
        self.assertEqual(links[linked_inputs["ref_audios.ref_audio_0"]][5], "AUDIO")

    def test_h3_chain_workflow_routes_fl2va_frames_and_audio_into_ref2va(self):
        workflow = json.loads(H3_CHAIN_WORKFLOW.read_text(encoding="utf-8"))
        nodes = {node["id"]: node for node in workflow["nodes"]}
        links = {link[0]: link for link in workflow["links"]}
        types = [node["type"] for node in nodes.values()]
        self.assertEqual(types.count(H3_LOADER_ID), 2)
        self.assertEqual(types.count(H3_VIDEO_ID), 1)
        self.assertEqual(types.count(H3_REFERENCE_ID), 1)
        self.assertEqual(types.count("SaveVideo"), 1)
        loaders = [node for node in nodes.values() if node["type"] == H3_LOADER_ID]
        self.assertEqual(
            [loader["widgets_values"][0] for loader in loaders],
            [
                "FL2VA - Text / First / Last Frame",
                "Ref2VA - Reference Images / Video / Audio",
            ],
        )
        reference = next(node for node in nodes.values() if node["type"] == H3_REFERENCE_ID)
        linked_inputs = {
            item["name"]: item["link"]
            for item in reference["inputs"]
            if item.get("link") is not None
        }
        self.assertEqual(links[linked_inputs["bundle"]][5], "MINIMAX_H3_BUNDLE")
        self.assertEqual(links[linked_inputs["ref_videos.ref_video_0"]][1:4], [2, 1, 4])
        self.assertEqual(links[linked_inputs["ref_videos.ref_video_0"]][5], "IMAGE")
        self.assertEqual(links[linked_inputs["ref_video_audios.ref_video_audio_0"]][1:4], [2, 2, 4])
        self.assertEqual(links[linked_inputs["ref_video_audios.ref_video_audio_0"]][5], "AUDIO")
        save = next(node for node in nodes.values() if node["type"] == "SaveVideo")
        video_input = next(item for item in save["inputs"] if item["name"] == "video")
        self.assertEqual(links[video_input["link"]][1:4], [reference["id"], 0, save["id"]])


if __name__ == "__main__":
    unittest.main()
