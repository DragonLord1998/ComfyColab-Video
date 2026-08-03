from __future__ import annotations

import json
import re
import subprocess
import sys
import unittest
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "comfycolab-pack.json"
PYPROJECT_PATH = ROOT / "pyproject.toml"
IMMUTABLE_REF = re.compile(r"^[0-9a-f]{40}$")
EXPECTED_TOP_LEVEL = {
    "schema",
    "id",
    "version",
    "display_name",
    "compatibility",
    "node_roots",
    "dependencies",
    "patches",
    "environments",
    "hooks",
    "runtime_env",
    "workflows",
    "probes",
    "health_checks",
    "licenses",
}
EXPECTED_NODE_IDS = [
    "ComfyColabLTX23Video",
    "ComfyColabMiniMaxH3BundleLoader",
    "ComfyColabMiniMaxH3Video",
    "ComfyColabMiniMaxH3ReferenceVideo",
]


def safe_relative_path(value: str) -> bool:
    path = PurePosixPath(value)
    return bool(value) and not path.is_absolute() and ".." not in path.parts


class PackManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))

    def test_manifest_has_exact_v1_top_level_contract(self) -> None:
        self.assertEqual(set(self.manifest), EXPECTED_TOP_LEVEL)
        self.assertEqual(self.manifest["schema"], 1)
        self.assertEqual(self.manifest["id"], "video")
        self.assertEqual(self.manifest["version"], "0.3.0-dev1")
        self.assertEqual(
            self.manifest["compatibility"]["core_manifest_api"],
            1,
        )
        comfyui = self.manifest["compatibility"]["comfyui"]
        self.assertEqual(comfyui["tested_refs"], comfyui["compatible_refs"])
        self.assertTrue(
            all(IMMUTABLE_REF.fullmatch(ref) for ref in comfyui["compatible_refs"])
        )

    def test_manifest_and_package_versions_match(self) -> None:
        matches = re.findall(
            r'^version = "([^"]+)"$',
            PYPROJECT_PATH.read_text(encoding="utf-8"),
            flags=re.MULTILINE,
        )
        self.assertEqual(matches, [self.manifest["version"]])

    def test_node_roots_hooks_workflows_and_notices_exist(self) -> None:
        for node_root in self.manifest["node_roots"]:
            self.assertTrue(safe_relative_path(node_root["source"]))
            self.assertNotIn("/", node_root["target"])
            self.assertTrue((ROOT / node_root["source"]).is_dir())

        for hook in self.manifest["hooks"].values():
            self.assertEqual(hook["network"], "none")
            self.assertTrue(safe_relative_path(hook["path"]))
            self.assertTrue((ROOT / hook["path"]).is_file())

        self.assertEqual(
            self.manifest["workflows"],
            [
                "workflows/comfycolab_ltx23_text_image_to_video.json",
                "workflows/comfycolab_minimax_h3_text_image_to_video.json",
                "workflows/comfycolab_minimax_h3_reference_to_video.json",
                "workflows/comfycolab_minimax_h3_fl2va_to_ref2va_chain.json",
            ],
        )
        for workflow in self.manifest["workflows"]:
            self.assertTrue(safe_relative_path(workflow))
            json.loads((ROOT / workflow).read_text(encoding="utf-8"))

        for license_record in self.manifest["licenses"]:
            self.assertTrue(license_record["url"].startswith("https://"))
            self.assertTrue(safe_relative_path(license_record["notice"]))
            self.assertTrue((ROOT / license_record["notice"]).is_file())

    def test_dependency_and_node_inventory_are_immutable(self) -> None:
        dependencies = self.manifest["dependencies"]
        self.assertEqual(
            [item["id"] for item in dependencies],
            ["comfyui-gguf", "comfyui-ltxvideo"],
        )
        expected_sources = {
            "comfyui-gguf": {
                "repository": "https://github.com/city96/ComfyUI-GGUF.git",
                "ref": "6ea2651e7df66d7585f6ffee804b20e92fb38b8a",
                "requirements_file": "requirements.txt",
            },
            "comfyui-ltxvideo": {
                "repository": "https://github.com/Lightricks/ComfyUI-LTXVideo.git",
                "ref": "aceeae9635f6d493f2893ba3c411a1c36031788a",
                "requirements_file": "requirements.txt",
            },
        }
        for dependency in dependencies:
            self.assertEqual(dependency["kind"], "git")
            self.assertEqual(dependency["scope"], "comfyui")
            self.assertTrue(dependency["repository"].startswith("https://"))
            self.assertTrue(dependency["repository"].endswith(".git"))
            self.assertRegex(dependency["ref"], IMMUTABLE_REF)
            self.assertTrue(safe_relative_path(dependency["destination"]))
            self.assertEqual(
                {
                    key: dependency[key]
                    for key in expected_sources[dependency["id"]]
                },
                expected_sources[dependency["id"]],
            )

        self.assertEqual(
            self.manifest["health_checks"]["node_ids"],
            EXPECTED_NODE_IDS,
        )
        post_start = next(
            probe
            for probe in self.manifest["probes"]
            if probe["phase"] == "post_start"
        )
        self.assertEqual(post_start["type"], "comfy_node_ids")
        self.assertEqual(post_start["values"], EXPECTED_NODE_IDS)

    def test_hooks_return_structured_success(self) -> None:
        for name, hook in self.manifest["hooks"].items():
            completed = subprocess.run(
                [sys.executable, str(ROOT / hook["path"])],
                check=True,
                capture_output=True,
                text=True,
                timeout=hook["timeout_seconds"],
            )
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["schema"], 1)
            self.assertEqual(payload["hook"], name)
            self.assertEqual(payload["status"], "ok")


if __name__ == "__main__":
    unittest.main()
