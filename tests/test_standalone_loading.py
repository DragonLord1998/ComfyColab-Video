from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import types
import unittest
from pathlib import Path

from test_ltx_node_pack import FakeIO


ROOT = Path(__file__).resolve().parents[1]


def load_root_package():
    name = "normalized_video_manager_name"
    for module_name in list(sys.modules):
        if module_name == name or module_name.startswith(name + "."):
            del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "__init__.py",
        submodule_search_locations=[str(ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    assert spec.loader
    spec.loader.exec_module(package)
    return package


class StandaloneLoadingTests(unittest.TestCase):
    def setUp(self):
        self.saved_modules = {
            name: sys.modules.get(name)
            for name in ("comfy_api", "comfy_api.latest")
        }
        latest = types.ModuleType("comfy_api.latest")
        latest.io = FakeIO
        latest.ComfyExtension = type("ComfyExtension", (), {})
        api = types.ModuleType("comfy_api")
        api.latest = latest
        sys.modules.update(
            {"comfy_api": api, "comfy_api.latest": latest}
        )

    def tearDown(self):
        for name, module in self.saved_modules.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    def test_root_discovers_the_video_node_from_any_folder_name(self):
        package = load_root_package()
        extension = asyncio.run(package.comfy_entrypoint())
        node_classes = asyncio.run(extension.get_node_list())
        node_ids = {node.define_schema().node_id for node in node_classes}
        declared = set(
            json.loads((ROOT / "node_list.json").read_text(encoding="utf-8"))
        )
        self.assertEqual(node_ids, declared)

    def test_manager_files_and_pinned_dependencies_are_present(self):
        self.assertTrue((ROOT / "requirements.txt").is_file())
        self.assertTrue((ROOT / "install.py").is_file())
        installer = (ROOT / "install.py").read_text(encoding="utf-8")
        self.assertIn("6ea2651e7df66d7585f6ffee804b20e92fb38b8a", installer)
        self.assertIn("aceeae9635f6d493f2893ba3c411a1c36031788a", installer)
        self.assertIn("kornia==0.8.1", installer)
        self.assertIn("and _commit(custom_nodes / name) == revision", installer)


if __name__ == "__main__":
    unittest.main()
