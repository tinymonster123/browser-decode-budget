"""Tests for the shared artifact packager.

The fixture is a tiny ONNX model plus two siblings, so the test exercises the
copy layout and the manifest contract without any weights.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import onnx
from onnx import helper

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from artifact import ArtifactError, sha256_file, write_artifact  # noqa: E402


def tiny_model() -> onnx.ModelProto:
    graph = helper.make_graph([], "tiny", [], [])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class ArtifactTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.base_root = self.root / "base"
        self.model_path = self.base_root / "onnx" / "decoder.onnx"
        self.model_path.parent.mkdir(parents=True)

        self.model = tiny_model()
        onnx.save(self.model, self.model_path)
        (self.base_root / "config.json").write_text("{}")
        (self.model_path.parent / "weights.bin").write_text("weights")

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_sha256_file_hashes_content(self) -> None:
        path = self.root / "payload.bin"
        path.write_bytes(b"browser-decode-budget")

        self.assertEqual(sha256_file(path), hashlib.sha256(b"browser-decode-budget").hexdigest())

    def test_write_artifact_copies_siblings_and_writes_manifest(self) -> None:
        out_dir = self.root / "artifact"

        out_model = write_artifact(
            self.model,
            self.model_path,
            out_dir,
            force=False,
            tool="tools/test.py",
            producer={"layers": [1, 2]},
            fusion={"net_saved": 7},
        )

        self.assertTrue(out_model.exists())
        self.assertTrue((out_dir / "config.json").exists())
        self.assertTrue((out_dir / "onnx" / "weights.bin").exists())

        manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(manifest["id"], "artifact")
        self.assertEqual(manifest["base"]["path"], str(self.model_path))
        self.assertEqual(
            manifest["base"]["decoder_sha256"], sha256_file(self.model_path)
        )
        self.assertEqual(manifest["producer"]["kind"], "surgery")
        self.assertEqual(manifest["producer"]["tool"], "tools/test.py")
        self.assertEqual(manifest["producer"]["layers"], [1, 2])
        self.assertEqual(manifest["dtype"], "q4f16")
        self.assertEqual(manifest["opset"], [["", 17]])
        self.assertEqual(
            manifest["files"]["onnx/decoder.onnx"], sha256_file(out_model)
        )
        self.assertEqual(manifest["fusion"], {"net_saved": 7})
        self.assertEqual(manifest["runtime"], {"browser_equivalence": "pending"})

    def test_non_empty_out_dir_requires_force(self) -> None:
        out_dir = self.root / "artifact"
        out_dir.mkdir()
        (out_dir / "stale.txt").write_text("stale")

        with self.assertRaises(ArtifactError):
            write_artifact(
                self.model,
                self.model_path,
                out_dir,
                force=False,
                tool="tools/test.py",
                fusion={},
            )

    def test_force_replaces_out_dir(self) -> None:
        out_dir = self.root / "artifact"
        out_dir.mkdir()
        (out_dir / "stale.txt").write_text("stale")

        write_artifact(
            self.model,
            self.model_path,
            out_dir,
            force=True,
            tool="tools/test.py",
            fusion={},
        )

        self.assertFalse((out_dir / "stale.txt").exists())
        self.assertTrue((out_dir / "manifest.json").exists())


if __name__ == "__main__":
    unittest.main()
