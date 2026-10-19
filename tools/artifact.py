"""Artifact packaging shared by the graph-surgery producers.

An artifact is the model plus its siblings, cloned copy-on-write, and one
manifest. The manifest is the cross-language contract the browser reads, so
its shape lives here once rather than in each producer.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import onnx


class ArtifactError(RuntimeError):
    """Raised when an artifact directory cannot be prepared."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clone_or_copy(source: Path, target: Path) -> None:
    """Clone copy-on-write when the filesystem supports it, else copy.

    The artifact must be self-contained: onnx.checker rejects symlinks and
    multi-link hardlinks for external data. `cp -c` gives an independent file
    without duplicating the weight file on disk.
    """
    result = subprocess.run(
        ["cp", "-c", str(source), str(target)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        shutil.copy2(source, target)


def _prepare_dir(model_path: Path, out_dir: Path, force: bool) -> Path:
    if out_dir.exists() and any(out_dir.iterdir()):
        if not force:
            raise ArtifactError(f"{out_dir} is not empty; pass --force to replace")
        shutil.rmtree(out_dir)

    onnx_dir = out_dir / "onnx"
    onnx_dir.mkdir(parents=True)

    base_root = model_path.parent.parent
    for item in base_root.iterdir():
        if item.name == "onnx":
            continue
        clone_or_copy(item, out_dir / item.name)
    for item in model_path.parent.iterdir():
        if item.name == model_path.name:
            continue
        clone_or_copy(item, onnx_dir / item.name)
    return onnx_dir


def write_artifact(
    model: onnx.ModelProto,
    model_path: Path,
    out_dir: Path,
    force: bool,
    *,
    tool: str,
    fusion: dict,
    producer: dict | None = None,
) -> Path:
    """Write the model and manifest; return the written model path."""
    onnx_dir = _prepare_dir(model_path, out_dir, force)
    out_model = onnx_dir / model_path.name
    onnx.save(model, out_model)

    manifest = {
        "id": out_dir.name,
        "base": {
            "path": str(model_path),
            "decoder_sha256": sha256_file(model_path),
        },
        "producer": {
            "kind": "surgery",
            "tool": tool,
            **(producer or {}),
        },
        "dtype": "q4f16",
        "opset": [[op.domain or "", op.version] for op in model.opset_import],
        "files": {f"onnx/{model_path.name}": sha256_file(out_model)},
        "fusion": fusion,
        "runtime": {"browser_equivalence": "pending"},
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False)
    )
    return out_model
