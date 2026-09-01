"""The browser bundle must stay consistent with the Python model.

These tests guard the demo the same way the unit tests guard the library: the
manifest has to describe every tensor in the weight blob, the exported scaler
statistics have to match the processed panel, and - when node is available -
the JavaScript port of the generator has to reproduce the PyTorch reference
output it ships with.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
MODEL_DIR = DOCS / "model"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "generator.json").exists(),
    reason="web bundle not built; run scripts/build_site.py",
)


@pytest.fixture(scope="module")
def manifest() -> dict:
    with open(MODEL_DIR / "generator.json") as f:
        return json.load(f)


def test_weight_blob_matches_manifest(manifest):
    blob = np.fromfile(MODEL_DIR / "generator.bin", dtype=np.float32)
    assert blob.size == manifest["param_count"]
    covered = np.zeros(blob.size, dtype=bool)
    for name, spec in manifest["tensors"].items():
        assert spec["size"] == int(np.prod(spec["shape"])), name
        end = spec["offset"] + spec["size"]
        assert end <= blob.size, name
        assert not covered[spec["offset"]:end].any(), f"{name} overlaps another tensor"
        covered[spec["offset"]:end] = True
    assert covered.all(), "weight blob has bytes no tensor claims"
    assert np.isfinite(blob).all()


def test_manifest_describes_the_trained_generator(manifest):
    from gafs.training.trainer import build_models

    arch = manifest["arch"]
    meta = dict(arch)
    meta.update({
        "dropout": 0.1,
        "critic_arch": "resnet",
        "critic_channels": [64, 128, 128],
        "critic_ctx_channels": 16,
        "proj_dim": manifest["arch"].get("proj_dim", 32),
    })
    generator, _ = build_models(meta)
    expected = {k: tuple(v.shape) for k, v in generator.state_dict().items()}
    shipped = {k: tuple(v["shape"]) for k, v in manifest["tensors"].items()}
    assert shipped == expected

    cols = manifest["columns"]
    assert len(cols["features"]) == arch["n_features"]
    assert len(cols["assets"]) == arch["n_assets"]
    assert len(cols["macro"]) == arch["cond_dim"]
    assert cols["features"] == cols["assets"] + cols["macro"]


def test_exported_scalers_match_the_processed_panel(manifest):
    from gafs.data.preprocess import ProcessedData

    processed_dir = ROOT / "outputs" / "demo" / "processed"
    if not (processed_dir / "meta.json").exists():
        pytest.skip("processed panel not present in this checkout")
    processed = ProcessedData.load(processed_dir)
    pre = manifest["preprocess"]
    assert pre["scaling"] == processed.meta["scaling"]
    assert pre["vol_window"] == processed.meta["vol_window"]
    assert pre["stationarity"] == processed.meta["stationarity"]
    for name, stats in processed.meta["macro_stats"].items():
        assert pre["macro_stats"][name]["mean"] == pytest.approx(stats["mean"])
        assert pre["macro_stats"][name]["std"] == pytest.approx(stats["std"])


def test_bundled_panels_are_listed_and_parseable():
    index = json.loads((DOCS / "data" / "index.json").read_text())
    assert index, "no panels exported to the site"
    assert any(entry["trained_on"] for entry in index), "no panel marked as the training panel"
    for entry in index:
        path = DOCS / "data" / entry["file"]
        assert path.exists(), entry["file"]
        lines = path.read_text().splitlines()
        assert len(lines) - 1 == entry["rows"]
        header = lines[0].split(",")[1:]
        assert [c for c in header if c.endswith("_close")] == entry["assets"]
        assert [c for c in header if not c.endswith("_close")] == entry["macro"]


def test_parity_sample_is_well_formed():
    parity = json.loads((MODEL_DIR / "parity.json").read_text())
    shapes = parity["shapes"]
    assert len(parity["x_hist"]) == shapes["x_hist"][0] * shapes["x_hist"][1]
    assert len(parity["z"]) == shapes["z"][0] * shapes["z"][1]
    assert len(parity["y"]) == shapes["y"][0] * shapes["y"][1]
    assert all(np.isfinite(parity["y"]))


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_generator_matches_pytorch(tmp_path):
    """Run the shipped JS forward pass and compare with the PyTorch reference."""
    script = tmp_path / "parity.js"
    script.write_text(
        "const fs = require('fs');\n"
        f"const docs = {json.dumps(str(DOCS))};\n"
        "require(docs + '/assets/js/nn.js');\n"
        "require(docs + '/assets/js/generator.js');\n"
        "const manifest = JSON.parse(fs.readFileSync(docs + '/model/generator.json'));\n"
        "const bin = fs.readFileSync(docs + '/model/generator.bin');\n"
        "const buf = bin.buffer.slice(bin.byteOffset, bin.byteOffset + bin.byteLength);\n"
        "const parity = JSON.parse(fs.readFileSync(docs + '/model/parity.json'));\n"
        "const model = globalThis.Generator.build(manifest, buf);\n"
        "const res = globalThis.Generator.verifyParity(model, parity);\n"
        "console.log(JSON.stringify(res));\n"
    )
    out = subprocess.run([shutil.which("node"), str(script)],
                         capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr
    res = json.loads(out.stdout.strip().splitlines()[-1])
    assert res["n"] > 0
    assert res["maxAbs"] < 1e-4, f"JS port drifted from PyTorch: {res}"


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_javascript_pipeline_matches_python(tmp_path):
    """The in-browser preprocessing must reproduce the Python feature matrix."""
    from gafs.config import load_config
    from gafs.data.preprocess import preprocess_market
    from scripts.preprocess_data import load_panel_csv

    panel_path = ROOT / "data" / "market" / "reference_panel.csv"
    if not panel_path.exists():
        pytest.skip("reference panel not generated")
    cfg = load_config(ROOT / "config" / "default.yaml")
    prices, macro = load_panel_csv(panel_path)
    processed = preprocess_market(prices, macro, cfg.preprocess)
    expected = processed.features.to_numpy(dtype=np.float32)

    dump = tmp_path / "features.json"
    script = tmp_path / "pipeline.js"
    script.write_text(
        "const fs = require('fs');\n"
        f"const docs = {json.dumps(str(DOCS))};\n"
        "require(docs + '/assets/js/nn.js');\n"
        "require(docs + '/assets/js/pipeline.js');\n"
        "const spec = JSON.parse(fs.readFileSync(docs + '/model/generator.json'));\n"
        f"const csv = fs.readFileSync({json.dumps(str(panel_path))}, 'utf8');\n"
        "const built = globalThis.Pipeline.buildFeatures(globalThis.Pipeline.parseCSV(csv), spec, {});\n"
        f"fs.writeFileSync({json.dumps(str(dump))}, JSON.stringify("
        "{rows: built.nRows, f: built.nFeatures, data: Array.from(built.features)}));\n"
    )
    out = subprocess.run([shutil.which("node"), str(script)],
                         capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr
    got = json.loads(dump.read_text())
    assert got["rows"] == expected.shape[0]
    assert got["f"] == expected.shape[1]
    arr = np.array(got["data"], dtype=np.float32).reshape(expected.shape)
    assert np.abs(arr - expected).max() < 1e-5
