"""Regression tests: the SegFormer segmentation model is loaded with the semantic-segmentation classes, and a failure
to load it is reported (model_errors + a warning) instead of disappearing into a log line.

The pipeline used AutoModelForImageSegmentation, which has no mapping for SegformerConfig, so segmentation-guided
enhancement could never load and was switched off silently. transformers is replaced by a stand-in module here.
"""
import importlib.machinery
import logging
import os
import sys
import types

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nesr.nesr import SuperResolutionPipeline  # noqa: E402

MODEL_ID = "nvidia/segformer-b0-finetuned-ade-512-512"


def _fake_transformers(monkeypatch, calls, fail=None):
    class FakeModel:
        def to(self, device):
            calls.append(("to", device))
            return self

        def eval(self):
            calls.append(("eval",))
            return self

    class Semantic:
        @staticmethod
        def from_pretrained(name):
            calls.append(("semantic", name))
            if fail:
                raise fail
            return FakeModel()

    class Processor:
        @staticmethod
        def from_pretrained(name):
            calls.append(("processor", name))
            return object()

    class Generic:  # the class that cannot load SegFormer
        @staticmethod
        def from_pretrained(name):
            raise ValueError("Unrecognized configuration class SegformerConfig for AutoModelForImageSegmentation")

    m = types.ModuleType("transformers")
    m.__spec__ = importlib.machinery.ModuleSpec("transformers", None)
    m.AutoModelForSemanticSegmentation, m.AutoImageProcessor = Semantic, Processor
    m.AutoModelForImageSegmentation, m.AutoFeatureExtractor = Generic, Processor
    monkeypatch.setitem(sys.modules, "transformers", m)


def _pipeline(tmp_path):
    return SuperResolutionPipeline(device="cpu", config={"use_esrgan": False, "use_diffusion": False,
                                                         "segment_enhancement": True, "output_dir": str(tmp_path)})


def test_segformer_loaded_as_semantic_segmentation(tmp_path, monkeypatch):
    calls = []
    _fake_transformers(monkeypatch, calls)
    p = _pipeline(tmp_path)
    p._load_models()
    assert ("semantic", MODEL_ID) in calls and ("processor", MODEL_ID) in calls and ("eval",) in calls
    assert "segmentation" in p.models and "segmentation_extractor" in p.models
    assert p.config["segment_enhancement"] and p.model_errors == {}


def test_load_failure_is_reported(tmp_path, monkeypatch, caplog):
    _fake_transformers(monkeypatch, [], fail=OSError("no network"))
    p = _pipeline(tmp_path)
    with caplog.at_level(logging.WARNING):
        p._load_models()
    assert "no network" in p.model_errors["segmentation"]
    assert not p.config["segment_enhancement"]
    assert any("Segmentation-guided enhancement disabled" in r.message for r in caplog.records)


def test_enhance_reports_unavailable_models(tmp_path, monkeypatch):
    import cv2
    _fake_transformers(monkeypatch, [], fail=OSError("no network"))
    p = _pipeline(tmp_path)
    p.config["iterations"] = 1
    seen = []
    p.config["progress_callback"] = lambda stage, i, n, msg=None: seen.append((stage, msg))
    src = tmp_path / "in.png"
    cv2.imwrite(str(src), np.full((20, 20, 3), 128, np.uint8))
    p.enhance_image(str(src))
    assert any(stage == "Models" and "segmentation unavailable" in msg for stage, msg in seen)
