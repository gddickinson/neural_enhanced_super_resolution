"""Regression tests for SuperResolutionPipeline configuration (no models are loaded: _init_models is lazy)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nesr.nesr import SuperResolutionPipeline  # noqa: E402


def test_default_config_none(tmp_path, monkeypatch):
    """SuperResolutionPipeline(device='cpu') with no config used to raise
    TypeError: argument of type 'NoneType' is not iterable (from `key not in config`)."""
    monkeypatch.chdir(tmp_path)
    p = SuperResolutionPipeline(device="cpu")
    assert p.device == "cpu"
    assert p.config["max_megapixels"] == 2  # the cpu device-specific default is applied
    assert (tmp_path / "outputs").is_dir()


def test_user_config_overrides_device_default(tmp_path):
    p = SuperResolutionPipeline(device="cpu", config={"output_dir": str(tmp_path), "max_megapixels": 7})
    assert p.config["max_megapixels"] == 7
