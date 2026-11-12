"""Regression test: the package and its image utilities import without PyTorch installed.

nesr/utils/downloader.py used to `import torch` at module level without using it, so `import nesr` (and therefore
`nesr.utils.image_utils` and the unit tests) failed on a machine without torch, even though the pipeline itself
falls back to CPU / bicubic when torch is missing.
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CODE = """
import sys
sys.modules['torch'] = None  # makes `import torch` raise ImportError, as on a machine without torch
sys.path.insert(0, {root!r})
import nesr
from nesr.utils.image_utils import downsample_image
from nesr.utils import download_models, check_models_exist
print('ok')
"""


def test_import_without_torch():
    r = subprocess.run([sys.executable, "-c", CODE.format(root=str(ROOT))], capture_output=True, text=True)
    assert r.returncode == 0 and r.stdout.strip().endswith("ok"), r.stderr[-2000:]
