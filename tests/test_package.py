"""The top-level package: fast to import, heavy parts loaded on first use."""
import importlib.util
import subprocess
import sys

import pytest


def _run(code):
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def test_import_does_not_load_heavy_dependencies():
    out = _run("import sys, zeit; print(sorted(m for m in ('torch', 'transformers', 'stackstac', 'pystac_client', "
               "'sklearn') if m in sys.modules))")
    assert out == "[]"


def test_zeit_ai_without_torch_says_how_to_install_it():
    out = _run("import sys\n"
               "class NoTorch:\n"
               "    def find_spec(self, name, path=None, target=None):\n"
               "        if name.split('.')[0] == 'torch':\n"
               "            raise ImportError(name)\n"
               "sys.meta_path.insert(0, NoTorch())\n"
               "import zeit\ntry:\n    zeit.ai\nexcept ImportError as e:\n    print(e)")
    assert 'pip install "zeit-cdts[ai]"' in out


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="zeit.ai needs torch")
def test_lazy_names_load_on_first_use():
    out = _run("import zeit; from zeit import ai; from zeit.ai import SOM; "
               "print(zeit.build_time_series.__module__, zeit.classify.__module__, "
               "zeit.ai.SOM is SOM, 'ai' in dir(zeit))")
    assert out == "zeit.cube zeit._classify_api True True"


def test_unknown_attribute_raises():
    out = _run("import zeit\ntry:\n    zeit.no_such_thing\nexcept AttributeError as e:\n    print(e)")
    assert "no_such_thing" in out
