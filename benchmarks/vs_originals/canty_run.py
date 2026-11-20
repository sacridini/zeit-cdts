"""Runs one of Mort Canty's scripts (CRC5Docker, MIT licence: iMad.py, iMadmap.py, radcal.py, sar_seqQ.py...) as a
black box, for the benchmarks of Zeit's change detection against them. Zeit never reads that code: the scripts are only
run and their outputs compared.

    python -I canty_run.py <script.py> <its arguments...>

The scripts come from a clone of https://github.com/mortcanty/CRC5Docker given by the environment variable
CANTY_SRC (its ``src`` folder). They are written for numpy 1 and Linux; this wrapper only adds what lets them run on
a current numpy and on Windows without touching them: the names numpy removed, the non-interactive matplotlib backend
(``radcal.py`` blocks on a window otherwise), and, on Windows, the C library ``libprov_means`` they load, which is
built from ``src/prov_means.c`` into the folder CANTY_LIB (``gcc -shared -O2 -o libprov_means.so prov_means.c``).
"""
import ctypes
import os
import runpy
import sys

SRC = os.environ["CANTY_SRC"]
LIB = os.environ.get("CANTY_LIB")
os.environ.setdefault("MPLBACKEND", "Agg")
if LIB:
    if hasattr(os, "add_dll_directory"):
        os.add_dll_directory(LIB)
    ctypes.cdll.LoadLibrary(os.path.join(LIB, "libprov_means.so"))
sys.path[:0] = [os.path.join(SRC, "scripts"), SRC, os.path.join(SRC, "auxil")]
from osgeo import gdal  # noqa: E402,F401 - before numpy, as the scripts import them

import numpy as np  # noqa: E402

for _name, _value in (("mat", np.asmatrix), ("float", float), ("int", int), ("bool", bool), ("complex", complex)):
    if not hasattr(np, _name):
        setattr(np, _name, _value)
import warnings  # noqa: E402

warnings.filterwarnings("ignore")

import atexit  # noqa: E402
import tempfile  # noqa: E402

_named = tempfile.NamedTemporaryFile


def _named_temporary_file(*args, **kwargs):
    """Windows cannot open a NamedTemporaryFile a second time (numpy.memmap does): keep it as an ordinary file that
    is deleted at exit."""
    kwargs["delete"] = False
    f = _named(*args, **kwargs)

    def clean():
        try:
            f.close()
            os.remove(f.name)
        except OSError:
            pass

    atexit.register(clean)
    return f


tempfile.NamedTemporaryFile = _named_temporary_file
script = os.path.join(SRC, "scripts", sys.argv[1])
sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
