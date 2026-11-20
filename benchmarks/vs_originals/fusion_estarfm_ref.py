"""Runs Xiaolin Zhu's ESTARFM Python program as a black box, unattended, for the comparison with zeit.fuse_estarfm.

Zhu's code has no licence (copyright Xiaolin Zhu): this script never reads it, copies it or ships it. It only
answers the program's pop-up dialogs from outside (replacing ``tkinter.filedialog``'s functions before the program
runs) and runs the program's entry file with ``runpy``. The program asks, in this order, for the parameter file, the
temporary folder, fine 1, coarse 1, fine 2, coarse 2 and the coarse image of the prediction date, and writes
``<prediction coarse name>_ESTARFM`` next to the inputs. Run it in an environment with gdal, numpy, pyyaml,
statsmodels, tk and idlwrap that is not zeit's (conda env ``estarfm-ref``).

    python fusion_estarfm_ref.py PROGRAM_DIR PARAMS.yaml TEMP_DIR FINE1 COARSE1 FINE2 COARSE2 COARSE_PRED
"""
import os
import runpy
import sys
import tkinter
import tkinter.filedialog

import numpy as np
from osgeo import gdal

sys.modules.setdefault("gdal", gdal)      # the program imports the pre-2.x top-level name
for _alias, _type in (("int", int), ("float", float), ("bool", bool), ("object", object), ("complex", complex)):
    if not hasattr(np, _alias):           # aliases removed in numpy 1.24 that the program still uses
        setattr(np, _alias, _type)


def main(argv):
    program, *answers = argv
    answers = [os.path.abspath(a) for a in answers]
    queue = list(answers)

    def ask(*args, **kwargs):
        return queue.pop(0)

    class Quiet:
        def __init__(self, *args, **kwargs):
            pass

        def __getattr__(self, name):
            return lambda *a, **k: None

    tkinter.filedialog.askopenfilename = ask
    tkinter.filedialog.askdirectory = ask
    tkinter.Tk = Quiet
    entry = os.path.join(program, "ESTARFM.py")
    sys.path.insert(0, program)
    os.chdir(program)
    runpy.run_path(entry, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv[1:])
