"""Smoke test of notebook_reference on a tiny synthetic image (seconds, one thread)."""
import os
import tempfile

import warnings

import numpy as np

import notebook_reference as nr

warnings.simplefilter("once")

rng = np.random.RandomState(1)
img = np.full((64, 64), 3000, dtype=np.uint16)
img[:, 30:33] = 1500  # a dark line, like a field boundary
img[30:33, :] = 1500
img = (img + rng.randint(0, 200, img.shape)).astype(np.uint16)
img[:8, :8] = 1000  # "forest" corner, masked out

# 64 px is too small for the notebook's 1000 markers at 20 px, so thin less.
sets = nr.draw_markers(img.shape, 5, 300, 4, seed=0)
path = os.path.join(tempfile.mkdtemp(), "m.npz")
nr.save_markers(path, sets)
back = nr.load_markers(path)
assert all(np.array_equal(a, b) for a, b in zip(sets, back))

out = nr.pipeline(img, markers=back, low=0.2, min_size=10, return_all=True)
again = nr.pipeline(img, markers=sets, low=0.2, min_size=10)
assert np.array_equal(out["final"], again)
assert (out["final"][:8, :8] == 0).all()
print("markers per realisation:", [m.shape[1] for m in sets])
print("counts max:", out["counts"].max(), "labels:", len(np.unique(out["final"])))
print("ok")
