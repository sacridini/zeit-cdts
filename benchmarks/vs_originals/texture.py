"""zeit.texture against scikit-image's graycomatrix + graycoprops, the window of each pixel in turn.

    python benchmarks/vs_originals/texture.py

Measured 2026-10-31 (Windows 11, 20 logical processors): a 512 x 512 image, 7 x 7 windows, 32 levels, distance 1, four
angles, eight properties: zeit 1.44 s on one thread (0.43 s on four); scikit-image 0.26 429 us a pixel, 113 s
(78 times slower on one thread). The properties agree to 1e-12 (tests/test_texture.py).
"""

import time

import numpy as np
from skimage.feature import graycomatrix, graycoprops

import zeit

SIDE, WINDOW, LEVELS = 512, 7, 32
PROPS = ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation", "mean", "variance"]
OURS = ["contrast", "dissimilarity", "homogeneity", "asm", "energy", "correlation", "glcm_mean", "glcm_variance"]
ANGLES = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]


def main():
    rng = np.random.default_rng(0)
    img = rng.integers(0, LEVELS, (SIDE, SIDE)).astype(float)
    for n_jobs in (1, 4):
        t0 = time.perf_counter()
        z = zeit.texture(img, features=OURS, window=WINDOW, levels=LEVELS, lo=0, hi=LEVELS, n_jobs=n_jobs)
        print(f"zeit, {n_jobs} thread(s): {time.perf_counter() - t0:.2f} s")
    # scikit-image on a sample of pixels (the whole image takes minutes), checked against zeit
    u, h, n = img.astype(np.uint8), WINDOW // 2, 2000
    t0 = time.perf_counter()
    worst = 0.0
    for k in range(n):
        y, x = h + k % (SIDE - 2 * h), h + (k * 7) % (SIDE - 2 * h)
        g = graycomatrix(u[y - h: y + h + 1, x - h: x + h + 1], [1], ANGLES, levels=LEVELS, symmetric=True, normed=True)
        for i, p in enumerate(PROPS):
            worst = max(worst, abs(graycoprops(g, p)[0].mean() - z.values[i, y, x]))
    per = (time.perf_counter() - t0) / n
    print(f"scikit-image: {per * 1e6:.0f} us a pixel, {per * SIDE * SIDE:.0f} s for the image; largest difference {worst:.1e}")


if __name__ == "__main__":
    main()
