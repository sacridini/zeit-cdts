# Field boundaries: the stochastic-watershed notebook as a reference

Reference for IDEAS.md phase 24. The method comes from the MappingDemo notebook
(Baumann, August 2020, after Emmanuelle Gouillart's segmentation tutorial); the
author authorised porting the code directly.

## Files

- `notebook_original.py`: the 20 code cells of the exported HTML, verbatim and in
  order (`# %% [cell N]`), with the markdown as comments. Not runnable (gdal,
  baumiTools, `%matplotlib inline`, `np.int`). Cell 20 is empty in the notebook.
- `notebook_reference.py`: the same method as functions with the notebook's
  values as defaults: `stretch`, `gradient`, `trim_close_points` (verbatim),
  `random_markers`, `draw_markers`, `save_markers`/`load_markers` (npz),
  `realisation`, `boundary_frequency`, `hysteresis_regions`, `final_watershed`,
  `pipeline`. All random draws are injectable as per-realisation marker arrays
  of shape (2, k), so Zeit's port can be compared with the same markers. The
  module docstring lists the library-version-dependent behaviour.
- `tiny_check.py`: smoke test on a 64 x 64 synthetic image, 5 realisations.

## Environment

Found: `C:\Users\eduar\miniforge3\envs\zeit` (Python 3.12.15, scikit-image 0.26.0,
scipy 1.18.1, scikit-learn 1.9.1, numpy 2.5.3, pyamg 5.4.1). pyamg is required
(`notebook_reference` raises an ImportError without it). The notebook calls
`vertex_coloring(mat)` with the default `method='MIS'`, and so does the module.

## Tiny check

```
set OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
C:\Users\eduar\miniforge3\envs\zeit\python.exe benchmarks\fields\tiny_check.py
```

It prints the markers per realisation and the max count, and ends with `ok`
(with pyamg 5.4.1: 107, 97, 99, 94 and 97 markers; max count 5; 10 labels).

## Notebook steps and parameters (checked against the code)

Input: the B5 (SWIR1) medoid, raw digital numbers (a 500 x 500 subset in the notebook).

1. (cell 8) `stretch`: percentiles 2 and 98 to [0, 1] (`rescale_intensity`).
2. (cell 9) `gradient`: `black_tophat(stretched, 3)` (flat 3 x 3 square) `- 0.5 *
   stretched`, then `morphology.dilation` with the default cross footprint.
3. (cells 10, 12) per realisation (500): 1000 uniform points
   (`x *= shape[0]`, `y *= shape[1]`), thinned greedily with a KDTree to pairs
   farther apart than 20 px (`query_pairs(20)`, drops the second of each pair);
   `kneighbors_graph(points, 20)`; pyamg `vertex_coloring`; markers = colour + 1
   at `int(x), int(y)`; grey dilation with `disk(3)`;
   `watershed(gradient, markers, mask=np.where(arr <= 2000, 0, 1))`.
4. (cell 13) sum over realisations of `find_boundaries(seg, connectivity=4)`.
5. (cell 15) `apply_hysteresis_threshold(counts, 0.3 * 500, 0.6 * 500)`; label the
   complement with `connectivity=1`; `remove_small_objects(regions + 1,
   min_size=50)`; `relabel_sequential`; random label permutation.
6. (cell 17) `watershed(arr, regions)` on the original image, **no mask** (the
   mask argument is commented out), random label permutation, then `np.where(arr
   <= 2000, 0, final)`.

## Corrections found while checking IDEAS.md against the code

1. `find_boundaries(connectivity=4)` in 2-D is the full 3 x 3 square (8-connected).
2. The mask `B5 > 2000` is applied inside every marker watershed.
3. `regions + 1` gives the boundary pixels a label of their own (one network, a
   marker of the final watershed); only regions under 50 px are removed.
4. Hysteresis thresholds are strictly greater than 150 and 300 (0.3 and 0.6 of 500).
5. `trim_close_points` depends on the iteration order of `query_pairs` (a set):
   on 20 seeds of 1000 points in 500 x 500, the kept set from set order differs
   from the one with sorted pairs in every seed (about 246 kept vs 280, and 211
   with reverse order). To compare ports, inject the marker sets
   (`save_markers`/`load_markers`) instead of re-deriving the thinning.

Details below.

## Differences between IDEAS.md and the code

- Step 4: `connectivity=4` is not 4-connected. In 2-D scikit-image builds the
  structuring element with `generate_binary_structure(2, 4)`, which is the full
  3 x 3 square (8-connected, same as `connectivity=2`); mode is the default
  'thick'. A boundary pixel is any pixel whose 3 x 3 neighbourhood holds
  two labels (dilation != erosion), so boundaries are 2 px thick.
- Step 3: the mask `B5 > 2000` is applied inside every marker watershed (pixels
  with B5 <= 2000 are unlabelled, label 0, and that edge counts as a boundary in
  `find_boundaries`), not only at the end as IDEAS.md's wording suggests.
  Markers falling on masked pixels are dropped by the watershed.
- Step 5: after `regions + 1` the hysteresis (boundary) pixels become label 1,
  one connected network that acts as one marker in the final watershed; only
  regions with fewer than 50 px are removed (set to 0, so they are flooded by the
  final watershed). IDEAS.md says "rótulos do complemento, objetos com menos de
  50 px fora", which omits that the boundary network keeps a label of its own.
  Thresholds are strictly greater than 150 and 300 (counts of 500).
- Step 2 matches IDEAS.md: gradient is `black_tophat(img, 3) - 0.5 * img` on the
  stretched image, then a (cross) dilation; the top-hat size 3 is a square.
- Step 6: the final watershed runs on the original unstretched image (matches
  IDEAS.md), without mask; forest is zeroed afterwards (matches).
- Not in IDEAS.md: coordinates are (row, col) with x scaled by `shape[0]`; the
  markers are cast with `astype(int)` (truncation); the neighbour count of the
  colouring graph is 20 (`kneighbors_graph(.., 20)`, a directed graph).
- Cells 7 (non-local-means denoising, marked "DON'T DO IT") and the random-walker
  alternatives (commented out) are not part of the method and are not ported.
