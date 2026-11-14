# Sensitivity of the field boundaries (phase 24e)

How the parameters of `zeit.fields` (the stochastic watershed of the MappingDemo notebook) move the result, on 64
cases: 62 Landsat mosaic windows (Chaco and Humid Chaco with the Trazo annotations, Luis Eduardo Magalhaes with
Fields of The World, Sorriso with CAR and pivots, Porto Murtinho with CAR). Everything runs on the data in
`ZEIT_FIELDS_DATA` (default `C:\Users\eduar\Documents\dev\zeit_data\fields`); the outputs go to its `sensitivity/`
folder, not to git.

Run with the `zeit` environment's Python **from PowerShell** (Git Bash breaks `pyexpat`/rasterio), at most 8 threads, one
heavy process at a time:

```powershell
$me = Get-Process -Id $PID; $me.ProcessorAffinity = [IntPtr]0xFF
$env:OMP_NUM_THREADS=1; $env:OPENBLAS_NUM_THREADS=1; $env:MKL_NUM_THREADS=1; $env:RAYON_NUM_THREADS=8; $env:GDAL_NUM_THREADS=4
cd benchmarks\fields\sensitivity
```

## Scripts, in order

| script | what | command | cost |
|---|---|---|---|
| `cases.py`, `pipeline.py`, `matched.py` | the cases, stage 1 (boundary map, cached) and stage 2 (hysteresis, `min_size`, final watershed) as `zeit.fields` does them, the responses, the corrected `matched` variants | (modules) | |
| `test_pipeline.py` | stage 1 + 2 equal `zeit.fields`; `matched` without clipping equals the package's | `python -m pytest benchmarks/fields/sensitivity/test_pipeline.py -q -p no:cacheprovider` (from the repo root) | 10 s |
| `run.py` | 24e task 2: `convergence` (R 25..2000 x 3 seeds), `exact`, `exactdensity`, `oat` (one factor at a time, R=200), `masks` | `python run.py convergence` ... | about 3 h in all (R=2000 dominates) |
| `diagnose.py` | why `matched` is low (8 cases from the cache): `matched_diagnosis.csv/.md` | `python diagnose.py` | 1 min |
| `morris.py` | Morris elementary effects, r trajectories x 9 points x 64 cases, R=200, resumable | `python morris.py run --r 20` | 3.3 min per trajectory (66 min for 20), 8 threads |
| `cv.py` | threshold choice by spatial cross-validation: `run` makes 10 stage-1 maps per case (2 gradients x 5 masks) and the 36-setting stage-2 grid on each; `analyze` the folds | `python cv.py run`, `python cv.py analyze` | `run` about 25 min; `analyze` 1 min |
| `summary.py` | `tidy` (results.csv), `maps`, `tables` (report.md, task 2), `report24e` (report_24e.md, Morris tables) | `python summary.py report24e` | 2 min |

`probe.py` times one stage 1 and one stage 2 per case.

## Outputs (in `sensitivity/`)

* `results.jsonl`: one row per case x parameter set x seed, appended by every study (`study` = `morris` / `cv`, the
  rest from task 2), and read back by the resumable runners (a finished row is not recomputed);
* `cache/<case>/*.npz`: the stage-1 maps (Monte Carlo counts, or the exact probability), keyed by the stage-1 parameters;
* `report.md` (task 2, long), `report_24e.md` (short: convergence, one factor at a time, `matched`, Morris, spatial CV,
  recommended defaults), `morris_effects.csv`, `cv_results.csv`, `cv_long.csv`, `matched_diagnosis.csv`;
* `plots/`: `convergence.png`, `oat_f.png`, `morris_mu_sigma.png`, `cv_heldout.png`.
* `recommend.md` and `matched_conclusion.md` are the hand-written text of sections 3 and 6 of `report_24e.md`.

## Method notes

* Morris: 8 factors in the unit cube on a grid of 6 levels (delta 0.6), r trajectories drawn from `default_rng(24500 + i)`; no
  SALib in the environment, the design is a few lines in `morris.design`. The Monte Carlo seed is the trajectory's index (one
  seed along a trajectory). The markers are a density (markers per Mpx x the window's pixels), the relative mask is
  `swir16 > the window's q-th percentile`.
* `matched_clip` / `matched_in` (`matched.py`): `segmentation_accuracy` counts predicted objects in full and the
  reference only inside the region; the variants clip the predicted objects to the region, and count only the reference
  fields fully inside the region and away from the window's edge.
* Cross-validation (`cv.py`): groups, not windows, are the unit of selection in leave-one-group-out; within Chaco the 45 windows
  are folded by their centroids with `zeit.spatial_folds` (k-means clusters and hexagonal blocks, k = 5) and, for
  contrast, at random.
