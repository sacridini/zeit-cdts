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

## Phase 24f: the candidate defaults

Outputs in `sensitivity/f24/`, the report in `sensitivity/report_24f.md`. All from PowerShell with the affinity and threads
above (stage 1 on 8 threads, the cheap stage 2 on a pool of 6 Python threads; one process at a time).

| script | what | command | cost |
|---|---|---|---|
| `fast.py`, `s24f.py` | the cheap responses (F, recall, D_w, crossing share; equal to `pipeline.evaluate`), the density / relative mask / hierarchy cut / threaded stage 2 | (modules) | |
| `test_24f.py` | fast = `pipeline.evaluate`, the threaded stage 2 = `pipeline.stage2`, the cut is a partition | `python -m pytest benchmarks/fields/sensitivity/test_24f.py -q -p no:cacheprovider` | 6 s |
| `a24f.py` | task A: 64 stage-1 configurations (gradient x density x mask) x 42 stage-2 settings; `lows` adds low 0.02 and 0.05 | `python a24f.py run`, `python a24f.py lows` | 53 min, 10 min |
| `cv24f.py` | the leave-one-group-out and Chaco folds on A (`A`, `Aext`) or C (`C`): rules mean, minimax regret, mean rank | `python cv24f.py A` | 10 s |
| `c24f.py` | task C: the hierarchy cut on the exact edge probabilities and on the Monte Carlo map | `python c24f.py run --grads sobel,multimean` | 25 min |
| `b24f.py` | task B: noise added to the gradient in every realisation (`check` that s = 0 equals the normal counts) | `python b24f.py check --cfg proposed`, `python b24f.py run --cfg proposed` | 3 min per s |
| `extras.py`, `extras24f.py` | the package's `clip` / `complete_only` metrics at the selected configurations | `python extras24f.py A` | 2 min |
| `report24f.py` | the tables of B and of the extras, and the assembly of `report_24f.md` | `python report24f.py pieces`, `python report24f.py report` | seconds |

`pipeline.gradient_of` gained `multimean` (the multitemporal gradient combined by the mean).


## Phase 24f, task 5: the Fields of The World predictions

`ftw.py` scores the FTW global predicted polygons (`ftw_predictions.gpkg`, 2024 and 2025, all and confidence >= 69, with the 0
background and with it filled by the nearest label) with the harness's evaluator, next to `zeit.fields` at the new defaults, with
the notebook's forest mask (Chaco groups) and the notebook preset. Outputs in `sensitivity/ftw/ftw_rows.csv` and
`sensitivity/report_ftw.md`.

| command | what | cost |
|---|---|---|
| `python ftw.py check` | the harness's `new` / `new_mask` / `notebook` equal a direct `zeit.fields` call, label for label (2 cases) | 1 min |
| `python ftw.py run` | every case x source -> `ftw/ftw_rows.csv` (stage-1 maps from the cache) | 1 min |
| `python ftw.py report` | `report_ftw.md` | seconds |
