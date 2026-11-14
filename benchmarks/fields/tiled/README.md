# zeit.fields_tiled against zeit.fields (phase 24g)

`lem_tiled.py` runs `zeit.fields` (package defaults, seed 0) on the whole LEM 2020 mosaic
(`zeit_data/fields/mosaics/lem_1_2020.tif`, 1334 x 1334 = 1.78 Mpx, six int16 bands, multitemporal gradient),
then `zeit.fields_tiled` with `tile=512` (3 x 3 tiles, overlap 80 = 4 x `min_distance`) and the same defaults,
then `zeit.fields` with another seed, which is the Monte Carlo's own noise. `--scan` also times other tile sizes.

Run on 2026-11-14, 8 threads, GDAL 4, Python from PowerShell, `no_power_throttling()`. The machine was not
quiet (other sessions kept it at about 30 % load), so the times are an upper bound, not a clean measurement.

## Agreement with the whole-image result

| Against `zeit.fields` seed 0 | fields | boundary F (1 px) | precision | recall | pixels in fields with IoU >= 0.9 | with IoU >= 0.5 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fields_tiled`, tile 512 | 2356 | 0.971 | 0.970 | 0.973 | 0.834 | 0.969 |
| `fields`, seed 1 (noise floor) | 2352 | 0.969 | 0.969 | 0.970 | 0.819 | 0.966 |

(the whole image has 2346 fields.) The tiled result is as close to the whole-image one as a second whole-image run
with another seed: the seams and the extended windows add nothing measurable to the Monte Carlo's own noise here.
`fields_tiled`'s polygons: 2356, one per field, as many as labels.

## Time

| tile | tiles | windows (Mpx) | seconds | s per window Mpx |
| ---: | :--- | ---: | ---: | ---: |
| 256 | 6 x 6 | 4.44 | 20.6 | 4.64 |
| 512 | 3 x 3 | 2.74 | 14.5 | 5.30 |
| 1024 | 2 x 2 | 2.23 | 16.0 | 7.17 |
| 2048 | 1 x 1 | 1.78 | 17.6 | 9.89 |

`zeit.fields` on the whole window: 17 to 18 s (about 10 s/Mpx); `fields_tiled` with tile 512, polygons included:
15 to 16 s. The cost is the Monte Carlo on the extended windows (nearly all of it), and a watershed is faster per
pixel on a small window (cache), so smaller tiles pay for their larger overlap share only up to a point: 512 is the
best of these.

**Extrapolation to 1.2e9 pixels (one year of the Gran Chaco), 8 threads, defaults (R = 200, 1000 markers per Mpx):**
in the middle of a large raster each tile's window is (1 + 2 x 80 / 512)^2 = 1.72 times its core, so tile 512 at
this run's 9.1 s/Mpx (polygons included) is about **3.4 hours**. The same scaling for tile 2048 (9.9 s per window Mpx
x 1.17) gives about 3.9 hours. It is a straight-line extrapolation from one 1.8 Mpx window of one region: a
different landscape (more or fewer fields, more NoData) changes the watershed's cost, the polygons of a whole
Chaco take longer than here (1 s per 3 Mpx), and the read of the raster is not in it (the window is in the page
cache here).

## Design measured

- Tiles in parallel (one Rust thread each) against tiles one after the other with every thread in the Rust
  call: for the 3 x 3 tiles, 75 s against 15 s. With 9 tiles on 8 workers the last tile ran alone, and the single
  threaded Rust calls did not scale as the multi-threaded one does. Pass 1 (the Monte Carlo) therefore runs the
  tiles one after the other with `n_jobs` threads, reading the next tile's window and making its gradient in a second
  thread meanwhile; pass 2 (hysteresis, labelling, the final watershed: single-threaded, 4 s in all here) runs
  `min(n_jobs, tiles)` tiles at a time.
