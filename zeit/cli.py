import argparse
import sys
from typing import Optional, List

from .spatial import apply_mmu_filter

def run_landtrendr(args: argparse.Namespace) -> None:
    import xarray as xr
    from ._load import load_raster
    from ._lt import landtrendr
    from ._save import save_raster
    from .metrics import extract_events

    try:
        # Lazy, block by block: the outputs are computed while they are written.
        cube = load_raster(args.input, start_year=args.start_year,
                           chunks={"time": -1, "y": args.chunk_size, "x": args.chunk_size})
        lt = landtrendr(
            cube,
            direction=args.event_type,
            max_segments=args.max_segments,
            recovery_threshold=args.recovery_threshold,
            prevent_fast_recovery=not args.allow_fast_recovery,
            spike_threshold=args.spike_threshold,
            best_model_proportion=args.best_model_proportion,
            vertex_count_overshoot=args.vertex_count_overshoot,
            min_observations_needed=args.min_observations_needed,
            nodata=args.no_data_value if args.no_data_value is not None else "auto",
            n_jobs=args.jobs,
        )
        events = extract_events(lt, event_type=args.event_type, sort_by=args.sort_by, min_magnitude=args.min_mag,
                                min_duration=args.min_dur, pre_val_threshold=args.pre_val_thresh)
        if args.output_scale != 1.0:
            for name in ("magnitude", "pre_val", "post_val", "rate"):
                events[name] = events[name] * args.output_scale
        layers = {f"{args.prefix}_{name}": events[name] for name in events.data_vars}
        if args.save_vertices:
            values = lt.vertex_value * args.output_scale if args.output_scale != 1.0 else lt.vertex_value
            layers["lt_vertices"] = xr.concat([lt.vertex_year.astype("float32"), values.fillna(0)], dim="vertex")
        # One call: every output is computed in the same pass over the image.
        save_raster(layers, args.output_dir, nodata=0)
        print(f"Successfully processed and saved layers to {args.output_dir}")
    except Exception as e:
        print(f"Error running LandTrendr: {e}")
        sys.exit(1)

def run_ccdc_cli(args: argparse.Namespace) -> None:
    import os
    from ._ccdc_api import ccdc
    from ._load import load_raster
    from ._save import save_raster

    try:
        dates = None
        if args.dates_file:
            # One date per line: an ISO date (2020-01-15) or a Python ordinal day (737439).
            with open(args.dates_file) as f:
                lines = [line.strip() for line in f if line.strip()]
            dates = [int(v) if v.isdigit() and len(v) > 4 else v for v in lines]
            dates = [_timestamp(v) for v in dates]
        cube = load_raster(args.input, chunks={"time": -1, "band": -1, "y": args.chunk_size, "x": args.chunk_size})
        bands, qa = None, None
        if "time" in cube.dims:          # dates in the band names (date_band)
            if args.qa_band >= 0:
                qa = str(cube.band.values[args.qa_band])
        else:                            # interleaved by date: --dates-file and --num-bands
            bands = [f"b{i + 1}" for i in range(args.num_bands)]
            if args.qa_band >= 0:
                qa = bands[args.qa_band]
        segments = ccdc(cube, qa=qa, dates=dates, bands=bands, max_segments=args.max_segments,
                        conseq_anom=6 if args.cold else args.conse, n_jobs=args.jobs)
        os.makedirs(args.output_dir, exist_ok=True)
        # One call: every output is computed in the same pass over the image.
        save_raster({f"{args.prefix}_{name}": segments[name] for name in segments.data_vars}, args.output_dir)
        print(f"Successfully processed and saved layers to {args.output_dir}")
    except Exception as e:
        print(f"Error running CCDC: {e}")
        sys.exit(1)


def _timestamp(value):
    import pandas as pd
    return pd.Timestamp.fromordinal(value) if isinstance(value, int) else pd.Timestamp(value)

def _run_series_cli(args: argparse.Namespace, name: str, **kwargs) -> None:
    """Shared body of the bfast-monitor, bfast-lite, bfast and mann-kendall commands: one
    multi-band GeoTIFF, <output_dir>/<prefix>.tif, one band per metric."""
    import os
    from . import _series_api
    from ._load import load_raster
    from ._save import save_raster

    try:
        cube = load_raster(args.input, chunks={"time": -1, "y": args.chunk_size, "x": args.chunk_size})
        result = getattr(_series_api, name)(cube, min_valid=args.min_valid, n_jobs=args.jobs, **kwargs)
        out = save_raster(result.astype("float32"), os.path.join(args.output_dir, f"{args.prefix}.tif"))
        print(f"Successfully processed and saved to {out}")
    except Exception as e:
        print(f"Error running {name}: {e}")
        sys.exit(1)


def _time_value(text: Optional[str]):
    """A decimal year (2019.5) or a date (2019-07-01)."""
    if text is None:
        return None
    try:
        return float(text)
    except ValueError:
        return text


def run_bfast_monitor_cli(args: argparse.Namespace) -> None:
    _run_series_cli(args, "bfast_monitor", monitor_start=_time_value(args.monitor_start_time),
                    start_time=args.start_time, frequency=args.frequency, order=args.order, h=args.h,
                    period=args.period, alpha=args.alpha)


def run_bfast_lite_cli(args: argparse.Namespace) -> None:
    _run_series_cli(args, "bfast_lite", start_time=args.start_time, frequency=args.frequency, order=args.order,
                    h=args.h, max_breaks=args.max_breaks_output)


def run_bfast_cli(args: argparse.Namespace) -> None:
    _run_series_cli(args, "bfast", start_time=args.start_time, frequency=args.frequency, order=args.order,
                    h=args.h, max_breaks_trend=args.max_breaks_trend, max_breaks_season=args.max_breaks_season,
                    max_iter=args.max_iter, level=args.level)


def run_mann_kendall_cli(args: argparse.Namespace) -> None:
    _run_series_cli(args, "mann_kendall", method=args.method, alpha=args.alpha, lag=args.lag, period=args.period)


def run_mmu_filter_cli(args: argparse.Namespace) -> None:
    try:
        from ._load import load_raster
        from ._save import save_raster

        save_raster(apply_mmu_filter(load_raster(args.input), mmu_pixels=args.mmu_pixels), args.output)
        print(f"MMU filtering applied. Saved to {args.output}")
    except Exception as e:
        print(f"Error running MMU filter: {e}")
        sys.exit(1)

def _run_cube_cli(args: argparse.Namespace, name: str, compute, *, lazy: bool = True, extras=None) -> None:
    """Shared body of the commands of the cube functions (phenology, smooth, tmask, twdtw,
    snic, classify, som): read the input (lazily, in blocks of --chunk-size), compute, and
    write <output_dir>/<prefix>.tif in one pass, one band per map or date. ``extras(result,
    stem)`` writes what is not a raster (polygons, prototypes) next to it and returns the
    paths."""
    import os
    import xarray as xr
    from ._load import load_raster
    from ._save import save_raster

    try:
        chunks = {"time": -1, "band": -1, "y": args.chunk_size, "x": args.chunk_size} if lazy else None
        result = compute(load_raster(args.input, chunks=chunks))
        os.makedirs(args.output_dir, exist_ok=True)
        stem = os.path.join(args.output_dir, args.prefix)
        maps = result
        if isinstance(result, xr.Dataset):
            maps = result[[v for v in result.data_vars if {"y", "x"} <= set(result[v].dims)]]
        elif result.dtype == bool:
            maps = result.astype("uint8")
        out = save_raster(maps, stem + ".tif")
        print(f"Successfully processed and saved to {out}")
        for path in (extras(result, stem) if extras is not None else []):
            print(f"Saved {path}")
    except Exception as e:
        print(f"Error running {name}: {e}")
        sys.exit(1)


def _weights(path: Optional[str]):
    """--weights: a raster of observation weights in [0, 1] on the grid and dates of the input."""
    if path is None:
        return None
    from ._load import load_raster

    return load_raster(path, masked=True, chunks="auto")


def run_phenology_cli(args: argparse.Namespace) -> None:
    from ._series_api import phenology

    _run_cube_cli(args, "phenology", lambda cube: phenology(
        cube, curve=args.curve, method=args.method, weights=_weights(args.weights), annual=not args.not_annual,
        max_seasons=args.max_seasons, whittaker_lambda=args.whittaker_lambda, n_jobs=args.jobs))


def run_smooth_cli(args: argparse.Namespace) -> None:
    from ._smooth import smooth

    _run_cube_cli(args, "smooth", lambda cube: smooth(
        cube, method=args.method, lmbda=args.lmbda, weights=_weights(args.weights), window=args.window,
        polyorder=args.polyorder, n_jobs=args.jobs))


def run_tmask_cli(args: argparse.Namespace) -> None:
    from ._tmask_api import tmask

    _run_cube_cli(args, "tmask", lambda cube: tmask(cube, green=args.green, swir=args.swir, scale=args.scale))


def read_patterns(path: str) -> dict:
    """TWDTW patterns from a CSV: columns ``pattern``, ``date`` and one value column (one
    band) or one column per band, named as the bands of the input."""
    import pandas as pd

    table = pd.read_csv(path)
    missing = [c for c in ("pattern", "date") if c not in table.columns]
    if missing:
        raise ValueError(f"{path}: the patterns need the columns pattern and date (missing {missing})")
    values = [c for c in table.columns if c not in ("pattern", "date")]
    if not values:
        raise ValueError(f"{path}: no value column next to pattern and date")
    table["date"] = pd.to_datetime(table["date"])
    patterns = {}
    for name, rows in table.groupby("pattern", sort=False):
        rows = rows.set_index("date").sort_index()[values]
        patterns[str(name)] = rows[values[0]] if len(values) == 1 else rows
    return patterns


def run_twdtw_cli(args: argparse.Namespace) -> None:
    from ._twdtw_api import twdtw

    patterns = read_patterns(args.patterns)
    _run_cube_cli(args, "twdtw", lambda cube: twdtw(
        cube, patterns, band=args.band, steepness=args.steepness, midpoint=args.midpoint,
        cycle=None if args.no_cycle else "year", max_elapsed=args.max_elapsed, n_jobs=args.jobs))


def run_snic_cli(args: argparse.Namespace) -> None:
    from ._snic_api import snic
    from .segmentation import snic_to_polygons

    def polygons(result, stem):
        if not args.polygons:
            return []
        snic_to_polygons(result, include_means=True).to_file(stem + ".gpkg")
        return [stem + ".gpkg"]

    _run_cube_cli(args, "snic", lambda cube: snic(
        cube, spacing=args.spacing, compactness=args.compactness, grid=args.grid, n_jobs=args.jobs),
        lazy=False, extras=polygons)


def run_classify_cli(args: argparse.Namespace) -> None:
    from ._classify_api import classify, train_classifier

    def compute(cube):
        import joblib

        if args.model is not None and args.samples is None:
            model = joblib.load(args.model)
        elif args.samples is not None:
            model = train_classifier(cube, args.samples, label=args.label)
            if args.model is not None:
                joblib.dump(model, args.model)
                print(f"Saved the model to {args.model}")
        else:
            raise ValueError("give --samples (to train) or --model (a model saved by an earlier run)")
        return classify(cube, model, probability=args.probability)

    _run_cube_cli(args, "classify", compute)


def run_som_cli(args: argparse.Namespace) -> None:
    from ._som_api import som

    def prototypes(result, stem):
        import pandas as pd
        from ._snic_api import _feature_names

        protos = result.prototypes
        dims = [d for d in protos.dims if d != "neuron"]
        names = _feature_names(protos, dims) if dims != ["feature"] else [str(f) for f in protos.feature.values]
        table = pd.DataFrame(protos.values.reshape(protos.sizes["neuron"], -1), columns=names)
        table.insert(0, "n_pixels", result.n_pixels.values)
        table.insert(0, "j", result.j.values)
        table.insert(0, "i", result.i.values)
        table.insert(0, "neuron", result.neuron.values)
        table.to_csv(stem + "_prototypes.csv", index=False)
        return [stem + "_prototypes.csv"]

    _run_cube_cli(args, "som", lambda cube: som(
        cube, x=args.x, y=args.y, sample=args.sample if args.sample > 0 else None, algorithm=args.algorithm,
        sigma=args.sigma, learning_rate=args.learning_rate, seed=args.seed, n_jobs=args.jobs),
        extras=prototypes)


def run_coded_cli(args: argparse.Namespace) -> None:
    from ._coded import coded

    start = args.start
    if start is not None:
        try:
            start = float(start)
        except ValueError:
            pass   # a date
    forest_label = args.forest_label
    if forest_label is not None and forest_label.lstrip("-").isdigit():
        forest_label = int(forest_label)
    _run_cube_cli(args, "coded", lambda cube: coded(
        cube, start=start, train_years=args.train_years, consec=args.consec, thresh=args.thresh,
        min_years=args.min_years, max_events=args.max_events, training=args.training, label=args.label,
        forest_label=forest_label, forest_ndfi=args.forest_ndfi, n_jobs=args.jobs))


def run_embeddings_cli(args: argparse.Namespace) -> None:
    import os

    from ._embeddings import load_embeddings
    from ._save import save_raster

    if (args.bbox is None) == (args.region is None):
        raise SystemExit("give the region with --bbox WEST SOUTH EAST NORTH (longitude and latitude) or --region FILE")
    years = None
    if args.years:
        years = []
        for part in args.years:
            first, _, last = part.partition("-")
            years.extend(range(int(first), int(last or first) + 1))
    emb = load_embeddings(tuple(args.bbox) if args.bbox else args.region, source=args.source, years=years,
                          crs=args.crs, res=args.res, version=args.version, variant=args.variant, depth=args.depth,
                          backend=args.backend, store=args.store, cache_dir=args.cache_dir)
    os.makedirs(args.output_dir, exist_ok=True)
    path = save_raster(emb, os.path.join(args.output_dir, f"{args.prefix or args.source}.tif"))
    print(f"Saved {emb.sizes['time']} years x {emb.sizes['band']} dimensions "
          f"({emb.sizes['y']} x {emb.sizes['x']} cells) to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="zeit: Change Detection Python Library")
    subparsers = parser.add_subparsers(dest="command", help="Available algorithms")
    
    # LandTrendr Subparser
    lt_parser = subparsers.add_parser("landtrendr", help="Run LandTrendr algorithm")
    lt_parser.add_argument("input", help="Path to input multi-band GeoTIFF")
    lt_parser.add_argument("output_dir", help="Directory to save the outputs")
    lt_parser.add_argument("--start-year", type=int, default=None, help="Year of the first band (default: read from the band descriptions, e.g. yr1985)")
    lt_parser.add_argument("--max-segments", type=int, default=6, help="Maximum number of segments (default: 6)")
    lt_parser.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores to use (-1 for all, default: -1)")
    lt_parser.add_argument("--save-vertices", action="store_true", help="Save the raw vertices stack")
    lt_parser.add_argument("--chunk-size", type=int, default=512, help="Size of the image chunks to process at once (default: 512)")
    
    # Event Extraction options
    lt_parser.add_argument("--event-type", choices=["loss", "gain"], default="loss", help="Event type to segment for and map (default: loss)")
    lt_parser.add_argument("--sort-by", choices=["greatest", "newest", "fastest", "longest", "dsnr"], default="greatest", help="How to select the event: dsnr is magnitude standardized by the fit's RMSE, LT-GEE's disturbance signal-to-noise ratio (default: greatest)")
    lt_parser.add_argument("--min-mag", type=float, default=0.0, help="Minimum magnitude filter")
    lt_parser.add_argument("--min-dur", type=int, default=1, help="Minimum duration filter")
    lt_parser.add_argument("--pre-val-thresh", type=float, default=0.0, help="Pre-value threshold filter")
    lt_parser.add_argument("--prefix", default="lt_event", help="Prefix for output metric files")
    lt_parser.add_argument("--output-scale", type=float, default=1.0, help="Scale factor to multiply output values (e.g. 0.0001 to convert back to float NDVI)")
    lt_parser.add_argument("--recovery-threshold", type=float, default=0.25, help="Max allowed recovery rate per year, LT-GEE's recoveryThreshold (default: 0.25)")
    lt_parser.add_argument("--allow-fast-recovery", action="store_true", help="Disable the fast-recovery rejection (LT-GEE's preventOneYearRecovery=false; default here is to reject, i.e. true)")
    lt_parser.add_argument("--spike-threshold", type=float, default=0.9, help="Desawtooth dampening factor, LT-GEE's spikeThreshold (1.0 = no dampening, default: 0.9)")
    lt_parser.add_argument("--best-model-proportion", type=float, default=0.75, help="Prefer the most-vertex candidate model whose p-value is at most (2 - this) times the lowest p-value found, as in the original LandTrendr (default: 0.75, i.e. within 1.25x of the best)")
    lt_parser.add_argument("--vertex-count-overshoot", type=int, default=3, help="LT-GEE's vertexCountOvershoot: extra vertices allowed in the initial candidate pool beyond max_segments + 1, pruned back down before model selection (default: 3)")
    lt_parser.add_argument("--min-observations-needed", type=int, default=6, help="LT-GEE's minObservationsNeeded: below this many observations, skip fitting entirely and pass the raw trajectory through unsegmented (default: 6)")
    lt_parser.add_argument("--no-data-value", type=float, default=None, help="Value marking a missing observation (default: the raster's NoData, or 0 for integer stacks without one)")
    
    # CCDC Subparser
    ccdc_parser = subparsers.add_parser("ccdc", help="Run CCDC algorithm")
    ccdc_parser.add_argument("input", help="Path to input stacked multi-band GeoTIFF (bands named date_band, as save_raster writes them, or interleaved by date with --dates-file and --num-bands)")
    ccdc_parser.add_argument("output_dir", help="Directory to save the outputs")
    ccdc_parser.add_argument("--num-bands", type=int, default=6, help="With --dates-file: number of bands per date in a stack interleaved by date (default: 6)")
    ccdc_parser.add_argument("--qa-band", type=int, default=-1, help="0-based index, among the bands of each date, of an Fmask QA band (default: -1 for none)")
    ccdc_parser.add_argument("--dates-file", help="Text file with one date per line (ISO date or Python ordinal day), for a stack without dates in its band names")
    ccdc_parser.add_argument("--max-segments", type=int, default=6, help="Maximum number of segments (default: 6)")
    ccdc_parser.add_argument("--chunk-size", type=int, default=512, help="Size of the image chunks to process at once (default: 512)")
    ccdc_parser.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores to use (-1 for all, default: -1)")
    ccdc_parser.add_argument("--conse", type=int, default=6, help="Consecutive anomalous observations to flag a change (default: 6, as in the original CCDC)")
    ccdc_parser.add_argument("--cold", action="store_true", help="Deprecated: same as --conse 6 (now the default)")
    ccdc_parser.add_argument("--prefix", default="ccdc", help="Prefix for output files (default: ccdc)")

    # Shared time-series arguments for the bfast family (bfastmonitor/bfastlite/bfast) and
    # Mann-Kendall: every band in the input is one equally-spaced observation
    # (`start_time + i/frequency`, matching R's `ts`/`time()` semantics), not a real per-band date.
    def _add_timeseries_args(p):
        p.add_argument("input", help="Path to input multi-band GeoTIFF (one band per equally-spaced time step; dates read from the band names)")
        p.add_argument("output_dir", help="Directory to save the output")
        p.add_argument("--chunk-size", type=int, default=512, help="Size of the image chunks to process at once (default: 512)")
        p.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores to use (-1 for all, default: -1)")

    # bfastmonitor Subparser
    bfm_parser = subparsers.add_parser("bfast-monitor", help="Run bfastmonitor (near-real-time disturbance monitoring)")
    _add_timeseries_args(bfm_parser)
    bfm_parser.add_argument("--start-time", type=float, default=None, help="Series start time (e.g. 2015.0; default: from the band dates)")
    bfm_parser.add_argument("--monitor-start-time", required=True, help="Time monitoring begins: a decimal year (2019.0) or a date (2019-01-01)")
    bfm_parser.add_argument("--frequency", type=int, default=None, help="Observations per year (e.g. 23 for 16-day composites; default: from the band dates)")
    bfm_parser.add_argument("--order", type=int, default=3, help="Harmonic order (default: 3)")
    bfm_parser.add_argument("--h", type=float, default=0.25, choices=[0.25, 0.5, 1.0], help="MOSUM window size, as a fraction of history length (default: 0.25)")
    bfm_parser.add_argument("--period", type=int, default=10, choices=[2, 4, 6, 8, 10], help="Monitoring period parameter (default: 10)")
    bfm_parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    bfm_parser.add_argument("--min-valid", type=int, default=10, help="Minimum valid history observations per pixel (default: 10)")
    bfm_parser.add_argument("--prefix", default="bfast_monitor", help="Prefix for the output file (default: bfast_monitor)")

    # bfastlite Subparser
    bfl_parser = subparsers.add_parser("bfast-lite", help="Run bfastlite (single-pass multiple-breakpoint detection)")
    _add_timeseries_args(bfl_parser)
    bfl_parser.add_argument("--start-time", type=float, default=None, help="Series start time (e.g. 2010.0; default: from the band dates)")
    bfl_parser.add_argument("--frequency", type=int, default=None, help="Observations per year (e.g. 23 for 16-day composites; default: from the band dates)")
    bfl_parser.add_argument("--order", type=int, default=3, help="Harmonic order (default: 3)")
    bfl_parser.add_argument("--h", type=float, default=0.15, help="Minimum segment size, as a fraction of the series length (default: 0.15)")
    bfl_parser.add_argument("--max-breaks-output", type=int, default=5, help="Maximum number of breakpoints to report per pixel (default: 5)")
    bfl_parser.add_argument("--min-valid", type=int, default=20, help="Minimum valid observations per pixel (default: 20)")
    bfl_parser.add_argument("--prefix", default="bfast_lite", help="Prefix for the output file (default: bfast_lite)")

    # bfast (classic) Subparser
    bf_parser = subparsers.add_parser("bfast", help="Run the classic iterative bfast() (trend + season break detection)")
    _add_timeseries_args(bf_parser)
    bf_parser.add_argument("--start-time", type=float, default=None, help="Series start time (e.g. 2000.0; default: from the band dates)")
    bf_parser.add_argument("--frequency", type=int, default=None, help="Observations per year (e.g. 23 for 16-day composites; default: from the band dates)")
    bf_parser.add_argument("--order", type=int, default=3, help="Harmonic order (default: 3)")
    bf_parser.add_argument("--h", type=float, default=0.15, help="Minimum segment size, as a fraction of valid observations (default: 0.15)")
    bf_parser.add_argument("--max-breaks-trend", type=int, default=5, help="Maximum number of trend breakpoints to report per pixel (default: 5)")
    bf_parser.add_argument("--max-breaks-season", type=int, default=5, help="Maximum number of season breakpoints to report per pixel (default: 5)")
    bf_parser.add_argument("--max-iter", type=int, default=10, help="Maximum trend/season re-estimation iterations (default: 10)")
    bf_parser.add_argument("--level", type=float, default=0.05, help="Significance threshold for the preliminary structural-stability pre-check (default: 0.05)")
    bf_parser.add_argument("--min-valid", type=int, default=20, help="Minimum valid observations per pixel (default: 20)")
    bf_parser.add_argument("--prefix", default="bfast", help="Prefix for the output file (default: bfast)")

    # Mann-Kendall Subparser
    mk_parser = subparsers.add_parser("mann-kendall", help="Run the Mann-Kendall trend test + Theil-Sen slope estimator")
    _add_timeseries_args(mk_parser)
    mk_parser.add_argument("--method", choices=["original", "hamed_rao", "yue_wang", "seasonal"], default="hamed_rao", help="Trend test variant (default: hamed_rao)")
    mk_parser.add_argument("--alpha", type=float, default=0.05, help="Significance level (default: 0.05)")
    mk_parser.add_argument("--lag", type=int, default=None, help="First significant lags for the autocorrelation correction (hamed_rao/yue_wang only; default: full series)")
    mk_parser.add_argument("--period", type=int, default=1, help="Season slots for method=seasonal (e.g. 23 for MODIS 16-day cycles; default: 1)")
    mk_parser.add_argument("--min-valid", type=int, default=4, help="Minimum valid observations per pixel (default: 4)")
    mk_parser.add_argument("--prefix", default="mann_kendall", help="Prefix for the output file (default: mann_kendall)")

    # MMU (Minimum Mapping Unit) Filter Subparser
    mmu_parser = subparsers.add_parser("mmu-filter", help="Apply a Minimum Mapping Unit spatial filter to a single-band raster")
    mmu_parser.add_argument("input", help="Path to input single-band GeoTIFF (e.g. a LandTrendr year-of-detection map)")
    mmu_parser.add_argument("output", help="Path to save the filtered GeoTIFF")
    mmu_parser.add_argument("--mmu-pixels", type=int, default=11, help="Minimum patch size in pixels; smaller patches are removed (default: 11)")

    # The cube functions: any raster load_raster reads, <output_dir>/<prefix>.tif out
    def _add_cube_args(p, prefix, what="Path to the input raster (dates in the band names, as save_raster writes them)"):
        p.add_argument("input", help=what)
        p.add_argument("output_dir", help="Directory to save the outputs")
        p.add_argument("--chunk-size", type=int, default=512, help="Size of the image chunks to process at once (default: 512)")
        p.add_argument("--jobs", type=int, default=-1, help="Number of CPU cores to use (-1 for all, default: -1)")
        p.add_argument("--prefix", default=prefix, help=f"Name of the output file (default: {prefix})")

    phen_parser = subparsers.add_parser("phenology", help="Land surface phenology: fit a seasonal curve, extract its metrics")
    _add_cube_args(phen_parser, "phenology")
    phen_parser.add_argument("--curve", default="beck", choices=["beck", "elmore", "gu", "klos", "zhang", "ag", "dl"], help="Curve fitted to each season (default: beck)")
    phen_parser.add_argument("--method", default="threshold", choices=["threshold", "derivative", "gu", "klosterman"], help="How transition dates are read from the curve (default: threshold)")
    phen_parser.add_argument("--weights", default=None, help="Raster of observation weights in [0, 1] on the input's grid and dates (e.g. from zeit.qc_sentinel2_scl)")
    phen_parser.add_argument("--not-annual", action="store_true", help="Metrics per season instead of per year")
    phen_parser.add_argument("--max-seasons", type=int, default=None, help="Seasons to report (default: the number of years)")
    phen_parser.add_argument("--whittaker-lambda", type=float, default=10.0, help="Whittaker smoothing before the fit (default: 10)")

    sm_parser = subparsers.add_parser("smooth", help="Smooth every pixel's series (Whittaker or Savitzky-Golay)")
    _add_cube_args(sm_parser, "smooth")
    sm_parser.add_argument("--method", choices=["whittaker", "savgol"], default="whittaker", help="Smoother (default: whittaker)")
    sm_parser.add_argument("--lmbda", type=float, default=10.0, help="Whittaker: smoothness (default: 10)")
    sm_parser.add_argument("--weights", default=None, help="Whittaker: raster of observation weights in [0, 1] on the input's grid and dates")
    sm_parser.add_argument("--window", type=int, default=5, help="Savitzky-Golay: window length (default: 5)")
    sm_parser.add_argument("--polyorder", type=int, default=2, help="Savitzky-Golay: polynomial order (default: 2)")

    tm_parser = subparsers.add_parser("tmask", help="Flag clouds and shadows from each pixel's series (Tmask): 1 clear, 0 not")
    _add_cube_args(tm_parser, "tmask", what="Path to a (time, band) stack with bands named date_band, as save_raster writes them")
    tm_parser.add_argument("--green", default="green", help="Name of the green band (default: green)")
    tm_parser.add_argument("--swir", default="swir1", help="Name of the SWIR-1 band (default: swir1)")
    tm_parser.add_argument("--scale", type=float, default=10000.0, help="Reflectance scale of the stack (default: 10000; 1 for 0-1 floats)")

    tw_parser = subparsers.add_parser("twdtw", help="Classify every pixel's series by TWDTW against patterns")
    _add_cube_args(tw_parser, "twdtw")
    tw_parser.add_argument("--patterns", required=True, help="CSV of the patterns: columns pattern, date and one value column (or one column per band)")
    tw_parser.add_argument("--band", default=None, help="With a multi-band input and one-band patterns: the band to classify")
    tw_parser.add_argument("--steepness", type=float, default=0.1, help="Steepness of the logistic time weight (default: 0.1)")
    tw_parser.add_argument("--midpoint", type=float, default=50.0, help="Midpoint of the logistic time weight, in days (default: 50)")
    tw_parser.add_argument("--max-elapsed", type=float, default=None, help="Never match observations farther apart than this many days")
    tw_parser.add_argument("--no-cycle", action="store_true", help="Measure elapsed time between dates, not days of the year")

    sn_parser = subparsers.add_parser("snic", help="Segment an image or a series into SNIC superpixels")
    _add_cube_args(sn_parser, "snic", what="Path to the input raster (an image, a stack or a series)")
    sn_parser.add_argument("--spacing", type=float, default=10, help="Pixels between seeds (default: 10)")
    sn_parser.add_argument("--compactness", type=float, default=0.5, help="Larger: more regular segments (default: 0.5, for 0-1 data)")
    sn_parser.add_argument("--grid", choices=["rectangular", "diamond", "hexagonal", "random"], default="rectangular", help="Seed grid (default: rectangular)")
    sn_parser.add_argument("--polygons", action="store_true", help="Also write the segments as polygons with their means (<prefix>.gpkg)")

    cl_parser = subparsers.add_parser("classify", help="Classify every pixel with a random forest trained on sample points")
    _add_cube_args(cl_parser, "classes", what="Path to the raster to classify (each band, or date and band, a feature)")
    cl_parser.add_argument("--samples", default=None, help="Vector file of sample points with their class")
    cl_parser.add_argument("--label", default="class", help="Column of the samples holding the class (default: class)")
    cl_parser.add_argument("--model", default=None, help="With --samples: where to save the trained model; without: a model to use (joblib)")
    cl_parser.add_argument("--probability", action="store_true", help="Also write each class's probability")

    som_parser = subparsers.add_parser("som", help="Cluster the pixels with a Self-Organizing Map")
    _add_cube_args(som_parser, "som")
    som_parser.add_argument("--x", type=int, default=3, help="Neurons along the first side of the grid (default: 3)")
    som_parser.add_argument("--y", type=int, default=3, help="Neurons along the second side of the grid (default: 3)")
    som_parser.add_argument("--sample", type=int, default=50000, help="Pixels to train on, at random (0 for all; default: 50000)")
    som_parser.add_argument("--algorithm", choices=["online", "batch"], default="online", help="Training algorithm (default: online)")
    som_parser.add_argument("--sigma", type=float, default=1.0, help="Initial neighbourhood radius (default: 1.0)")
    som_parser.add_argument("--learning-rate", type=float, default=0.5, help="Initial learning rate (default: 0.5)")
    som_parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    cd_parser = subparsers.add_parser("coded", help="Forest degradation and deforestation from the NDFI (CODED)")
    _add_cube_args(cd_parser, "coded", what="Path to a Landsat reflectance series (date_band bands: blue, green, red, "
                                            "nir, swir1, swir2), as save_raster writes it")
    cd_parser.add_argument("--start", default=None, help="Start of the monitoring, a year or a date (default: after the "
                                                         "first training period)")
    cd_parser.add_argument("--train-years", type=float, default=3.0, help="Years of the training period (default: 3)")
    cd_parser.add_argument("--consec", type=int, default=3, help="Consecutive observations that make a change (default: 3)")
    cd_parser.add_argument("--thresh", type=float, default=3.0, help="Change threshold, in RMSEs of the model (default: 3)")
    cd_parser.add_argument("--min-years", type=float, default=3.0, help="Years modelled after a change (default: 3)")
    cd_parser.add_argument("--max-events", type=int, default=3, help="Changes kept per pixel (default: 3)")
    cd_parser.add_argument("--training", default=None, help="Vector file of land cover points for the random forest")
    cd_parser.add_argument("--label", default="label", help="Column of the training points with the class (default: label)")
    cd_parser.add_argument("--forest-label", default=None, help="The class that is forest (default: forest or 1)")
    cd_parser.add_argument("--forest-ndfi", type=float, default=0.5, help="Without training points: forest where the "
                                                                          "model's mean NDFI is at least this (default: 0.5)")

    em_parser = subparsers.add_parser("embeddings", help="Download the yearly embeddings of a foundation model "
                                                         "(TESSERA, AlphaEarth) over a region, as one GeoTIFF")
    em_parser.add_argument("output_dir", help="Directory to save the GeoTIFF")
    em_parser.add_argument("--source", choices=["tessera", "alphaearth"], required=True, help="Which embeddings")
    em_parser.add_argument("--bbox", type=float, nargs=4, metavar=("WEST", "SOUTH", "EAST", "NORTH"), default=None,
                           help="Region bounds in longitude and latitude")
    em_parser.add_argument("--region", default=None, help="Vector file of the region (cells outside its polygons are NoData)")
    em_parser.add_argument("--years", nargs="+", default=None, help="Years, e.g. 2018 2020 or 2018-2024 (default: all)")
    em_parser.add_argument("--crs", default=None, help="CRS of the output (default: the region's UTM zone, the native grid)")
    em_parser.add_argument("--res", type=float, default=None, help="Cell size, in units of --crs (default: 10 m)")
    em_parser.add_argument("--version", default=None, help="TESSERA: dataset version (default: v1.1)")
    em_parser.add_argument("--variant", default=None, help="TESSERA: dataset variant (default: the version's)")
    em_parser.add_argument("--depth", type=int, default=None, help="TESSERA v2: the first DEPTH dimensions only")
    em_parser.add_argument("--backend", choices=["source.coop", "gee"], default=None,
                           help="AlphaEarth: the open COGs (default) or Earth Engine")
    em_parser.add_argument("--store", default=None, help="A local copy of the product to read instead of the public one")
    em_parser.add_argument("--cache-dir", default=None, help="Where downloads are kept (default: ~/.cache/zeit/embeddings)")
    em_parser.add_argument("--prefix", default=None, help="Name of the output file (default: the source)")

    args = parser.parse_args()

    if args.command == "landtrendr":
        run_landtrendr(args)
    elif args.command == "ccdc":
        run_ccdc_cli(args)
    elif args.command == "bfast-monitor":
        run_bfast_monitor_cli(args)
    elif args.command == "bfast-lite":
        run_bfast_lite_cli(args)
    elif args.command == "bfast":
        run_bfast_cli(args)
    elif args.command == "mann-kendall":
        run_mann_kendall_cli(args)
    elif args.command == "mmu-filter":
        run_mmu_filter_cli(args)
    elif args.command == "phenology":
        run_phenology_cli(args)
    elif args.command == "smooth":
        run_smooth_cli(args)
    elif args.command == "tmask":
        run_tmask_cli(args)
    elif args.command == "twdtw":
        run_twdtw_cli(args)
    elif args.command == "snic":
        run_snic_cli(args)
    elif args.command == "classify":
        run_classify_cli(args)
    elif args.command == "som":
        run_som_cli(args)
    elif args.command == "coded":
        run_coded_cli(args)
    elif args.command == "embeddings":
        run_embeddings_cli(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
