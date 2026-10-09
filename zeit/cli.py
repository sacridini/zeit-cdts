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
        apply_mmu_filter(args.input, args.output, mmu_pixels=args.mmu_pixels)
    except Exception as e:
        print(f"Error running MMU filter: {e}")
        sys.exit(1)

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
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
