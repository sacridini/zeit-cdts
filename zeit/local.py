import xarray as xr


def build_local_cube(data_dir: str, regex_pattern: str, date_format: str = "%Y%m%d") -> xr.DataArray:
    """
    Build a lazy (time, band, y, x) cube from a folder of single-date rasters.

    A shortcut for ``load_raster(data_dir, pattern=regex_pattern, date_format=date_format,
    recursive=True, chunks="auto")``: ``regex_pattern`` needs a named group ``date`` and
    may have a group ``band``; files without a band group get the band ``"value"``.
    """
    from ._load import load_raster

    cube = load_raster(data_dir, pattern=regex_pattern, date_format=date_format,
                       recursive=True, chunks="auto")
    if "band" not in cube.dims:
        cube = cube.expand_dims(band=["value"], axis=1)
    return cube
