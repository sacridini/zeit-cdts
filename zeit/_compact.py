"""Compact (int8) copies of embedding cubes: ``save_raster(emb, path, compact=True)`` and the matching
read of ``load_raster``.

Two encodings, named in the file's ``ZEIT_EMBEDDING_ENCODING`` tag; NoData is -128 in both.

``aef-int8`` (AlphaEarth): the product's own curve, ``v = sign(x) * round(sqrt(|x|) * 127.5)`` clipped to
[-127, 127], decoded as ``x = sign(v) * (v / 127.5)**2``. Exact for values that came from the product.

``scaled-int8`` (everything else): per pixel and date, D dimension bands ``q`` and one band of the scale
code ``c``: ``x = q * 2**(c / 8)``, with ``c = ceil(8 * log2(m / 127))`` for ``m = max |x|`` (-127 when
``m == 0``), clipped to [-127, 127], and ``q = round(x / 2**(c / 8))`` clipped to [-127, 127].
"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import xarray as xr

ENCODING_TAG = "ZEIT_EMBEDDING_ENCODING"
AEF = "aef-int8"
SCALED = "scaled-int8"
ENCODINGS = (AEF, SCALED)
NODATA = -128
SCALE_BAND = "scale"
_LIMIT = 2.0 ** (127 / 8)    # the largest scale 2**(c/8)


def encoding_of(attrs: Any) -> str:
    """The encoding of a cube: AlphaEarth's own curve, or a per-pixel scale for any other embeddings."""
    return AEF if attrs.get("embedding_source") == "alphaearth" else SCALED


def is_scale_label(label: Any) -> bool:
    text = str(label)
    return text == SCALE_BAND or text.endswith("_" + SCALE_BAND)


# ---------------------------------------------------------------------------
# Blocks (numpy), the dimensions on axis -3
# ---------------------------------------------------------------------------

def encode_aef(block: np.ndarray) -> np.ndarray:
    """(..., D, y, x) floats as int8 with AlphaEarth's curve; -128 in all the dimensions of a pixel-date
    where one is NaN."""
    x = np.asarray(block, dtype=np.float64)
    bad = np.isnan(x).any(axis=-3, keepdims=True)
    x = np.where(np.isnan(x), 0.0, x)
    v = np.sign(x) * np.rint(np.sqrt(np.abs(x)) * 127.5)
    v = np.clip(v, -127, 127)
    return np.where(bad, NODATA, v).astype(np.int8)


def decode_aef(block: np.ndarray) -> np.ndarray:
    from ._alphaearth import dequantise

    return dequantise(block)


def encode_scaled(block: np.ndarray) -> np.ndarray:
    """(..., D, y, x) floats as (..., D + 1, y, x) int8: the D dimensions and the scale code."""
    x = np.asarray(block, dtype=np.float64)
    bad = np.isnan(x).any(axis=-3, keepdims=True)
    x = np.where(np.isnan(x), 0.0, x)
    m = np.abs(x).max(axis=-3, keepdims=True)
    if (m / 127 > _LIMIT).any():
        raise ValueError(f"the embeddings are too large for compact=True (a value of {float(m.max()):.6g}; the "
                         f"largest that fits is {127 * _LIMIT:.6g})")
    with np.errstate(divide="ignore"):
        c = np.ceil(8 * np.log2(m / 127))
    c = np.clip(np.where(m > 0, c, -127), -127, 127)
    c = np.minimum(c + (np.rint(m / np.exp2(c / 8)) > 127), 127)   # float rounding at a power-of-two edge
    q = np.clip(np.rint(x / np.exp2(c / 8)), -127, 127)
    out = np.concatenate([q, c], axis=-3)
    return np.where(bad, NODATA, out).astype(np.int8)


def decode_scaled(block: np.ndarray, dates: int) -> np.ndarray:
    """(dates * (D + 1), y, x) int8 as (dates * D, y, x) float32, NaN where there is no embedding."""
    n, h, w = block.shape
    group = n // dates
    b = np.asarray(block).reshape(dates, group, h, w)
    q, c = b[:, :-1], b[:, -1:]
    bad = (c == NODATA) | (q == NODATA).any(axis=1, keepdims=True)
    x = q.astype(np.float64) * np.exp2(c.astype(np.float64) / 8)
    x = np.where(bad, np.nan, x).astype(np.float32)
    return x.reshape(dates * (group - 1), h, w)


# ---------------------------------------------------------------------------
# Cubes
# ---------------------------------------------------------------------------

def encode_cube(da: xr.DataArray) -> Tuple[xr.DataArray, str]:
    """The int8 cube to write (lazy when ``da`` is) and its encoding. Its dims are those of ``da``, with the
    ``band`` axis one longer (the scale) for ``scaled-int8``."""
    from ._save import _spatial_last

    attrs = da.attrs
    if not attrs.get("embedding_source"):
        raise ValueError("compact=True is for cubes of embeddings (zeit.load_embeddings, zeit.ai.presto_embeddings): "
                         "the data has no embedding_source attribute")
    da = _spatial_last(da)
    if "band" not in da.dims or "y" not in da.dims or "x" not in da.dims:
        raise ValueError(f"compact=True needs (time, band, y, x) or (band, y, x) embeddings, got dims {da.dims}")
    da = da.transpose(*[d for d in da.dims if d not in ("band", "y", "x")], "band", "y", "x")
    encoding = encoding_of(attrs)
    func = encode_aef if encoding == AEF else encode_scaled
    n = da.sizes["band"]
    axis = da.ndim - 3
    data = da.data
    if hasattr(data, "chunks") and data.chunks is not None:
        data = data.rechunk({axis: -1})
        chunks = list(data.chunks)
        if encoding == SCALED:
            chunks[axis] = (n + 1,)
        out = data.map_blocks(func, dtype=np.int8, chunks=tuple(chunks))
    else:
        out = func(np.asarray(data))
    names = [str(v) for v in da["band"].values] if "band" in da.coords else [str(i + 1) for i in range(n)]
    if encoding == SCALED:
        names = names + [SCALE_BAND]
    coords = {k: v for k, v in da.coords.items() if "band" not in v.dims}
    coords["band"] = names
    enc = xr.DataArray(out, dims=da.dims, coords=coords, attrs=dict(attrs), name=da.name)
    return enc, encoding


def decode_cube(da: xr.DataArray, encoding: str, descriptions: Sequence[Optional[str]]
                ) -> Tuple[xr.DataArray, List[Optional[str]]]:
    """A ``(band, y, x)`` int8 raster as float32 (lazy): the dimension bands only, NaN where there is no
    embedding. ``descriptions`` are the file's band descriptions; the ones kept are returned."""
    if encoding not in ENCODINGS:
        raise ValueError(f"unknown embedding encoding {encoding!r} in the file (this zeit reads "
                         f"{', '.join(ENCODINGS)}); update zeit to read it")
    if da.chunks is None:
        da = da.chunk({"band": -1, "y": 1024, "x": 1024})
    descriptions = list(descriptions)
    if encoding == AEF:
        out = da.data.map_blocks(decode_aef, dtype=np.float32)
        return _as_float(da, out, None), descriptions
    count = da.sizes["band"]
    scale_at = [i for i, d in enumerate(descriptions) if d and is_scale_label(d)]
    dates = len(scale_at)
    if dates == 0 or count % dates or scale_at != list(range(count // dates - 1, count, count // dates)):
        raise ValueError(f"a {SCALED} file has one scale band after the dimensions of each date: bands "
                         f"{scale_at or 'named scale'} of {count}")
    keep = [i for i in range(count) if i not in scale_at]
    data = da.data.rechunk({0: -1})
    chunks = ((len(keep),),) + tuple(data.chunks[1:])
    out = data.map_blocks(decode_scaled, dates, dtype=np.float32, chunks=chunks)
    return _as_float(da, out, keep), [descriptions[i] for i in keep]


def _as_float(da: xr.DataArray, data: Any, keep: Optional[List[int]]) -> xr.DataArray:
    base = da if keep is None else da.isel(band=keep)
    out = base.copy(data=data)
    out.attrs = {k: v for k, v in da.attrs.items() if k not in ("_FillValue", ENCODING_TAG)}
    out.encoding = {}
    return out.rio.write_nodata(np.nan, encoded=False)
