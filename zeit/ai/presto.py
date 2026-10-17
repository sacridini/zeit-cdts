"""Presto (Tseng et al. 2023): a small transformer pretrained on the time series of single
pixels (Sentinel-1, Sentinel-2, ERA5, SRTM, up to 24 months), any of which may be missing.
Ported from nasaharvest/presto (MIT); with the authors' weights it gives their encodings.

``Presto`` reads the bands of a zeit cube as they are (reflectance, dB, kelvin, metres),
NaN where missing, and does Presto's own normalization; ``presto_embeddings`` maps a cube
to Presto's 128-dimension encodings, an embedding cube like TESSERA's or AlphaEarth's.
"""
import hashlib
import math
from collections import OrderedDict
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#: Presto's inputs in its order, and the shift and divisor of its normalization.
INPUTS = ("VV", "VH", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
          "temperature_2m", "total_precipitation", "elevation", "slope", "NDVI")
_SHIFT = (25.0, 25.0) + (0.0,) * 10 + (-272.15, 0.0, 0.0, 0.0, 0.0)
_DIV = (25.0, 25.0) + (1e4,) * 10 + (35.0, 0.03, 2000.0, 50.0, 1.0)
_GROUPS = OrderedDict(S1=[0, 1], S2_RGB=[2, 3, 4], S2_Red_Edge=[5, 6, 7], S2_NIR_10m=[8], S2_NIR_20m=[9],
                      S2_SWIR=[10, 11], ERA5=[12, 13], SRTM=[14, 15], NDVI=[16])
_S1, _S2, _STATIC = ("VV", "VH"), INPUTS[2:12], ("elevation", "slope")
_DW_MISSING = 9         # Dynamic World has 9 classes; 9 is "no observation"
MAX_DATES = 24

#: Band names of a cube (any case) that hold each input; ``band_map`` adds others.
ALIASES = {
    "VV": ("vv",), "VH": ("vh",),
    "B2": ("blue", "b02", "b2"), "B3": ("green", "b03", "b3"), "B4": ("red", "b04", "b4"),
    "B5": ("rededge1", "b05", "b5"), "B6": ("rededge2", "b06", "b6"), "B7": ("rededge3", "b07", "b7"),
    "B8": ("nir", "b08", "b8"), "B8A": ("nir08", "b8a"),
    "B11": ("swir1", "swir16", "b11"), "B12": ("swir2", "swir22", "b12"),
    "temperature_2m": ("temperature_2m", "t2m"), "total_precipitation": ("total_precipitation", "tp"),
    "elevation": ("elevation", "dem"), "slope": ("slope",),
}

WEIGHTS_URL = ("https://raw.githubusercontent.com/nasaharvest/presto/"
               "11e207a668a34336ced1d8e492a1bd5849b96c4a/data/default_model.pt")
WEIGHTS_SHA256 = "559c9d76f525bdf2dd3774da8b1e7f1510769b73baacac12e9e4d8d5c67fbe65"
VERSION = "nasaharvest/presto@11e207a"


def pretrained_weights():
    """The authors' weights (3 MB): see ``zeit.ai._weights.fetch`` (``ZEIT_WEIGHTS`` for a
    local copy)."""
    from ._weights import fetch

    return fetch(WEIGHTS_URL, WEIGHTS_SHA256, "Presto's weights")


def match_bands(bands: Sequence[str], band_map: Optional[Dict[str, str]] = None) -> Dict[str, int]:
    """Presto input -> index of the cube band that holds it."""
    band_map = dict(band_map or {})
    unknown = set(band_map) - set(INPUTS[:-1])
    if unknown:
        raise ValueError(f"band_map keys are Presto's inputs {list(INPUTS[:-1])}, got {sorted(unknown)}")
    lower = {str(b).lower(): i for i, b in enumerate(bands)}
    found = {}
    for name in INPUTS[:-1]:
        if name in band_map:
            if band_map[name] not in list(bands):
                raise ValueError(f"band_map gives {band_map[name]!r} for {name}, but the bands are {list(bands)}")
            found[name] = list(bands).index(band_map[name])
            continue
        for alias in ALIASES[name]:
            if alias in lower and lower[alias] not in found.values():
                found[name] = lower[alias]
                break
    if not found:
        raise ValueError(f"none of the bands {list(bands)} is an input of Presto; name them as Sentinel-2 "
                         f"(blue, red, nir, B8A, swir1...), Sentinel-1 (VV, VH), ERA5 or SRTM bands, or pass "
                         f"band_map={{presto input: band}} (inputs: {list(INPUTS[:-1])})")
    return found


# ---------------------------------------------------------------------------
# The encoder, as the authors' (presto/presto.py), with the names of its weights
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        b, n, c = x.shape
        q, k, v = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).unbind(0)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[:, None, None])  # True: attend
        return self.proj(x.transpose(1, 2).reshape(b, n, c))


class _Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _Mlp(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor, keep: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), keep)
        return x + self.mlp(self.norm2(x))


def _sinusoids(n: int, d: int, period: float = 1000.0) -> torch.Tensor:
    angles = np.array([[p / np.power(period, 2 * (j // 2) / d) for j in range(d)] for p in range(n)])
    angles[:, 0::2] = np.sin(angles[:, 0::2])
    angles[:, 1::2] = np.cos(angles[:, 1::2])
    return torch.FloatTensor(angles)


def _month_table(d: int) -> torch.Tensor:
    angles = np.arange(0, 13) / (12 / (2 * np.pi))
    sin = np.sin(np.stack([angles] * (d // 2), axis=-1))
    cos = np.cos(np.stack([angles] * (d // 2), axis=-1))
    return torch.FloatTensor(np.concatenate([sin[:-1], cos[:-1]], axis=-1))


def _cartesian(latlons: torch.Tensor) -> torch.Tensor:
    lat, lon = (latlons * math.pi / 180).unbind(-1)
    return torch.stack([torch.cos(lat) * torch.cos(lon), torch.cos(lat) * torch.sin(lon), torch.sin(lat)], dim=-1)


class _Encoder(nn.Module):
    def __init__(self, embedding_size: int = 128, depth: int = 2, mlp_ratio: float = 4.0, num_heads: int = 8):
        super().__init__()
        self.embedding_size = embedding_size
        self.eo_patch_embed = nn.ModuleDict({g: nn.Linear(len(idx), embedding_size) for g, idx in _GROUPS.items()})
        self.dw_embed = nn.Embedding(_DW_MISSING + 1, embedding_size)
        self.latlon_embed = nn.Linear(3, embedding_size)
        self.blocks = nn.ModuleList([_Block(embedding_size, num_heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(embedding_size)
        # each token's embedding: its month, its group of bands (channel) and its date
        quarter = embedding_size // 4
        self.pos_embed = nn.Parameter(_sinusoids(MAX_DATES, embedding_size - 2 * quarter)[None], requires_grad=False)
        self.month_embed = nn.Embedding.from_pretrained(_month_table(quarter), freeze=True)
        self.channel_embed = nn.Embedding(len(_GROUPS) + 1, quarter)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, missing: torch.Tensor, dynamic_world: torch.Tensor,
                latlons: torch.Tensor, months: torch.Tensor) -> torch.Tensor:
        """x, missing: (b, t, 17), normalized; dynamic_world, months (0-11): (b, t);
        latlons: (b, 2) in degrees. The mean of the tokens observed: (b, embedding)."""
        b, t = x.shape[:2]
        month = self.month_embed(months)
        pos = self.pos_embed[:, :t].expand(b, -1, -1)
        tokens, masks = [], []
        for i, (group, idx) in enumerate(_GROUPS.items()):
            tok = self.eo_patch_embed[group](x[:, :, idx])
            channel = self.channel_embed.weight[i].expand(b, t, -1)
            if group == "SRTM":   # static: one token, with neither month nor date
                tok = tok[:, :1] + torch.cat((torch.zeros_like(month[:, :1]), channel[:, :1],
                                              torch.zeros_like(pos[:, :1])), -1)
                masks.append(missing[:, :1, idx].amax(-1))
            else:
                tok = tok + torch.cat((month, channel, pos), -1)
                masks.append(missing[:, :, idx].amax(-1))
            tokens.append(tok)
        channel = self.channel_embed.weight[len(_GROUPS)].expand(b, t, -1)
        tokens.append(self.dw_embed(dynamic_world) + torch.cat((month, channel, pos), -1))
        masks.append(dynamic_world == _DW_MISSING)
        x = torch.cat(tokens, 1)
        mask = torch.cat(masks, 1).bool()
        # the observed tokens first, cut at the most any sample has
        kept, order = torch.sort((~mask).int(), dim=1, descending=True, stable=True)
        x = x.gather(1, order[:, :, None].expand_as(x)) * kept.unsqueeze(-1)
        n = int(kept.sum(-1).max())
        x, keep = x[:, :n], kept[:, :n].bool()
        x = torch.cat((self.latlon_embed(_cartesian(latlons)).unsqueeze(1), x), 1)
        keep = torch.cat((keep.new_ones(b, 1), keep), 1)
        for block in self.blocks:
            x = block(x, keep)
        w = keep.unsqueeze(-1).to(x.dtype)
        return self.norm((x * w).sum(1) / w.sum(1))


# ---------------------------------------------------------------------------
# On the bands of a cube
# ---------------------------------------------------------------------------

class Presto(nn.Module):
    """Presto's encoder and a linear classifier, on the series of a pixel ``(batch, dates,
    bands)`` in the cube's units, NaN where missing; ``embed`` gives the encodings.

    Parameters
    ----------
    bands
        The cube's bands, in order; matched to Presto's inputs by name (``ALIASES``).
    months
        Calendar month (1-12) of each date; up to 24 dates.
    num_classes
        Classes of the classifier.
    latlon
        Latitude and longitude (degrees) given to every pixel when ``forward``/``embed``
        get none of their own.
    band_map
        ``{Presto input: band}`` for bands whose names are not recognized.
    reflectance_scale
        Factor bringing the Sentinel-2 bands to Presto's 0-10000 (``1e4`` for reflectance
        from 0 to 1).
    s1_linear
        The Sentinel-1 bands hold linear backscatter (converted to dB) rather than dB.
    pretrained
        Start from the authors' weights (``pretrained_weights``).
    """

    def __init__(self, bands: Sequence[str], months: Sequence[int], num_classes: int = 2,
                 latlon: Optional[Sequence[float]] = None, band_map: Optional[Dict[str, str]] = None,
                 reflectance_scale: float = 1.0, s1_linear: bool = False, pretrained: bool = True):
        super().__init__()
        if not 1 <= len(months) <= MAX_DATES:
            raise ValueError(f"Presto takes 1 to {MAX_DATES} dates (e.g. monthly composites of two years), "
                             f"got {len(months)}")
        if any(not 1 <= int(m) <= 12 for m in months):
            raise ValueError(f"months are calendar months, 1 to 12; got {list(months)}")
        self.n_bands = len(bands)
        self.inputs = match_bands(bands, band_map)   # input -> band index
        names = list(self.inputs)
        self.register_buffer("_src", torch.tensor([self.inputs[k] for k in names]), persistent=False)
        self.register_buffer("_dst", torch.tensor([INPUTS.index(k) for k in names]), persistent=False)
        self.register_buffer("_scale", torch.tensor([reflectance_scale if k in _S2 else 1.0 for k in names]),
                             persistent=False)
        self.register_buffer("_db", torch.tensor([s1_linear and k in _S1 for k in names]), persistent=False)
        self.register_buffer("_static", torch.tensor([k in _STATIC for k in names]), persistent=False)
        self.register_buffer("_shift", torch.tensor([_SHIFT[INPUTS.index(k)] for k in names]), persistent=False)
        self.register_buffer("_div", torch.tensor([_DIV[INPUTS.index(k)] for k in names]), persistent=False)
        self.register_buffer("months", torch.tensor([int(m) - 1 for m in months]), persistent=False)
        self.register_buffer("latlon", torch.tensor(list(latlon) if latlon is not None else [math.nan] * 2,
                                                    dtype=torch.float32), persistent=False)
        self.encoder = _Encoder()
        if pretrained:
            state = torch.load(pretrained_weights(), map_location="cpu", weights_only=True)
            self.encoder.load_state_dict({k[len("encoder."):]: v for k, v in state.items() if k.startswith("encoder.")})
        self.head = nn.Linear(self.encoder.embedding_size, num_classes)

    def _prepare(self, x: torch.Tensor):
        """The 17 normalized inputs and which are missing, from the cube's bands."""
        b, t, nb = x.shape
        if t != len(self.months) or nb != self.n_bands:
            raise ValueError(f"this Presto takes {len(self.months)} dates of {self.n_bands} bands, "
                             f"got {t} of {nb}")
        v = x[..., self._src].float() * self._scale
        if bool(self._db.any()):
            v = torch.where(self._db, 10 * torch.log10(v.clamp_min(1e-10)), v)
        if bool(self._static.any()):   # Presto reads SRTM from the first date: the mean of the dates
            v = torch.where(self._static, torch.nanmean(v, dim=1, keepdim=True).expand_as(v), v)
        missing = torch.isnan(v)
        v = torch.where(missing, torch.zeros_like(v), (v + self._shift) / self._div)
        out = v.new_zeros(b, t, len(INPUTS))
        miss = torch.ones(b, t, len(INPUTS), dtype=torch.bool, device=x.device)
        out[..., self._dst] = v
        miss[..., self._dst] = missing
        nir, red = out[..., INPUTS.index("B8")], out[..., INPUTS.index("B4")]
        total = nir + red
        out[..., -1] = torch.where(total > 0, (nir - red) / torch.where(total > 0, total, torch.ones_like(total)), 0)
        miss[..., -1] = miss[..., INPUTS.index("B8")] | miss[..., INPUTS.index("B4")]
        return out, miss

    def embed(self, x: torch.Tensor, latlons: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Presto's encoding of each series: ``(batch, 128)``."""
        b = x.shape[0]
        if latlons is None:
            if bool(torch.isnan(self.latlon).any()):
                raise ValueError("Presto needs the latitude and longitude of the pixels: build it with latlon=")
            latlons = self.latlon.expand(b, 2)
        inputs, missing = self._prepare(x)
        dw = torch.full((b, inputs.shape[1]), _DW_MISSING, dtype=torch.long, device=x.device)
        return self.encoder(inputs, missing, dw, latlons.to(inputs.dtype), self.months.expand(b, -1))

    def forward(self, x: torch.Tensor, latlons: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.head(self.embed(x, latlons))


# ---------------------------------------------------------------------------
# Settings from the data
# ---------------------------------------------------------------------------

def settings(bands: Sequence[str], times: Optional[Sequence[Any]], low: Sequence[float], high: Sequence[float],
             band_map: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Presto's arguments for a cube with these bands and dates, whose bands range from
    ``low`` to ``high`` (2% and 98% quantiles): the months, and the units of the bands
    (reflectance from 0 to 1 or 0 to 10000; dB or linear backscatter)."""
    import pandas as pd

    if times is None:
        raise ValueError("Presto needs the dates of the cube (it encodes their months)")
    found = match_bands(bands, band_map)
    s2 = [i for k, i in found.items() if k in _S2]
    s1 = [i for k, i in found.items() if k in _S1]
    out: Dict[str, Any] = {"bands": [str(b) for b in bands], "months": [int(pd.Timestamp(t).month) for t in times]}
    if band_map:
        out["band_map"] = dict(band_map)
    if s2:
        out["reflectance_scale"] = 1e4 if max(high[i] for i in s2) <= 2.0 else 1.0
    if s1:
        out["s1_linear"] = bool(min(low[i] for i in s1) >= 0.0)   # dB backscatter is mostly negative
    return out


def presto_embeddings(
    data: Any,
    *,
    model: Optional[Presto] = None,
    band_map: Optional[Dict[str, str]] = None,
    batch_size: int = 8192,
    device: str = "auto",
    nodata: Any = "auto",
    chunks: Any = None,
):
    """Presto's encoding of every pixel of a cube: an embedding cube of 128 dimensions.

    Parameters
    ----------
    data
        A ``(time, band, y, x)`` cube of up to 24 dates (e.g. monthly composites of a year)
        with Sentinel-2 and/or Sentinel-1, ERA5 or SRTM bands, in memory or dask, or
        anything ``zeit.load_raster`` reads. Missing values are NaN or ``nodata``.
    model
        A ``Presto`` (e.g. one ``zeit.ai.train`` fine-tuned) whose encoder to use; by
        default the authors' pretrained one, set for the cube's bands and dates.
    band_map
        ``{Presto input: band}`` for bands whose names are not recognized.
    batch_size, device
        Pixels per step, and where (``"auto"``: the GPU if there is one).
    nodata, chunks
        As in ``zeit.ai.predict``.

    Returns
    -------
    xarray.DataArray
        ``(time, band, y, x)`` with one time (the first date) and bands ``A00``-``A127``,
        tagged as Presto embeddings: ``similarity``, ``som``, ``train_classifier``,
        ``zeit.plot`` and the rest take it as they take TESSERA's. Each pixel gets its own
        latitude and longitude.
    """
    import xarray as xr

    from .pipeline import _cube, _device

    cube, times = _cube(data, nodata, chunks)
    if cube.rio.crs is None:
        raise ValueError("the cube has no CRS: Presto needs the latitude and longitude of its pixels")
    bands = [str(b) for b in cube.band.values]
    if model is None:
        step = max(1, int(math.sqrt(cube.sizes["y"] * cube.sizes["x"] / 250_000)))
        sample = np.asarray(cube.isel(y=slice(None, None, step), x=slice(None, None, step)).values, dtype=np.float32)
        by_band = np.moveaxis(sample, 1, 0).reshape(len(bands), -1)
        low, high = np.zeros(len(bands)), np.ones(len(bands))
        for i, values in enumerate(by_band):
            values = values[np.isfinite(values)]
            if len(values):
                low[i], high[i] = np.quantile(values, [0.02, 0.98])
        model = Presto(**settings(bands, times, low, high, band_map), num_classes=1)
        variant = "pretrained"
    else:
        state = b"".join(v.detach().cpu().numpy().tobytes() for v in model.encoder.state_dict().values())
        variant = "fine-tuned " + hashlib.sha256(state).hexdigest()[:12]   # its own space
    dev = _device(device)
    model = model.to(dev).eval()
    dims = model.encoder.embedding_size
    crs, ys, xs = cube.rio.crs, cube.y.values, cube.x.values

    def run(block: np.ndarray, r0: int, c0: int) -> np.ndarray:
        t, nb, h, w = block.shape
        series = np.moveaxis(block.reshape(t, nb, h * w), -1, 0).astype(np.float32)   # (pixels, time, band)
        lon, lat = _lonlat(crs, ys[r0:r0 + h], xs[c0:c0 + w])
        latlon = np.stack([lat.ravel(), lon.ravel()], axis=1)
        out = np.full((h * w, dims), np.nan, dtype=np.float32)
        has = np.flatnonzero(np.isfinite(series).any(axis=(1, 2)))
        with torch.inference_mode():
            for i in range(0, len(has), batch_size):
                idx = has[i:i + batch_size]
                emb = model.embed(torch.from_numpy(series[idx]).to(dev), torch.from_numpy(latlon[idx]).to(dev))
                out[idx] = emb.float().cpu().numpy()
        return out.T.reshape(1, dims, h, w)

    arr = cube.data
    if hasattr(arr, "dask"):
        def block(b, block_info=None):
            (r0, _), (c0, _) = block_info[0]["array-location"][2:]
            return run(b, r0, c0)

        arr = arr.rechunk({0: -1, 1: -1})
        values = arr.map_blocks(block, dtype=np.float32, chunks=((1,), (dims,)) + arr.chunks[2:])
    else:
        values = run(np.asarray(arr), 0, 0)
    first = cube.time.values[:1] if "time" in cube.coords else [0]
    out = xr.DataArray(values, dims=("time", "band", "y", "x"), name="embeddings",
                       coords={"time": first, "band": [f"A{i:02d}" for i in range(dims)], "y": cube.y, "x": cube.x})
    out = out.rio.write_crs(crs)
    out.attrs.update({"embedding_source": "presto", "embedding_version": VERSION, "embedding_variant": variant,
                      "embedding_model": "Presto", "embedding_dimensions": dims, "embedding_license": "MIT",
                      "embedding_citation": "Tseng et al. (2023), Lightweight, Pre-trained Transformers for Remote "
                                            "Sensing Timeseries, arXiv:2304.14065"})
    return out


def _lonlat(crs, ys: np.ndarray, xs: np.ndarray):
    """Longitude and latitude (degrees) of the cells with these centres, ``(y, x)`` each."""
    from rasterio.warp import transform

    gx, gy = np.meshgrid(xs, ys)
    lon, lat = transform(crs, "EPSG:4326", gx.ravel(), gy.ravel())
    return (np.asarray(lon, dtype=np.float32).reshape(gx.shape), np.asarray(lat, dtype=np.float32).reshape(gx.shape))
