"""How to colour a map: chosen from the data, used by the static and the interactive plots.

A ``Style`` turns values into colour indices 1..255 (0 is NoData, drawn transparent) and
carries the 256-colour lookup table that paints them, so the browser only needs bytes and
a LUT, and matplotlib gets the same colours.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import re

import numpy as np

KINDS = ("continuous", "diverging", "categorical", "years", "rgb")
#: Index-like names that read better green-high (vegetation indices).
GREEN_HIGH = {"ndvi", "evi", "evi2", "savi", "msavi", "ndmi", "nbr", "nbr2", "gndvi", "kndvi", "ndwi", "mndwi"}
#: Names of maps that hold years (LandTrendr's yod, BFAST break years...).
YEAR_NAMES = {"yod", "year", "years", "break_year", "loss_year", "gain_year", "first_year", "last_year"}
MAX_CLASSES = 32
_PALETTE = [  # tab20, reordered so the first ten are the strong tab10 colours
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
]


def _hex_rgba(color: Any) -> Tuple[int, int, int, int]:
    from matplotlib.colors import to_rgba
    r, g, b, a = to_rgba(color)
    return int(round(r * 255)), int(round(g * 255)), int(round(b * 255)), int(round(a * 255))


@dataclass
class Style:
    kind: str
    cmap: str = "viridis"
    vmin: float = 0.0
    vmax: float = 1.0
    classes: List[Tuple[float, str, str]] = field(default_factory=list)   # (value, label, colour)
    label: str = ""

    # ---- colour indices -------------------------------------------------------------
    def encode(self, values: np.ndarray, nodata: Optional[float] = None) -> np.ndarray:
        """Values -> uint8 colour indices: 0 NoData/NaN, 1..255 the colours.
        RGB frames ``(h, w, 3)`` give ``(h, w, 3)`` bytes, 0 where any band is missing."""
        values = np.asarray(values)
        missing = _missing(values, nodata)
        if self.kind == "rgb":
            scaled = np.clip((values.astype(np.float32) - self.vmin) * (254.0 / max(self.vmax - self.vmin, 1e-12)) + 1,
                             1, 255)
            out = np.nan_to_num(scaled).astype(np.uint8)
            out[missing.any(axis=-1) if missing.ndim == 3 else missing] = 0
            return out
        if self.kind == "categorical":
            out = np.zeros(values.shape, dtype=np.uint8)
            for i, (value, _, _) in enumerate(self.classes, start=1):
                out[values == value] = i
            out[missing] = 0
            return out
        lo, hi = self.vmin, self.vmax
        scaled = (values.astype(np.float32) - lo) * (254.0 / max(hi - lo, 1e-12)) + 1
        out = np.nan_to_num(np.clip(scaled, 1, 255), nan=0).astype(np.uint8)
        out[missing] = 0
        return out

    def decode(self, index: int) -> Optional[float]:
        """The value at the centre of a colour index (for hover read-outs)."""
        if index <= 0:
            return None
        if self.kind == "categorical":
            return self.classes[index - 1][0] if index - 1 < len(self.classes) else None
        return self.vmin + (index - 1) * (self.vmax - self.vmin) / 254.0

    def lut(self) -> np.ndarray:
        """(256, 4) uint8 RGBA: index 0 transparent, 1..255 the colours."""
        out = np.zeros((256, 4), dtype=np.uint8)
        if self.kind == "categorical":
            for i, (_, _, color) in enumerate(self.classes, start=1):
                out[i] = _hex_rgba(color)
            return out
        if self.kind == "rgb":
            ramp = np.linspace(0, 255, 255).astype(np.uint8)
            out[1:, 0] = out[1:, 1] = out[1:, 2] = ramp
            out[1:, 3] = 255
            return out
        from matplotlib import colormaps
        out[1:] = (colormaps[self.cmap](np.linspace(0, 1, 255)) * 255).round().astype(np.uint8)
        return out

    def matplotlib_cmap(self):
        from matplotlib import colormaps
        from matplotlib.colors import ListedColormap
        if self.kind in ("continuous", "diverging", "years"):
            return colormaps[self.cmap]
        return ListedColormap(self.lut()[1:].astype(float) / 255.0)

    def ticks(self, n: int = 5) -> List[Tuple[float, str]]:
        """(value, text) colorbar ticks; whole years for ``years``."""
        if self.kind == "categorical":
            return [(v, label) for v, label, _ in self.classes]
        if self.kind == "years":
            lo, hi = int(np.ceil(self.vmin)), int(np.floor(self.vmax))
            step = max(1, int(np.ceil((hi - lo + 1) / n)))
            return [(float(y), str(y)) for y in range(lo, hi + 1, step)]
        from matplotlib.ticker import MaxNLocator
        values = [v for v in MaxNLocator(n).tick_values(self.vmin, self.vmax) if self.vmin <= v <= self.vmax]
        return [(float(v), _fmt(v, self.vmax - self.vmin)) for v in values]

    def to_json(self) -> Dict[str, Any]:
        return {"kind": self.kind, "cmap": self.cmap, "vmin": float(self.vmin), "vmax": float(self.vmax),
                "label": self.label, "ticks": [[float(v), t] for v, t in self.ticks()],
                "classes": [[float(v), str(t), c] for v, t, c in self.classes]}


def _fmt(value: float, span: float) -> str:
    if span == 0:
        return f"{value:g}"
    digits = max(0, 2 - int(np.floor(np.log10(abs(span)))))
    return f"{value:.{min(digits, 6)}f}"


def _missing(values: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    missing = np.zeros(values.shape, dtype=bool)
    if np.issubdtype(values.dtype, np.floating):
        missing |= np.isnan(values)
    if nodata is not None and not (isinstance(nodata, float) and np.isnan(nodata)):
        missing |= values == nodata
    return missing


def infer_style(sample: np.ndarray, *, name: str = "", dtype: Any = None, rgb: bool = False,
                kind: Optional[str] = None, cmap: Optional[str] = None, vmin: Optional[float] = None,
                vmax: Optional[float] = None, classes: Union[Dict[Any, Any], Sequence[Any], None] = None,
                nodata: Optional[float] = None, label: Optional[str] = None) -> Style:
    """Choose how to colour the data from a sample of it; every choice can be overridden.

    - RGB cubes -> ``rgb`` with a robust 2-98 % stretch shared by the three bands;
    - ``bool``, ``classes=`` given, or integers with at most 32 distinct values (and not a
      year map) -> ``categorical``, one colour per value and a legend;
    - year maps (``yod``, ``year``...; integers between 1800 and 2200) -> ``years``;
    - floats spanning 0 with comparable negative and positive ranges -> ``diverging``,
      symmetric around 0;
    - anything else -> ``continuous``, limits from the 2nd and 98th percentiles.
    """
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    dtype = np.dtype(dtype if dtype is not None else sample.dtype)
    values = np.asarray(sample)
    valid = values[~_missing(values, nodata)]
    if valid.dtype.kind == "M":
        valid = valid.astype("datetime64[D]").astype(np.int64) / 365.2425 + 1970
    finite = valid[np.isfinite(valid)] if valid.dtype.kind == "f" else valid
    lname = (name or "").lower()
    text = label if label is not None else (name or "")

    if rgb or kind == "rgb":
        lo, hi = _robust(finite)
        return Style("rgb", vmin=vmin if vmin is not None else lo, vmax=vmax if vmax is not None else hi, label=text)

    uniques = np.unique(finite) if finite.size and finite.dtype.kind in "iub" else None
    years_found = uniques[uniques != 0] if uniques is not None else None   # 0: "no event" in year maps
    looks_years = (lname in YEAR_NAMES or lname.endswith("_year") or dtype.kind == "M") or (
        years_found is not None and years_found.size > 1 and years_found.min() >= 1800 and years_found.max() <= 2200)
    if kind == "categorical" or classes is not None or dtype == bool or (
            kind is None and uniques is not None and uniques.size <= MAX_CLASSES and not looks_years):
        return Style("categorical", classes=_classes(classes, uniques, dtype), label=text)
    if kind == "years" or (kind is None and looks_years):
        known = finite[finite != 0] if finite.size and dtype.kind in "iu" else finite
        lo = vmin if vmin is not None else float(known.min()) if known.size else 1985.0
        hi = vmax if vmax is not None else float(known.max()) if known.size else 2025.0
        return Style("years", cmap=cmap or "plasma", vmin=lo, vmax=max(hi, lo + 1), label=text)

    lo, hi = _robust(finite)
    if kind == "diverging" or (kind is None and lo < 0 < hi and min(-lo, hi) / max(-lo, hi) > 0.2):
        m = max(abs(lo), abs(hi)) if vmin is None and vmax is None else max(abs(vmin or 0), abs(vmax or 0))
        return Style("diverging", cmap=cmap or "RdBu", vmin=vmin if vmin is not None else -m,
                     vmax=vmax if vmax is not None else m, label=text)
    tokens = set(re.split(r"[^a-z0-9]+", lname))
    default = "RdYlGn" if tokens & GREEN_HIGH else "viridis"
    return Style("continuous", cmap=cmap or default, vmin=vmin if vmin is not None else lo,
                 vmax=vmax if vmax is not None else max(hi, lo + 1e-9), label=text)


def _robust(values: np.ndarray) -> Tuple[float, float]:
    if values.size == 0:
        return 0.0, 1.0
    lo, hi = np.percentile(values.astype(np.float64), [2, 98])
    if lo == hi:
        lo, hi = float(values.min()), float(values.max())
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def classes_from_attrs(attrs: Dict[str, Any]) -> Optional[Dict[float, str]]:
    """``{value: name}`` from CF's ``flag_values`` and ``flag_meanings`` (e.g. the ``label``
    of ``zeit.twdtw``), or None."""
    values, meanings = attrs.get("flag_values"), attrs.get("flag_meanings")
    if values is None or not isinstance(meanings, str):
        return None
    values = np.atleast_1d(np.asarray(values)).tolist()
    names = meanings.split()
    if len(values) != len(names):
        return None
    return {float(v): n for v, n in zip(values, names)}


def _classes(classes: Any, uniques: Optional[np.ndarray], dtype: np.dtype) -> List[Tuple[float, str, str]]:
    """(value, label, colour) per class, from ``classes=`` or the values found."""
    if isinstance(classes, dict):
        items = []
        for i, (value, spec) in enumerate(classes.items()):
            if isinstance(spec, (tuple, list)) and len(spec) == 2:
                text, color = spec
            else:
                text, color = spec, _PALETTE[i % len(_PALETTE)]
            items.append((float(value), str(text), color))
        return items
    if classes is not None:
        values = list(classes)
    elif dtype == bool:
        values = [0, 1]
    else:
        values = [] if uniques is None else uniques.tolist()
    if len(values) > MAX_CLASSES:
        raise ValueError(f"{len(values)} classes: too many to tell apart; use kind='continuous'")
    labels = {0: "False", 1: "True"} if dtype == bool else {}
    return [(float(v), labels.get(int(v), _fmt_class(v)), _PALETTE[i % len(_PALETTE)]) for i, v in enumerate(values)]


def _fmt_class(value: Any) -> str:
    return str(int(value)) if float(value).is_integer() else f"{value:g}"
