"""The cases of the field-boundary sensitivity analysis (phase 24e): one per mosaic window and reference.

A case is the Landsat medoid of a window (6 bands, int16, reflectance x 10000), the reference rasterised onto
its grid through ``zeit``'s own rasteriser (the GeoDataFrame path of ``segmentation_accuracy``), the region
the reference speaks about and the kind of reference (``fields`` or ``property``).

Groups and references:

* ``chaco`` (2020), ``humidchaco`` (2022): the Trazo annotations; they are complete only inside the chip
  footprints, so the region is the union of the footprints that fall in the window.
* ``lem``: the Fields of The World fields of Luis Eduardo Magalhaes; no completeness claim, so the region is
  the default one (around the reference fields).
* ``sorriso``: the CAR properties (``property``) and the 2019 pivots (``fields``, default region).
* ``porto_murtinho``: the CAR properties (``property``).

Data: ``ZEIT_FIELDS_DATA`` (default ``C:\\Users\\eduar\\Documents\\dev\\zeit_data\\fields``).
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

DATA = Path(os.environ.get("ZEIT_FIELDS_DATA", r"C:\Users\eduar\Documents\dev\zeit_data\fields"))
OUT = DATA / "sensitivity"
TOLERANCES = (0, 1, 2)
BANDS = ("blue", "green", "red", "nir08", "swir16", "swir22")
SWIR = BANDS.index("swir16")
MASK_THRESHOLDS = (1500, 2000, 2500)


@dataclass
class Case:
    name: str            # mosaic window, e.g. trazo_chaco_007
    ref: str             # reference name: trazo, ftw, car, pivots
    group: str           # chaco, humidchaco, lem, sorriso, porto_murtinho
    kind: str            # "fields" or "property"
    year: int
    transform: object
    crs: object
    bands: np.ndarray    # (6, h, w) int16
    valid: np.ndarray    # (h, w) bool, False where any band is NoData
    ref_labels: np.ndarray  # (h, w) int64, 0 = none
    regions: Dict[float, Optional[np.ndarray]] = field(default_factory=dict)  # tolerance -> region (None: default)
    info: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"{self.name}:{self.ref}"

    @property
    def shape(self):
        return self.valid.shape

    @property
    def img(self) -> np.ndarray:
        """The notebook's B5 (swir16), NoData filled with the median so a percentile stretch is not distorted."""
        v = self.bands[SWIR].copy()
        if not self.valid.all():
            v[~self.valid] = int(np.median(v[self.valid])) if self.valid.any() else 0
        return v

    @property
    def stack(self) -> np.ndarray:
        """(6, h, w) float64, NaN at NoData (what the multitemporal gradient takes)."""
        out = self.bands.astype(np.float64)
        out[:, ~self.valid] = np.nan
        return out

    def keep(self, threshold: Optional[float]) -> Optional[np.ndarray]:
        """The mask (True = flood): ``swir16 > threshold`` and valid; None = no mask (valid only if any NoData)."""
        if threshold is None:
            return None if self.valid.all() else self.valid.copy()
        return (self.bands[SWIR] > threshold) & self.valid


def _rasterise(gdf, shape, transform, crs):
    from zeit._segacc import _rasterise as r

    return r(gdf, shape, transform, crs)


def _region_for(case_kind: str, ref: np.ndarray, tol: float, explicit: Optional[np.ndarray]):
    """The region at ``tol``: the explicit one, or the default of ``segmentation_accuracy`` (made here once)."""
    if explicit is not None:
        return explicit
    has = ref > 0
    if case_kind == "fields" and tol > 0 and has.any():
        from scipy.ndimage import distance_transform_edt

        return distance_transform_edt(~has) <= tol
    return has.copy()


def _read(name: str):
    import rasterio

    with rasterio.open(DATA / "mosaics" / f"{name}.tif") as src:
        arr = src.read()
        nodata = src.nodata
        transform, crs = src.transform, src.crs
    valid = np.all(arr != nodata, axis=0) if nodata is not None else np.ones(arr.shape[1:], bool)
    return arr, valid, transform, crs


def index() -> List[dict]:
    return json.loads((DATA / "mosaics" / "index.json").read_text())["mosaics"]


def windows() -> Dict[str, dict]:
    return {w["name"]: w for w in json.loads((DATA / "mosaics" / "windows.json").read_text())}


def load_cases(groups=None, only=None, max_per_group: Optional[int] = None) -> List[Case]:
    """All the cases (or those of ``groups`` / named ``only``). ``max_per_group`` keeps an evenly spread subset."""
    import geopandas as gpd

    wins = windows()
    idx = [m for m in index() if (groups is None or m["group"] in groups) and (only is None or m["name"] in only)]
    by_group: Dict[str, list] = {}
    for m in idx:
        by_group.setdefault(m["group"], []).append(m)
    if max_per_group:
        for g, ms in by_group.items():
            if len(ms) > max_per_group:
                sel = np.unique(np.linspace(0, len(ms) - 1, max_per_group).round().astype(int))
                by_group[g] = [ms[i] for i in sel]

    refs = {}   # reference layers read once per file

    def layer(path, name=None):
        key = (str(path), name)
        if key not in refs:
            refs[key] = gpd.read_file(path, layer=name) if name else gpd.read_file(path)
        return refs[key]

    cases: List[Case] = []
    for g, ms in by_group.items():
        for m in ms:
            arr, valid, transform, crs = _read(f"{m['name']}_{m['year']}")
            shape = valid.shape
            w = wins[m["name"]]
            bx0, by0, bx1, by1 = w["bounds"]
            from shapely.geometry import box

            def clip(gdf):
                gdf = gdf.to_crs(crs)
                return gdf[gdf.intersects(box(bx0, by0, bx1, by1))]

            def make(ref, kind, gdf, explicit=None, info=None):
                lab = _rasterise(gdf, shape, transform, crs) if len(gdf) else np.zeros(shape, np.int64)
                regs = {t: _region_for(kind, lab, t, explicit) for t in TOLERANCES}
                cases.append(Case(m["name"], ref, g, kind, m["year"], transform, crs, arr, valid, lab, regs,
                                  {"n_polygons": int(len(gdf)), **(info or {})}))

            if g in ("chaco", "humidchaco"):
                p = DATA / "trazo" / f"trazo_{g}.gpkg"
                ann = clip(layer(p, "annotations"))
                foot = clip(layer(p, "chip_footprints"))
                # the footprints of this window (those the cluster is made of), as a raster region
                reg = _rasterise(foot.assign(geometry=foot.geometry), shape, transform, crs) > 0
                make("trazo", "fields", ann, reg, {"n_footprints": int(len(foot))})
            elif g == "lem":
                make("ftw", "fields", clip(layer(DATA / "ftw_brazil_2020.gpkg")))
            elif g == "sorriso":
                make("car", "property", clip(layer(DATA / "car_sorriso.gpkg")))
                make("pivots", "fields", clip(layer(DATA / "pivots_2019.gpkg")))
            elif g == "porto_murtinho":
                make("car", "property", clip(layer(DATA / "car_porto_murtinho.gpkg")))
    return cases


def mask_shares(case: Case) -> List[dict]:
    """The masked share (``swir16 <= threshold``) of the window, of the region and of the reference pixels."""
    s = case.bands[SWIR]
    region = case.regions[1]
    has = case.ref_labels > 0
    rows = []
    for thr in MASK_THRESHOLDS:
        masked = (s <= thr) | ~case.valid
        rows.append({
            "case": case.id, "group": case.group, "kind": case.kind, "threshold": thr,
            "masked_window": float(masked.mean()),
            "masked_region": float(masked[region].mean()) if region.any() else np.nan,
            "masked_reference": float(masked[has].mean()) if has.any() else np.nan,
            "region_pixels": int(region.sum()), "reference_pixels": int(has.sum()),
        })
    return rows
