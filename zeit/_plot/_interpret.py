"""``zeit.interpret``: label reference points by eye, one after the other, for ``zeit.accuracy``.

The interpreter goes through a queue of points. For each one the viewer centres the map on
it, shows the pixel's series (click it to date the change) and a strip of image chips
around the point, one per year, and records the class, the date, a confidence and a note.
Every label is saved to a file at once, so the work survives a closed window or a restarted
kernel, and calling ``interpret`` again with the same file resumes it.

It is the viewer of ``zeit.plot`` with another layout: the same ``Session`` (here an
``InterpretSession``, with four more requests) behind the same widget and window transports.

Requests, besides the viewer's:

- ``points``: the queue (cell of each point, its state), the classes and the progress;
- ``point`` (id): the point's series, chips, and -- once labelled, or with ``blind=False``
  -- what the map and the algorithm's fit say there;
- ``label`` (id, ref, ref_date, confidence, note, status): record a label (and save it);
- ``review``: the error matrix and accuracies of the labels so far, with the points of
  each cell of the matrix.
"""

import datetime as _dt
import getpass
import os
import pathlib
import re
import tempfile
import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import xarray as xr

from ._data import Frames
from ._session import Reply, Session

HERE = pathlib.Path(__file__).parent

LABEL_COLUMNS = ["ref", "ref_date", "confidence", "note", "status", "interpreter", "labelled_at"]
STATUSES = ("todo", "done", "skipped")
CONFIDENCE = ("low", "medium", "high")
FORMATS = {".gpkg": "GPKG", ".geojson": "GeoJSON", ".json": "GeoJSON", ".parquet": None}


# ---------------------------------------------------------------------------
# Points and their file
# ---------------------------------------------------------------------------

def _read(path: pathlib.Path):
    import geopandas as gpd

    if path.suffix.lower() == ".parquet":
        return gpd.read_parquet(path)
    return gpd.read_file(path)


def _write(gdf, path: pathlib.Path) -> None:
    """Write the points to ``path`` through a temporary file, so a crash never leaves half a file."""
    suffix = path.suffix.lower()
    if suffix not in FORMATS:
        raise ValueError(f"save= must end in one of {sorted(FORMATS)}, got {path.name!r}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f"{path.stem}_tmp", suffix=suffix, dir=str(path.parent))
    os.close(fd)
    try:
        if suffix == ".parquet":
            gdf.to_parquet(tmp)
        else:
            os.remove(tmp)   # some drivers refuse to write over an existing (empty) file
            layer = {"layer": re.sub(r"\W", "_", path.stem) or "points"} if suffix == ".gpkg" else {}
            gdf.to_file(tmp, driver=FORMATS[suffix], **layer)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def _prepare_points(samples: Any, save: Optional[pathlib.Path]):
    """The points with the label columns, resumed from ``save`` when it exists."""
    import geopandas as gpd

    if save is not None and save.exists():
        gdf = _read(save)
        if samples is not None:
            n = len(samples) if hasattr(samples, "__len__") else None
            if n is not None and n != len(gdf):
                warnings.warn(f"{save.name} has {len(gdf)} points and samples= has {n}: resuming from the file",
                              stacklevel=3)
    elif samples is None:
        raise ValueError("give the points to label (samples=), or save= a file of a session to resume")
    else:
        gdf = samples if isinstance(samples, gpd.GeoDataFrame) else _read(pathlib.Path(samples))
        gdf = gdf.copy()
    if not (gdf.geometry.geom_type == "Point").all():
        raise ValueError("the samples must be points")
    gdf = gdf.reset_index(drop=True)
    for column in LABEL_COLUMNS:
        if column not in gdf.columns:
            gdf[column] = None
    gdf["ref_date"] = gdf["ref_date"].map(_iso_date)
    status = gdf["status"].astype(object).where(gdf["status"].notna(), None)
    has_ref = gdf["ref"].notna() & (gdf["ref"].astype(str).str.strip() != "")
    gdf["status"] = [s if s in STATUSES else ("done" if r else "todo") for s, r in zip(status, has_ref)]
    for column in ("ref", "confidence", "note", "interpreter", "labelled_at"):
        gdf[column] = gdf[column].astype(object).where(gdf[column].notna(), None)
    return gdf


def _iso_date(value: Any) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value is pd.NaT:
        return None
    try:
        stamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return None
    return None if pd.isna(stamp) else stamp.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

class InterpretSession(Session):
    """A viewer Session that also serves a queue of points and records their labels."""

    def __init__(self, data: Any, points, *, classes: Sequence[str], save: Optional[pathlib.Path] = None,
                 series: Optional[xr.DataArray] = None, chips: Any = "year", chip_size: int = 33,
                 fit: Any = None, map_info: Optional[Dict[str, Any]] = None, review: Any = None,
                 blind: bool = True, interpreter: Optional[str] = None, zoom: int = 120, **kwargs):
        super().__init__(data, **kwargs)
        self.handlers.update({"points": self._points, "point": self._point, "label": self._label,
                              "review": self._review})
        self.points = points
        self.classes = [str(c) for c in classes]
        self.save_path = save
        self.series_da = series
        self.chips = chips
        self.chip_size = max(5, int(chip_size) | 1)   # odd: the point in the middle
        self.point_fit = fit
        self.map_info = map_info
        self.review_fn = review
        self.blind = bool(blind)
        self.interpreter = interpreter
        self.zoom = int(zoom)
        self._locate()

    # ------------------------------------------------------------------ setup
    def _locate(self) -> None:
        """Each point's cell in the frames' grid (``top``: counted from the top row)."""
        from .._warp import transform_of

        f = self.frames
        da = f.da
        pts = self.points
        crs = None
        try:
            crs = da.rio.crs
        except Exception:  # noqa: BLE001
            crs = None
        if crs is not None and pts.crs is not None and pts.crs != crs:
            pts = pts.to_crs(crs)
        if "x" in da.coords and f.width > 1 and f.height > 1:
            t = transform_of(da)
            cols, rows = ~t * (pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy())
        else:
            cols, rows = pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()
        self.rows = np.floor(rows).astype(np.int64)
        self.cols = np.floor(cols).astype(np.int64)
        self.world = np.column_stack([pts.geometry.x.to_numpy(), pts.geometry.y.to_numpy()])
        self.inside = (self.rows >= 0) & (self.rows < f.height) & (self.cols >= 0) & (self.cols < f.width)
        if not self.inside.all():
            warnings.warn(f"{int((~self.inside).sum())} points fall outside the data and are left out of the "
                          "queue", stacklevel=4)
        self.tops = np.where(self.flip, f.height - 1 - self.rows, self.rows)

    # ------------------------------------------------------------------ state
    def _state(self, i: int) -> Dict[str, Any]:
        row = self.points.iloc[i]
        out = {"id": int(i), "col": int(self.cols[i]), "top": int(self.tops[i]), "status": row["status"],
               "ref": row["ref"], "ref_date": row["ref_date"], "confidence": row["confidence"],
               "note": row["note"]}
        if not self.blind and "stratum_name" in self.points.columns:
            out["stratum"] = str(row["stratum_name"])
        return out

    def progress(self) -> Dict[str, int]:
        status = self.points.loc[self.inside, "status"]
        return {"done": int((status == "done").sum()), "skipped": int((status == "skipped").sum()),
                "total": int(self.inside.sum())}

    def review_ready(self) -> bool:
        if self.map_info is None:
            return False
        if not self.blind:
            return True
        return bool((self.points.loc[self.inside, "status"] != "todo").all())

    def _visible(self, i: int) -> bool:
        return not self.blind or self.points.at[i, "status"] != "todo"

    # ------------------------------------------------------------------ requests
    def _points(self, request: Dict[str, Any]) -> Reply:
        ids = [int(i) for i in np.flatnonzero(self.inside)]
        return {"points": [self._state(i) for i in ids], "classes": self.classes, "blind": self.blind,
                "has_map": self.map_info is not None, "review_ready": self.review_ready(),
                "progress": self.progress(), "zoom": self.zoom, "saving": str(self.save_path or ""),
                "chips": bool(self.chips)}, []

    def _point(self, request: Dict[str, Any]) -> Reply:
        from ._fit import overlays

        i = int(request["id"])
        if not (0 <= i < len(self.points)) or not self.inside[i]:
            raise ValueError(f"no point {i}")
        series = self._series_at(i)
        visible = self._visible(i)
        series["overlays"] = overlays(self.point_fit, series, shape=(self.frames.height, self.frames.width),
                                      band=self.band) if (visible and self.point_fit is not None) else []
        content = {"state": self._state(i), "series": series, "map": None, "fit_hidden":
                   self.point_fit is not None and not visible}
        if self.map_info is not None:
            content["map"] = self.map_info["labels"][i] if visible else "hidden until labelled"
        buffers: List[bytes] = []
        if self.chips:
            chips, data = self._chips(i)
            content["chips"] = chips
            buffers.append(data)
        return content, buffers

    def _label(self, request: Dict[str, Any]) -> Reply:
        i = int(request["id"])
        if not (0 <= i < len(self.points)) or not self.inside[i]:
            raise ValueError(f"no point {i}")
        status = request.get("status", "done")
        if status not in STATUSES:
            raise ValueError(f"status must be one of {STATUSES}, got {status!r}")
        ref = request.get("ref")
        if status == "done":
            if ref is None or str(ref) not in self.classes:
                raise ValueError(f"choose a class among {self.classes}")
        confidence = request.get("confidence")
        if confidence is not None and confidence not in CONFIDENCE:
            raise ValueError(f"confidence must be one of {CONFIDENCE}, got {confidence!r}")
        p = self.points
        p.at[i, "status"] = status
        p.at[i, "ref"] = str(ref) if (ref is not None and status == "done") else None
        p.at[i, "ref_date"] = _iso_date(request.get("ref_date")) if status == "done" else None
        p.at[i, "confidence"] = confidence if status == "done" else None
        note = request.get("note")
        p.at[i, "note"] = str(note) if note else None
        p.at[i, "interpreter"] = self.interpreter
        p.at[i, "labelled_at"] = _dt.datetime.now().isoformat(timespec="seconds") if status != "todo" else None
        self.save()
        return {"point": self._state(i), "progress": self.progress(), "review_ready": self.review_ready()}, []

    def save(self) -> None:
        if self.save_path is not None:
            _write(self.points, self.save_path)

    def _review(self, request: Dict[str, Any]) -> Reply:
        if self.map_info is None:
            return {"available": False, "reason": "give the map to assess (map=) to review its accuracy"}, []
        if not self.review_ready():
            todo = int((self.points.loc[self.inside, "status"] == "todo").sum())
            return {"available": False, "reason": f"label every point first ({todo} to go): the review shows "
                                                  "what the map says, which would bias the labels"}, []
        done = (self.points["status"] == "done").to_numpy() & self.inside
        if not done.any():
            return {"available": False, "reason": "no point labelled yet"}, []
        labels = np.asarray(self.map_info["plain"], dtype=object)
        refs = self.points["ref"].to_numpy(dtype=object)
        classes = list(dict.fromkeys(list(self.map_info["classes"]) + self.classes))
        counts, cells = [], {}
        for a, m in enumerate(classes):
            row = []
            for b, r in enumerate(classes):
                ids = np.flatnonzero(done & (labels == m) & (refs == r))
                row.append(int(len(ids)))
                if len(ids):
                    cells[f"{a},{b}"] = [int(k) for k in ids]
            counts.append(row)
        content = {"available": True, "classes": classes, "counts": counts, "cells": cells, "accuracy": None}
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                acc = self.review_fn(self.points[done].reset_index(drop=True))
            content["warnings"] = [str(w.message) for w in caught]
            content["accuracy"] = _accuracy_json(acc)
        except Exception as err:  # noqa: BLE001 - the matrix is still useful
            content["error"] = f"{type(err).__name__}: {err}"
        return content, []

    # ------------------------------------------------------------------ the point's data
    def _series_at(self, i: int) -> Dict[str, Any]:
        from ._fit import _ms, clean

        wx, wy = (float(v) for v in self.world[i])
        if self.series_da is None:
            from ._fit import pixel_series
            series = pixel_series(self.frames, int(self.cols[i]), int(self.rows[i]))
        else:
            da = self.series_da
            col = int(np.abs(da.x.values - wx).argmin()) if "x" in da.coords else int(self.cols[i])
            row = int(np.abs(da.y.values - wy).argmin()) if "y" in da.coords else int(self.rows[i])
            at = da.isel(y=row, x=col)
            values = np.asarray(at.values, dtype=float)
            if "band" in at.dims:
                at = at.transpose("time", "band")
                values = np.asarray(at.values, dtype=float)
                names = [str(b) for b in at.band.values]
                lines = [{"name": names[k], "y": clean(values[:, k])} for k in range(values.shape[1])]
            else:
                lines = [{"name": str(da.name or "value"), "y": clean(values)}]
            times = pd.DatetimeIndex(da.time.values)
            series = {"x": _ms(times), "series": lines, "labels": [t.strftime("%Y-%m-%d") for t in times],
                      "is_time": True, "world": [wx, wy], "cell": [col, row]}
        series["world"] = [wx, wy]
        series["cell_top"] = [int(self.cols[i]), int(self.tops[i])]
        return series

    def _chips(self, i: int):
        """Image chips around point ``i``: one per year (or per date), as RGBA bytes."""
        f = self.frames
        half = self.chip_size // 2
        r, c = int(self.rows[i]), int(self.cols[i])
        r0, r1 = max(0, r - half), min(f.height, r + half + 1)
        c0, c1 = max(0, c - half), min(f.width, c + half + 1)
        window = f.da.isel(y=slice(r0, r1), x=slice(c0, c1))
        values = np.asarray(window.values, dtype=np.float32)   # (frames..., [band], y, x)
        lead = values.shape[:len(f.lead)]
        values = values.reshape((int(np.prod(lead)) if lead else 1,) + values.shape[len(f.lead):])
        # beyond the edge of the data: NoData, so every chip is as large and the point in its middle
        pad = [(0, 0)] * (values.ndim - 2) + [(half - (r - r0), half - (r1 - 1 - r)),
                                               (half - (c - c0), half - (c1 - 1 - c))]
        values = np.pad(values, pad, constant_values=np.nan)
        groups = _chip_groups(f, self.chips)
        stack = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN groups
            for members in groups["members"]:
                stack.append(np.nanmedian(values[members], axis=0) if len(members) > 1 else values[members[0]])
        lut = self.style.lut()
        out = []
        for chip in stack:
            if f.rgb:
                chip = np.moveaxis(chip, 0, -1)
            codes = self.style.encode(chip, self.nodata)
            if self.flip:
                codes = codes[::-1]
            if f.rgb:
                rgba = np.zeros(codes.shape[:2] + (4,), dtype=np.uint8)
                rgba[..., :3] = ((codes.astype(np.float32) - 1).clip(0) * (255.0 / 254.0)).astype(np.uint8)
                rgba[..., 3] = np.where((codes == 0).any(axis=-1), 0, 255)
            else:
                rgba = lut[codes]
            out.append(np.ascontiguousarray(rgba))
        h, w = (out[0].shape[:2] if out else (0, 0))
        content = {"labels": groups["labels"], "first": [int(m[0]) for m in groups["members"]],
                   "members": [[int(k) for k in m] for m in groups["members"]], "width": int(w), "height": int(h),
                   "center": [half, half], "size": self.chip_size}
        return content, b"".join(chip.tobytes() for chip in out)


def _chip_groups(frames: Frames, chips: Any) -> Dict[str, Any]:
    """Which frames each chip is the median of, and its label."""
    n = frames.n
    if chips in ("date", True) or frames.times is None:
        step = 1 if chips in ("date", True, "year") else max(1, int(chips))
        members = [[k] for k in range(0, n, step)]
        return {"members": members, "labels": [frames.labels[m[0]] for m in members]}
    if isinstance(chips, (int, np.integer)) and not isinstance(chips, bool):
        members = [list(range(k, min(n, k + int(chips)))) for k in range(0, n, int(chips))]
        return {"members": members, "labels": [frames.labels[m[0]] for m in members]}
    if chips not in ("year", "month"):
        raise ValueError(f"chips must be 'year', 'month', 'date', a number of frames or None, got {chips!r}")
    keys = frames.times.year if chips == "year" else frames.times.to_period("M").astype(str)
    members, labels = [], []
    for key in pd.unique(keys):
        members.append([int(k) for k in np.flatnonzero(keys == key)])
        labels.append(str(key))
    return {"members": members, "labels": labels}


def _accuracy_json(acc) -> Dict[str, Any]:
    def rows(df, cols):
        return [[None if not np.isfinite(v) else float(v) for v in df.loc[c, cols]] for c in df.index]

    return {"overall": [float(v) for v in acc.overall[["estimate", "ci_low", "ci_high"]]],
            "classes": [str(c) for c in acc.users.index],
            "users": rows(acc.users, ["estimate", "ci_low", "ci_high"]),
            "producers": rows(acc.producers, ["estimate", "ci_low", "ci_high"]),
            "area": rows(acc.area, ["mapped", "estimate", "ci"]), "unit": acc.area_unit, "n": acc.n,
            "confidence": acc.confidence}


# ---------------------------------------------------------------------------
# What interpret returns
# ---------------------------------------------------------------------------

class Interpretation:
    """A labelling session: its points (``samples``), progress, and the accuracy of a map.

    In a notebook the viewer is shown when ``zeit.interpret`` is called and this object
    follows it live; outside one, ``zeit.interpret`` returns it when the window is closed.
    """

    def __init__(self, session: InterpretSession, map_data: Any = None, strata: Any = None):
        self.session = session
        self._map, self._strata = map_data, strata
        self.view = None   # the widget or the window

    @property
    def samples(self):
        """The points with their labels (``ref``, ``ref_date``, ``confidence``, ``note``,
        ``status``, ``interpreter``, ``labelled_at``), in their own CRS."""
        return self.session.points.copy()

    @property
    def progress(self) -> Dict[str, int]:
        """Points labelled (``done``), skipped and in the queue (``total``)."""
        return self.session.progress()

    def accuracy(self, map: Any = None, **kwargs: Any):  # noqa: A002 - the name says what it is
        """``zeit.accuracy`` of a map (default: the ``map=`` given to ``interpret``) from the
        labelled points; ``strata`` defaults to the ``strata=`` given to ``interpret``."""
        from .._accuracy import accuracy

        target = map if map is not None else self._map
        if target is None:
            raise ValueError("give the map to assess: interpretation.accuracy(map)")
        if self._strata is not None:
            kwargs.setdefault("strata", self._strata)
        done = self.session.points[self.session.points.status == "done"].reset_index(drop=True)
        return accuracy(target, done, **kwargs)

    def save(self, path: Union[str, os.PathLike, None] = None) -> pathlib.Path:
        """Write the points and labels (to ``save=`` by default)."""
        target = pathlib.Path(path) if path is not None else self.session.save_path
        if target is None:
            raise ValueError("give a path: interpretation.save('labels.gpkg')")
        _write(self.session.points, target)
        return target

    def close(self) -> None:
        """Close the window (outside a notebook)."""
        if self.view is not None and hasattr(self.view, "close"):
            self.view.close()

    def __repr__(self) -> str:
        p = self.progress
        where = f", saved to {self.session.save_path}" if self.session.save_path else ""
        return f"<zeit.interpret: {p['done']} of {p['total']} points labelled, {p['skipped']} skipped{where}>"


# ---------------------------------------------------------------------------
# zeit.interpret
# ---------------------------------------------------------------------------

def interpret(
    data: Any,
    samples: Any = None,
    *,
    classes: Optional[Sequence[str]] = None,
    save: Union[str, os.PathLike, None] = None,
    rgb: Optional[Sequence[str]] = None,
    band: Any = None,
    series: Any = None,
    chips: Any = "year",
    chip_size: int = 33,
    fit: Any = None,
    map: Any = None,  # noqa: A002 - the name says what it is
    strata: Any = None,
    blind: bool = True,
    interpreter: Optional[str] = None,
    basemap: Any = None,
    zoom: int = 120,
    height: int = 480,
    block: Optional[bool] = None,
    open: Union[bool, str] = True,  # noqa: A002 - as in show_window
) -> Interpretation:
    """Label reference points by eye, one after the other: the reference for ``zeit.accuracy``.

    The viewer goes through the points: it centres the map on each one, shows its series (a
    click on it dates the change) and a strip of chips around the point, one per year, and
    records the class (keys 1-9), the date, a confidence and a note. Enter saves and goes to
    the next point. Each label is written to ``save`` at once; calling ``interpret`` again
    with the same file resumes where it stopped.

    Parameters
    ----------
    data
        The imagery to interpret: a ``(time, band, y, x)`` cube (``rgb`` picks the bands of
        the image), a ``(time, y, x)`` index, or a raster path.
    samples
        The points (``GeoDataFrame`` or vector file), e.g. from ``zeit.stratified_sample``.
        Not needed to resume from ``save``.
    classes
        Names of the reference classes (keys 1-9 pick them). Default: the classes of ``map``.
    save
        File the labels are written to after each one (``.gpkg``, ``.geojson`` or
        ``.parquet``). When it exists, the session resumes from it.
    rgb
        Three bands of ``data`` for the image, e.g. ``["swir1", "nir", "red"]`` (default:
        red, green and blue when the cube has them).
    band
        One band of ``data`` to show instead of an RGB image.
    series
        What the chart shows: band name(s) of ``data`` (e.g. ``"nbr"``), or a ``(time, y,
        x)`` cube on the same area. Default: the bands of the image.
    chips
        One chip per ``"year"`` (the median of its dates), ``"month"``, ``"date"``, every
        ``n`` frames, or ``None`` for no chips.
    chip_size
        Side of a chip, in pixels.
    fit
        A result of zeit (``landtrendr``, ``ccdc``, ``extract_events``...) drawn over the
        series, as in ``zeit.plot``.
    map
        The map being assessed (classes, or events). Its value at the point is shown, and
        the review tab gives its accuracy.
    strata
        The map the points were stratified by, when it is not ``map`` (see ``zeit.accuracy``).
    blind
        Hide what ``map`` and ``fit`` say until the point is labelled, and the review until
        every point is: the reference should not be swayed by the map.
    interpreter
        Name recorded with each label (default: the user's login).
    basemap
        A web map under the data, as in ``zeit.plot``.
    zoom
        How many pixels across the map shows around a point.
    height
        Height of the map, in pixels (notebook).
    block, open
        Outside a notebook, as in ``zeit.plot``: wait for the window to close (default: yes
        in a script), and how to open it (``"native"``, ``"browser"``, False to only serve it).

    Returns
    -------
    Interpretation
        ``samples`` (the points with ``ref``, ``ref_date``, ``confidence``, ``note``,
        ``status``, ``interpreter``, ``labelled_at``), ``progress`` and ``accuracy(map)``.

    Examples
    --------
    >>> points = zeit.stratified_sample(loss, n=300)
    >>> s = zeit.interpret(cube, points, classes=["no change", "change"], rgb=["swir1", "nir", "red"],
    ...                    series="nbr", map=loss, save="reference.gpkg")
    >>> acc = s.accuracy()
    """
    from . import style_for   # noqa: F401 - fail early without matplotlib
    from .._accuracy import _strata, _to_crs, _values_at, accuracy
    from .._load import load_raster

    save_path = pathlib.Path(save) if save is not None else None
    points = _prepare_points(samples, save_path)
    if isinstance(data, (str, os.PathLike)):
        data = load_raster(data, chunks="auto")
    cube = data
    rgb_option = None
    if isinstance(cube, xr.DataArray) and "band" in cube.dims:
        names = [str(b) for b in cube.band.values]
        if rgb is not None:
            missing = [b for b in rgb if str(b) not in names]
            if len(rgb) != 3 or missing:
                raise ValueError(f"rgb= takes three bands of the cube {names}" + (f"; not found: {missing}"
                                                                                     if missing else ""))
            display = cube.sel(band=list(rgb)).assign_coords(band=["red", "green", "blue"])
            rgb_option = True
        elif band is not None:
            display = cube.sel(band=band)
        else:
            from ._data import rgb_bands
            if rgb_bands(cube) is None:
                raise ValueError(f"the cube has the bands {names}: choose the image with rgb=[...] or band=")
            display = cube
            rgb_option = True
    else:
        display = cube

    series_da = None
    if series is not None:
        if isinstance(series, xr.DataArray):
            series_da = series
        else:
            if not (isinstance(cube, xr.DataArray) and "band" in cube.dims):
                raise ValueError("series= as band names needs a cube with bands; or give a (time, y, x) cube")
            series_da = cube.sel(band=series)
        if "time" not in series_da.dims:
            raise ValueError("series= needs a time dimension")

    map_info = None
    if map is not None:
        map_strata = _strata(map)
        codes, inside = _values_at(map_strata.codes, _to_crs(points, map_strata.codes))
        labels = []
        events = map_strata.events
        years = None
        if events is not None and "yod" in events:
            years, _ = _values_at(events["yod"].astype(np.int64), _to_crs(points, map_strata.codes))
        for k, (code, ok) in enumerate(zip(codes, inside)):
            text = map_strata.names.get(int(code), "outside the map") if ok and code >= 0 else "outside the map"
            if years is not None and text != "no change" and ok and years[k] > 0:
                text += f" (last year before: {int(years[k])})"
            labels.append(text)
        # "plain": the map's class alone, what the review's error matrix compares
        plain = [map_strata.names.get(int(c), "outside the map") if ok and c >= 0 else "outside the map"
                 for c, ok in zip(codes, inside)]
        map_info = {"labels": labels, "plain": plain, "classes": list(map_strata.names.values())}
        if classes is None:
            classes = list(map_strata.names.values())
    if classes is None:
        raise ValueError("give the reference classes (classes=[...]), or the map to assess (map=)")
    if len(classes) > 9:
        warnings.warn("more than 9 classes: keys 1-9 pick the first nine, the others with the mouse", stacklevel=2)

    def review(done):
        kwargs = {"strata": strata} if strata is not None else {}
        return accuracy(map, done, **kwargs)

    session = InterpretSession(
        display, points, classes=classes, save=save_path, series=series_da, chips=chips, chip_size=chip_size,
        fit=fit, map_info=map_info, review=review if map is not None else None, blind=blind,
        interpreter=interpreter or _login(), zoom=zoom, rgb=rgb_option, basemap=basemap,
        opacity=0.8 if basemap else 1.0, title="zeit.interpret")
    result = Interpretation(session, map, strata)

    from ._widget import in_notebook
    if in_notebook():
        from IPython.display import display as show
        from ._widget import make_interpret_widget
        result.view = make_interpret_widget(session, height=height)
        show(result.view)
        return result
    from ._window import show_window
    result.view = show_window(session, block=False, open=open, title="zeit.interpret", kind="interpret")
    if block is None:
        from ._window import is_interactive
        block = not is_interactive()
    if block:
        result.view.wait()
    return result


def _login() -> Optional[str]:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no login name in this environment
        return None
