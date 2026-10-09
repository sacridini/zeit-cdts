"""Dates of a cube: parsed from band names and file names, and converted to the
time units each algorithm works in (years, ordinal days, fractional years)."""

import re
from datetime import datetime
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Band descriptions and file names, most specific first. Each pattern captures the
# date in a group named "date", parsed with the format next to it.
_DATE_PATTERNS = [
    (re.compile(r"(?<!\d)(?P<date>\d{4}-\d{2}-\d{2})(?!\d)"), "%Y-%m-%d"),
    (re.compile(r"(?<!\d)(?P<date>\d{4}_\d{2}_\d{2})(?!\d)"), "%Y_%m_%d"),
    (re.compile(r"(?<!\d)(?P<date>\d{4}\.\d{2}\.\d{2})(?!\d)"), "%Y.%m.%d"),
    (re.compile(r"(?<!\d)(?P<date>\d{8})(?!\d)"), "%Y%m%d"),
    (re.compile(r"(?<!\d)(?P<date>\d{7})(?!\d)"), "%Y%j"),
]
_YEAR = re.compile(r"(?<!\d)(?P<date>(?:1[89]|2[01])\d{2})(?!\d)")
# A band description that is only a year: "1985", "yr1985", "year_1985", "y1985".
_YEAR_LABEL = re.compile(r"^(?:yr|year|y)?[_\- ]?(?P<date>\d{4})$", re.I)


def _strptime(text: str, fmt: str) -> Optional[pd.Timestamp]:
    try:
        stamp = datetime.strptime(text, fmt)
    except ValueError:
        return None
    if not 1800 <= stamp.year <= 2199:
        return None
    return pd.Timestamp(stamp)


def find_date(text: str, date_format: Optional[str] = None) -> Optional[Tuple[pd.Timestamp, str]]:
    """The first date in a file name or band description, and the text left once
    the date and its separator are removed (``"2020-01-15_red"`` -> ``"red"``).

    ``date_format`` (strptime) parses the whole text or the first run of characters
    that matches it; without it the usual layouts are tried, then a lone year.
    """
    text = str(text).strip()
    if date_format is not None:
        stamp = _strptime(text, date_format)
        if stamp is not None:
            return stamp, ""
        pattern = re.escape(date_format)
        for code, regex in (("%Y", r"\d{4}"), ("%m", r"\d{2}"), ("%d", r"\d{2}"), ("%j", r"\d{3}"),
                            ("%y", r"\d{2}"), ("%H", r"\d{2}"), ("%M", r"\d{2}"), ("%S", r"\d{2}")):
            pattern = pattern.replace(re.escape(code), regex)
        match = re.search(pattern, text)
        if match is None:
            return None
        stamp = _strptime(match.group(0), date_format)
        return (stamp, _rest(text, match)) if stamp is not None else None

    match = _YEAR_LABEL.match(text)
    if match:
        return pd.Timestamp(int(match.group("date")), 1, 1), ""
    for regex, fmt in _DATE_PATTERNS:
        for match in regex.finditer(text):
            stamp = _strptime(match.group("date"), fmt)
            if stamp is not None:
                return stamp, _rest(text, match)
    match = _YEAR.search(text)
    if match:
        return pd.Timestamp(int(match.group("date")), 1, 1), _rest(text, match)
    return None


def _rest(text: str, match: "re.Match") -> str:
    rest = text[:match.start()] + text[match.end():]
    return rest.strip("_- .")


def dates_from_labels(labels: Sequence[Any]) -> Optional[Tuple[pd.DatetimeIndex, Optional[List[str]]]]:
    """Dates of band descriptions, when every one holds a date.

    Returns ``(dates, band_names)``: ``band_names`` is ``None`` when the labels are
    dates only (``yr1985``, ``2020-01-15``), or the spectral band of each label
    when they are ``date_band`` pairs (``2020-01-15_red``).
    """
    if len(labels) == 0 or any(lab is None or str(lab).strip() == "" for lab in labels):
        return None
    found = [find_date(str(lab)) for lab in labels]
    if any(f is None for f in found):
        return None
    dates = pd.DatetimeIndex([f[0] for f in found])
    rests = [f[1] for f in found]
    if all(r == "" for r in rests):
        return (dates, None) if dates.is_unique else None
    if any(r == "" for r in rests):
        return None
    return dates, rests


def to_datetime_index(dates: Iterable[Any]) -> pd.DatetimeIndex:
    """Dates given by the user, as a DatetimeIndex.

    Integers up to 9999 are years (``1985`` -> 1985-01-01), other numbers fractional
    years (``2020.5``); anything else goes through ``pandas.to_datetime``.
    """
    values = list(np.asarray(dates).ravel()) if not isinstance(dates, pd.Index) else list(dates)
    if values and all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)
                      for v in values):
        nums = np.asarray(values, dtype=float)
        if np.all(nums == np.round(nums)) and np.all(nums <= 9999):
            return pd.DatetimeIndex([pd.Timestamp(int(y), 1, 1) for y in nums])
        return pd.DatetimeIndex([_from_fractional_year(v) for v in nums])
    return pd.DatetimeIndex(pd.to_datetime(values))


def _from_fractional_year(value: float) -> pd.Timestamp:
    year = int(np.floor(value))
    start = pd.Timestamp(year, 1, 1)
    days = (pd.Timestamp(year + 1, 1, 1) - start).days
    return start + pd.Timedelta(days=(value - year) * days)


def years(time: Any) -> np.ndarray:
    """Calendar year of each date."""
    return pd.DatetimeIndex(np.asarray(time)).year.to_numpy().astype(np.int64)


def ordinal_days(time: Any) -> np.ndarray:
    """Proleptic Gregorian ordinal of each date (``datetime.toordinal``), the time
    unit of CCDC/COLD."""
    return np.array([d.toordinal() for d in pd.DatetimeIndex(np.asarray(time))], dtype=np.int64)


def fractional_years(time: Any) -> np.ndarray:
    """Each date as a decimal year (2020-07-02 -> ~2020.5)."""
    idx = pd.DatetimeIndex(np.asarray(time))
    days_in_year = np.where(idx.is_leap_year, 366.0, 365.0)
    return idx.year.to_numpy() + (idx.dayofyear.to_numpy() - 1) / days_in_year
