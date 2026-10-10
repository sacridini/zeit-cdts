"""zeit.gee.harmonization without an Earth Engine account: stand-in images record each call."""
import pytest

pytest.importorskip("ee")

import zeit.gee.harmonization as hz


class Image:
    def __init__(self, name, calls):
        self.name, self.calls = name, calls

    def _record(self, op, *args):
        self.calls.append((self.name, op, args))
        return self

    def select(self, *args): return self._record("select", *args)
    def bitwiseAnd(self, *args): return self._record("bitwiseAnd", *args)
    def eq(self, *args): return self._record("eq", *args)
    def And(self, *args): return self._record("And", *args)
    def updateMask(self, *args): return self._record("updateMask", *args)
    def toFloat(self): return self._record("toFloat")
    def multiply(self, *args): return self._record("multiply", *args)
    def add(self, *args): return self._record("add", *args)
    def addBands(self, *args, **kw): return self._record("addBands", *args)


class Collection:
    def __init__(self, images):
        self.images = images

    def filterBounds(self, roi): return self
    def filterDate(self, a, b): return self
    def merge(self, other): return Collection(self.images + other.images)
    def map(self, fn): return Collection([fn(i) for i in self.images])
    def select(self, *args): return Collection([i.select(*args) for i in self.images])
    def sort(self, key): return self


class FakeEE:
    Image = object   # the type hints inside get_harmonized_collection

    def __init__(self):
        self.calls = []

    def ImageCollection(self, name):
        return Collection([Image(name.split("/")[1], self.calls)])


def test_collection_is_not_adjusted_between_sensors(monkeypatch):
    fake = FakeEE()
    monkeypatch.setattr(hz, "ee", fake)
    col = hz.get_harmonized_collection("roi", "1985-01-01", "2025-12-31")
    assert sorted(i.name for i in col.images) == ["LC08", "LC09", "LE07", "LT05"]
    assert not [c for c in fake.calls if c[1] in ("multiply", "add", "addBands")]   # reflectance as it comes
    oli = [args for name, op, args in fake.calls if name == "LC08" and op == "select"]
    etm = [args for name, op, args in fake.calls if name == "LT05" and op == "select" and len(args) == 2]
    assert (["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"],) in oli
    assert etm == [(["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"],
                    ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"])]


def test_the_old_adjustment_warns(monkeypatch):
    monkeypatch.setattr(hz.ee, "Image", type("I", (), {"constant": staticmethod(lambda v: v)}))
    with pytest.warns(DeprecationWarning, match="2.0"):
        hz.harmonize_oli_to_etm(Image("LC08", []))
