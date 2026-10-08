import pytest

pytest.importorskip("ee")

from zeit.gee.main import _band_names, _stm_output_sources, _stm_reducer_plan
from zeit.indices import parse_metrics


def test_stm_reducer_plan_merges_percentiles():
    pct, simple = _stm_reducer_plan(parse_metrics(["median", "p10", "iqr", "std", "count", "p25"]))
    assert pct == [10, 25, 50, 75]
    assert simple == ["std", "count"]
    assert _stm_reducer_plan(["mean"]) == ([], ["mean"])


def test_stm_output_sources_follow_earth_engine_names():
    assert _stm_output_sources("NDVI", "median") == ["NDVI_p50"]
    assert _stm_output_sources("NDVI", "p5") == ["NDVI_p5"]
    assert _stm_output_sources("NDVI", "std") == ["NDVI_stdDev"]
    assert _stm_output_sources("SR_B4", "count") == ["SR_B4_count"]
    assert _stm_output_sources("NBR", "iqr") == ["NBR_p75", "NBR_p25"]


def test_band_names():
    assert _band_names(["ndvi", "SR_B4", "kndvi"]) == ["NDVI", "SR_B4", "kNDVI"]
    assert _band_names(None) == ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]


def test_bands_is_a_deprecated_alias_of_indices():
    from zeit.gee.main import _resolve_indices

    assert _resolve_indices(["NDVI"], None) == ["NDVI"]
    with pytest.warns(DeprecationWarning, match="indices"):
        assert _resolve_indices(None, ["NBR"]) == ["NBR"]
    with pytest.raises(TypeError):
        _resolve_indices(["NDVI"], ["NBR"])
