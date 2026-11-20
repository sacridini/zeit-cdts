//! zeit._core.fusion: the Python module of STARFM and ESTARFM (zeit-core's `fusion`).

use numpy::{PyArray1, PyArray4, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::fusion::{
    self, Coarse, CoarseGeom, Common, DateRule, EstarfmParams, Fallback, Filter, Fine, Method, PairMatch, PairsMode,
    Plan, Pure, Region, Resampling, SigmaScope, SinglePair, StarfmParams, Target, VScope, Weighting,
};

fn bad<T>(what: &str, got: &str, allowed: &str) -> PyResult<T> {
    Err(PyValueError::new_err(format!("{what} must be {allowed}, got {got:?}")))
}

fn common(radius_px: usize, sigma: &str, resampling: &str, fallback: &str) -> PyResult<Common> {
    Ok(Common {
        radius_px,
        sigma: match sigma {
            "image" => SigmaScope::Image,
            "window" => SigmaScope::Window,
            s => return bad("sigma", s, "'image' or 'window'"),
        },
        resampling: match resampling {
            "nearest" => Resampling::Nearest,
            "bilinear" => Resampling::Bilinear,
            s => return bad("resampling", s, "'nearest' or 'bilinear'"),
        },
        fallback: match fallback {
            "missing" => Fallback::Missing,
            "coarse" => Fallback::Coarse,
            s => return bad("fallback", s, "'missing' or 'coarse'"),
        },
    })
}

/// A target: `(day, coarse, [(fine, coarse), ...], observed, extrapolated)`.
type TargetTuple = (i64, usize, Vec<(usize, usize)>, Option<usize>, bool);

fn plan_of(targets: Vec<TargetTuple>) -> Plan {
    Plan {
        targets: targets
            .into_iter()
            .map(|(day, coarse, pairs, observed, extrapolated)| Target {
                day,
                coarse,
                pairs,
                observed,
                extrapolated,
            })
            .collect(),
        ..Default::default()
    }
}

struct Inputs<'a> {
    fine: Fine<'a>,
    coarse: Coarse<'a>,
}

#[allow(clippy::too_many_arguments)]
fn inputs<'a>(
    fine: &'a PyReadonlyArray4<'_, f32>,
    coarse: &'a PyReadonlyArray4<'_, f32>,
    affine: Option<[f64; 6]>,
    u: Option<&'a PyReadonlyArray1<'_, f32>>,
    v: Option<&'a PyReadonlyArray1<'_, f32>>,
) -> PyResult<Inputs<'a>> {
    let fs = fine.shape();
    let cs = coarse.shape();
    let fv = fine
        .as_slice()
        .map_err(|_| PyValueError::new_err("fine must be C-contiguous"))?;
    let cv = coarse
        .as_slice()
        .map_err(|_| PyValueError::new_err("coarse must be C-contiguous"))?;
    let geom = match (affine, u, v) {
        (Some(a), None, None) => CoarseGeom::Affine(a),
        (None, Some(u), Some(v)) => CoarseGeom::Lookup {
            u: u.as_slice()
                .map_err(|_| PyValueError::new_err("u must be contiguous"))?,
            v: v.as_slice()
                .map_err(|_| PyValueError::new_err("v must be contiguous"))?,
        },
        _ => return Err(PyValueError::new_err("give either affine or u and v")),
    };
    Ok(Inputs {
        fine: Fine {
            nx: fs[3],
            ny: fs[2],
            nb: fs[1],
            values: fv,
        },
        coarse: Coarse {
            ncx: cs[3],
            ncy: cs[2],
            nb: cs[1],
            values: cv,
            geom,
        },
    })
}

/// STARFM: `fine` `(pair, band, y, x)` and `coarse` `(date, band, cy, cx)` float32; `targets` as in
/// `zeit._fusion`; returns `(prediction (target, band, ry, rx), quality (target, 2, ry, rx) uint16)`.
#[pyfunction]
#[pyo3(signature = (fine, coarse, targets, region, sigma_values, radius_px, affine = None, u = None, v = None,
    sigma = "image", resampling = "bilinear", fallback = "missing", classes = 40.0, fine_unc = 0.002,
    coarse_unc = 0.005, spatial_scale_px = None, weighting = "direct", filter = "both", pure = "epsilon",
    zero_tol = 0.5e-4, want_quality = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn starfm<'py>(
    py: Python<'py>,
    fine: PyReadonlyArray4<'py, f32>,
    coarse: PyReadonlyArray4<'py, f32>,
    targets: Vec<TargetTuple>,
    region: (usize, usize, usize, usize),
    sigma_values: Option<PyReadonlyArray1<'py, f64>>,
    radius_px: usize,
    affine: Option<[f64; 6]>,
    u: Option<PyReadonlyArray1<'py, f32>>,
    v: Option<PyReadonlyArray1<'py, f32>>,
    sigma: &str,
    resampling: &str,
    fallback: &str,
    classes: f64,
    fine_unc: f64,
    coarse_unc: f64,
    spatial_scale_px: Option<f64>,
    weighting: &str,
    filter: &str,
    pure: &str,
    zero_tol: f64,
    want_quality: bool,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray4<f32>>, Option<Bound<'py, PyArray4<u16>>>)> {
    let inp = inputs(&fine, &coarse, affine, u.as_ref(), v.as_ref())?;
    let p = StarfmParams {
        common: common(radius_px, sigma, resampling, fallback)?,
        classes,
        fine_unc,
        coarse_unc,
        spatial_scale_px: spatial_scale_px.unwrap_or(radius_px as f64),
        weighting: match weighting {
            "direct" => Weighting::Direct,
            "log" => Weighting::Log,
            s => return bad("weighting", s, "'direct' or 'log'"),
        },
        filter: match filter {
            "both" => Filter::Both,
            "either" => Filter::Either,
            "none" => Filter::None,
            s => return bad("filter", s, "'both', 'either' or 'none'"),
        },
        pure: match pure {
            "epsilon" => Pure::Epsilon,
            "tiers" => Pure::Tiers,
            "cells" => Pure::Cells,
            s => return bad("pure", s, "'epsilon', 'tiers' or 'cells'"),
        },
        zero_tol,
    };
    let plan = plan_of(targets);
    let reg = Region {
        x0: region.0,
        y0: region.1,
        nx: region.2,
        ny: region.3,
    };
    let sig = sigma_array(&inp.fine, &reg, sigma_values, n_jobs)?;
    let nt = plan.targets.len();
    let nb = inp.fine.nb;
    let out = PyArray4::<f32>::zeros(py, [nt, nb, reg.ny, reg.nx], false);
    let qual = want_quality.then(|| PyArray4::<u16>::zeros(py, [nt, 2, reg.ny, reg.nx], false));
    {
        let mut w = out.readwrite();
        let o = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let mut qw = qual.as_ref().map(|q| q.readwrite());
        let qs = match qw.as_mut() {
            Some(q) => Some(
                q.as_slice_mut()
                    .map_err(|_| PyValueError::new_err("quality not contiguous"))?,
            ),
            None => None,
        };
        py.detach(|| fusion::starfm::starfm(&inp.fine, &inp.coarse, &plan, &sig, &p, &reg, o, qs, n_jobs))
            .map_err(PyValueError::new_err)?;
    }
    Ok((out, qual))
}

fn sigma_array(fine: &Fine, reg: &Region, given: Option<PyReadonlyArray1<'_, f64>>, n_jobs: i32) -> PyResult<Vec<f64>> {
    Ok(match given {
        Some(s) => s
            .as_slice()
            .map_err(|_| PyValueError::new_err("sigma_values must be contiguous"))?
            .to_vec(),
        None => fusion::image_sigma(fine, reg, n_jobs),
    })
}

/// ESTARFM; same layout as `starfm`.
#[pyfunction]
#[pyo3(signature = (fine, coarse, targets, region, sigma_values, radius_px, affine = None, u = None, v = None,
    sigma = "image", resampling = "bilinear", fallback = "missing", classes = 4.0, v_scope = "window", alpha = 0.05,
    single_pair = "fallback", min_change = 0.02, allow_negative = false, want_quality = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn estarfm<'py>(
    py: Python<'py>,
    fine: PyReadonlyArray4<'py, f32>,
    coarse: PyReadonlyArray4<'py, f32>,
    targets: Vec<TargetTuple>,
    region: (usize, usize, usize, usize),
    sigma_values: Option<PyReadonlyArray1<'py, f64>>,
    radius_px: usize,
    affine: Option<[f64; 6]>,
    u: Option<PyReadonlyArray1<'py, f32>>,
    v: Option<PyReadonlyArray1<'py, f32>>,
    sigma: &str,
    resampling: &str,
    fallback: &str,
    classes: f64,
    v_scope: &str,
    alpha: f64,
    single_pair: &str,
    min_change: f64,
    allow_negative: bool,
    want_quality: bool,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray4<f32>>, Option<Bound<'py, PyArray4<u16>>>)> {
    let inp = inputs(&fine, &coarse, affine, u.as_ref(), v.as_ref())?;
    let p = EstarfmParams {
        common: common(radius_px, sigma, resampling, fallback)?,
        classes,
        v_scope: match v_scope {
            "window" => VScope::Window,
            "cell" => VScope::Cell,
            s => return bad("v_scope", s, "'window' or 'cell'"),
        },
        alpha,
        single_pair: match single_pair {
            "fallback" => SinglePair::Fallback,
            "missing" => SinglePair::Missing,
            s => return bad("single_pair", s, "'fallback' or 'missing'"),
        },
        min_change,
        allow_negative,
    };
    let plan = plan_of(targets);
    let reg = Region {
        x0: region.0,
        y0: region.1,
        nx: region.2,
        ny: region.3,
    };
    let sig = sigma_array(&inp.fine, &reg, sigma_values, n_jobs)?;
    let nt = plan.targets.len();
    let nb = inp.fine.nb;
    let out = PyArray4::<f32>::zeros(py, [nt, nb, reg.ny, reg.nx], false);
    let qual = want_quality.then(|| PyArray4::<u16>::zeros(py, [nt, 2, reg.ny, reg.nx], false));
    {
        let mut w = out.readwrite();
        let o = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let mut qw = qual.as_ref().map(|q| q.readwrite());
        let qs = match qw.as_mut() {
            Some(q) => Some(
                q.as_slice_mut()
                    .map_err(|_| PyValueError::new_err("quality not contiguous"))?,
            ),
            None => None,
        };
        py.detach(|| fusion::estarfm::estarfm(&inp.fine, &inp.coarse, &plan, &sig, &p, &reg, o, qs, n_jobs))
            .map_err(PyValueError::new_err)?;
    }
    Ok((out, qual))
}

/// Population standard deviation per `(pair, band)` over `region` of `fine` `(pair, band, y, x)`.
#[pyfunction]
#[pyo3(signature = (fine, region, n_jobs = -1))]
fn image_sigma<'py>(
    py: Python<'py>,
    fine: PyReadonlyArray4<'py, f32>,
    region: (usize, usize, usize, usize),
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let s = fine.shape();
    let f = Fine {
        nx: s[3],
        ny: s[2],
        nb: s[1],
        values: fine
            .as_slice()
            .map_err(|_| PyValueError::new_err("fine must be C-contiguous"))?,
    };
    let r = Region {
        x0: region.0,
        y0: region.1,
        nx: region.2,
        ny: region.3,
    };
    if r.x0 + r.nx > f.nx || r.y0 + r.ny > f.ny {
        return Err(PyValueError::new_err("region outside the data extent"));
    }
    let v = py.detach(|| fusion::image_sigma(&f, &r, n_jobs));
    Ok(PyArray1::from_vec(py, v))
}

/// The plan: pairs, output dates (days since 1970-01-01). Returns
/// `(targets [(day, coarse, pairs, observed, extrapolated)], skipped [(day, why)], unpaired [(fine, why)])`.
#[pyfunction]
#[pyo3(signature = (fine_days, fine_valid, coarse_days, coarse_windows = None, method = "starfm", dates = None,
    start = None, end = None, include_fine_dates = true, keep_observed = true, exclude = Vec::new(),
    pair_match = "auto", tolerance = None, min_valid = 0.3, max_gap = None, pairs = "bracket"))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn plan(
    fine_days: Vec<i64>,
    fine_valid: Vec<f64>,
    coarse_days: Vec<i64>,
    coarse_windows: Option<Vec<(i64, i64)>>,
    method: &str,
    dates: Option<Vec<i64>>,
    start: Option<i64>,
    end: Option<i64>,
    include_fine_dates: bool,
    keep_observed: bool,
    exclude: Vec<i64>,
    pair_match: &str,
    tolerance: Option<i64>,
    min_valid: f64,
    max_gap: Option<i64>,
    pairs: &str,
) -> PyResult<(Vec<TargetTuple>, Vec<(i64, String)>, Vec<(usize, String)>)> {
    let m = match method {
        "starfm" => Method::Starfm,
        "estarfm" => Method::Estarfm,
        s => return bad("method", s, "'starfm' or 'estarfm'"),
    };
    let rule = DateRule {
        dates,
        start,
        end,
        include_fine_dates,
        keep_observed,
        exclude,
        pair_match: match pair_match {
            "auto" => PairMatch::Auto,
            "window" => PairMatch::Window,
            "nearest" => PairMatch::Nearest,
            s => return bad("pair_match", s, "'auto', 'window' or 'nearest'"),
        },
        tolerance,
        min_valid,
        max_gap,
        pairs: match pairs {
            "bracket" => PairsMode::Bracket,
            "nearest" => PairsMode::Nearest,
            s => return bad("pairs", s, "'bracket' or 'nearest'"),
        },
    };
    let pl = fusion::plan(
        &fine_days,
        &fine_valid,
        &coarse_days,
        coarse_windows.as_deref(),
        &rule,
        m,
    )
    .map_err(PyValueError::new_err)?;
    Ok((
        pl.targets
            .into_iter()
            .map(|t| (t.day, t.coarse, t.pairs, t.observed, t.extrapolated))
            .collect(),
        pl.skipped,
        pl.unpaired,
    ))
}

/// The size of a coarse cell in fine pixels from an affine geometry (`required_halo`'s cell term).
#[pyfunction]
fn halo(method: &str, radius_px: usize, cell_px: f64, v_scope: &str) -> PyResult<usize> {
    let m = if method == "estarfm" {
        Method::Estarfm
    } else {
        Method::Starfm
    };
    let v = if v_scope == "cell" {
        VScope::Cell
    } else {
        VScope::Window
    };
    Ok(fusion::required_halo(m, radius_px, cell_px, v))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "fusion")?;
    m.add("__doc__", "STARFM and ESTARFM spatiotemporal fusion")?;
    m.add_function(wrap_pyfunction!(starfm, &m)?)?;
    m.add_function(wrap_pyfunction!(estarfm, &m)?)?;
    m.add_function(wrap_pyfunction!(image_sigma, &m)?)?;
    m.add_function(wrap_pyfunction!(plan, &m)?)?;
    m.add_function(wrap_pyfunction!(halo, &m)?)?;
    Ok(m)
}
