//! zeit._core.correlation: the Python module of zeit-core's lagged correlation,
//! date matching and false discovery rate.

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::correlation::{self, Best, Lag, LagSettings, Match, Method, YRef};

use crate::arrays::c_order;

/// For each lag, the row of `y` paired with each date of `x`, or -1 ->
/// (lags, dates of x) int32. `matching`: "exact", "nearest" (within
/// `tolerance` days), "month" or "year"; `unit` "steps" (observations of x)
/// or "months".
#[pyfunction]
#[pyo3(signature = (x_days, y_days, lags, matching = "nearest", tolerance = 8.0, unit = "steps"))]
fn match_index<'py>(
    py: Python<'py>,
    x_days: PyArrayLike1<'py, i64, AllowTypeChange>,
    y_days: PyArrayLike1<'py, i64, AllowTypeChange>,
    lags: Vec<i32>,
    matching: &str,
    tolerance: f64,
    unit: &str,
) -> PyResult<Bound<'py, numpy::PyArray2<i32>>> {
    let m = match matching {
        "exact" => Match::Exact,
        "nearest" => Match::Nearest(tolerance),
        "month" => Match::SameMonth,
        "year" => Match::SameYear,
        other => {
            return Err(PyValueError::new_err(format!(
                "matching must be 'exact', 'nearest', 'month' or 'year', got {other:?}"
            )))
        }
    };
    let x: Vec<i64> = x_days.as_array().iter().copied().collect();
    let y: Vec<i64> = y_days.as_array().iter().copied().collect();
    if y.windows(2).any(|w| w[1] < w[0]) {
        return Err(PyValueError::new_err("the dates of y must be ascending"));
    }
    let lag_of = |k: i32| match unit {
        "steps" => Ok(Lag::Steps(k)),
        "months" => Ok(Lag::Months(k)),
        other => Err(PyValueError::new_err(format!("unit must be 'steps' or 'months', got {other:?}"))),
    };
    let mut flat = Vec::with_capacity(lags.len() * x.len());
    for &k in &lags {
        flat.extend(correlation::match_index(&x, &y, m, lag_of(k)?));
    }
    PyArray1::from_vec(py, flat).reshape([lags.len(), x.len()])
}

/// The lagged correlation of every pixel of `x` (dates, pixels) with `y` (a cube
/// (dates of y, pixels) or one series (dates of y,)), `index` (lags, dates of x)
/// the row of y of each date at each lag (-1 none) -> dict of `r`, `p`, `n`,
/// `n_eff` (lags, pixels) and `best_lag` (int16, -1 none), `best_r`, `best_p`
/// (pixels).
#[pyfunction]
#[pyo3(signature = (x, y, index, method = "pearson", min_pairs = 10, effective_n = true, best = "absolute",
    sidak = true, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn lagged<'py>(
    py: Python<'py>,
    x: PyArrayLike2<'py, f64, AllowTypeChange>,
    y: &Bound<'py, PyAny>,
    index: PyArrayLike2<'py, i32, AllowTypeChange>,
    method: &str,
    min_pairs: usize,
    effective_n: bool,
    best: &str,
    sidak: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let method = match method {
        "pearson" => Method::Pearson,
        "spearman" => Method::Spearman,
        other => return Err(PyValueError::new_err(format!("method must be 'pearson' or 'spearman', got {other:?}"))),
    };
    let best = match best {
        "absolute" => Best::Absolute,
        "positive" => Best::Positive,
        "negative" => Best::Negative,
        other => {
            return Err(PyValueError::new_err(format!(
                "best must be 'absolute', 'positive' or 'negative', got {other:?}"
            )))
        }
    };
    let xv = x.as_array();
    let (t, n) = xv.dim();
    let xs = c_order(&xv);
    let xs: &[f64] = &xs;
    let iv = index.as_array();
    let (nl, ti) = iv.dim();
    if ti != t {
        return Err(PyValueError::new_err(format!("index has {ti} dates, x has {t}")));
    }
    let idx: Vec<Vec<i32>> = (0..nl).map(|l| iv.row(l).iter().copied().collect()).collect();
    let s = LagSettings {
        method,
        min_pairs,
        effective_n,
        best,
        sidak,
    };
    let cube: Option<PyArrayLike2<'py, f64, AllowTypeChange>> = y.extract().ok();
    let r = if let Some(yc) = cube {
        let yv = yc.as_array();
        let (ty, ny) = yv.dim();
        if ny != n {
            return Err(PyValueError::new_err(format!("y has {ny} pixels, x has {n}")));
        }
        let ys = c_order(&yv);
        let ys: &[f64] = &ys;
        py.detach(|| correlation::lagged(xs, t, n, YRef::Cube(ys, ty), &idx, &s, n_jobs))
    } else {
        let y1: PyArrayLike1<'py, f64, AllowTypeChange> = y
            .extract()
            .map_err(|_| PyValueError::new_err("y must be a (dates, pixels) array or a 1-D series"))?;
        let yv = y1.as_array();
        let ys = c_order(&yv);
        let ys: &[f64] = &ys;
        py.detach(|| correlation::lagged(xs, t, n, YRef::Series(ys), &idx, &s, n_jobs))
    };
    let d = PyDict::new(py);
    d.set_item("r", PyArray1::from_vec(py, r.r).reshape([nl, n])?)?;
    d.set_item("p", PyArray1::from_vec(py, r.p).reshape([nl, n])?)?;
    d.set_item("n", PyArray1::from_vec(py, r.n).reshape([nl, n])?)?;
    d.set_item("n_eff", PyArray1::from_vec(py, r.n_eff).reshape([nl, n])?)?;
    d.set_item("best_lag", PyArray1::from_vec(py, r.best_lag))?;
    d.set_item("best_r", PyArray1::from_vec(py, r.best_r))?;
    d.set_item("best_p", PyArray1::from_vec(py, r.best_p))?;
    Ok(d)
}

/// Benjamini and Hochberg's false discovery rate over the non-NaN p-values ->
/// (adjusted q, threshold p*).
#[pyfunction]
#[pyo3(signature = (p, alpha = 0.10))]
fn fdr<'py>(
    py: Python<'py>,
    p: PyArrayLike1<'py, f64, AllowTypeChange>,
    alpha: f64,
) -> PyResult<(Bound<'py, PyArray1<f64>>, f64)> {
    if !(alpha > 0.0 && alpha < 1.0) {
        return Err(PyValueError::new_err(format!("alpha must be in (0, 1), got {alpha}")));
    }
    let v: Vec<f64> = p.as_array().iter().copied().collect();
    let (q, star) = py.detach(|| correlation::fdr_bh(&v, alpha));
    Ok((PyArray1::from_vec(py, q), star))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "correlation")?;
    m.add("__doc__", "Lagged correlation with effective sample size, date matching, Sidak and FDR")?;
    m.add_function(wrap_pyfunction!(match_index, &m)?)?;
    m.add_function(wrap_pyfunction!(lagged, &m)?)?;
    m.add_function(wrap_pyfunction!(fdr, &m)?)?;
    Ok(m)
}
