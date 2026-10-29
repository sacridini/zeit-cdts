//! zeit._core.variogram: the Python module of zeit-core's semivariogram (the range of a map's spatial
//! autocorrelation, blockCV's cv_spatial_autocor).

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::variogram;

/// The empirical semivariogram of v at the points (x, y), in bins equal bands
/// of distance up to cutoff, and the exponential model fitted to it -> dict of
/// distance, gamma, pairs (per band with pairs) and nugget, sill, scale, range
/// (None without at least three bands).
#[pyfunction]
#[pyo3(signature = (x, y, v, bins = 15, cutoff = 1.0, n_jobs = -1))]
fn fit<'py>(
    py: Python<'py>,
    x: PyArrayLike1<'py, f64, AllowTypeChange>,
    y: PyArrayLike1<'py, f64, AllowTypeChange>,
    v: PyArrayLike1<'py, f64, AllowTypeChange>,
    bins: usize,
    cutoff: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let (x, y, v) = (x.as_array().to_vec(), y.as_array().to_vec(), v.as_array().to_vec());
    if x.len() != v.len() || y.len() != v.len() {
        return Err(PyValueError::new_err("x, y and v must have the same length"));
    }
    let (e, model) = py.detach(|| {
        let e = variogram::empirical(&x, &y, &v, bins, cutoff, n_jobs);
        let model = variogram::fit_exponential(&e);
        (e, model)
    });
    let d = PyDict::new(py);
    d.set_item("distance", PyArray1::from_vec(py, e.distance))?;
    d.set_item("gamma", PyArray1::from_vec(py, e.gamma))?;
    d.set_item("pairs", PyArray1::from_vec(py, e.pairs))?;
    for (key, value) in [
        ("nugget", model.map(|m| m.nugget)),
        ("sill", model.map(|m| m.sill)),
        ("scale", model.map(|m| m.scale)),
        ("range", model.map(|m| m.range)),
    ] {
        d.set_item(key, value)?;
    }
    Ok(d)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "variogram")?;
    m.add(
        "__doc__",
        "The semivariogram of a map and its exponential model (the range of its spatial autocorrelation)",
    )?;
    m.add_function(wrap_pyfunction!(fit, &m)?)?;
    Ok(m)
}
