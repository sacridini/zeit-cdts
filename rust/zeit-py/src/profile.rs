//! zeit._core.profile: the Python module of zeit-core's monotone curve of a model's error against the DI (CAST's
//! errorProfiles).

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::profile::{self, Monotone};

/// The monotone P-spline of y on x (increasing, or decreasing) with k
/// B-splines -> dict of lo, hi, coef, lambda, edf; None without 4 distinct x.
#[pyfunction]
#[pyo3(signature = (x, y, increasing = true, k = 10))]
fn fit<'py>(
    py: Python<'py>,
    x: PyArrayLike1<'py, f64, AllowTypeChange>,
    y: PyArrayLike1<'py, f64, AllowTypeChange>,
    increasing: bool,
    k: usize,
) -> PyResult<Option<Bound<'py, PyDict>>> {
    let (x, y) = (x.as_array().to_vec(), y.as_array().to_vec());
    if x.len() != y.len() {
        return Err(PyValueError::new_err("x and y must have the same length"));
    }
    let Some(m) = py.detach(|| profile::fit_monotone(&x, &y, increasing, k)) else {
        return Ok(None);
    };
    let d = PyDict::new(py);
    d.set_item("lo", m.lo)?;
    d.set_item("hi", m.hi)?;
    d.set_item("coef", PyArray1::from_vec(py, m.coef))?;
    d.set_item("lambda", m.lambda)?;
    d.set_item("edf", m.edf)?;
    Ok(Some(d))
}

/// The curve (lo, hi, coef of fit) at each x.
#[pyfunction]
#[pyo3(signature = (lo, hi, coef, x, n_jobs = -1))]
fn predict<'py>(
    py: Python<'py>,
    lo: f64,
    hi: f64,
    coef: PyArrayLike1<'py, f64, AllowTypeChange>,
    x: PyArrayLike1<'py, f64, AllowTypeChange>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let coef = coef.as_array().to_vec();
    // NaN fails too
    if coef.len() < 4 || hi.partial_cmp(&lo) != Some(std::cmp::Ordering::Greater) {
        return Err(PyValueError::new_err(
            "a curve needs at least 4 coefficients and hi > lo",
        ));
    }
    let m = Monotone {
        lo,
        hi,
        coef,
        lambda: f64::NAN,
        edf: f64::NAN,
    };
    let x = x.as_array().to_vec();
    let out = py.detach(|| m.predict(&x, n_jobs));
    Ok(PyArray1::from_vec(py, out))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "profile")?;
    m.add(
        "__doc__",
        "A monotone P-spline of a model's error against the dissimilarity index (CAST's errorProfiles)",
    )?;
    m.add_function(wrap_pyfunction!(fit, &m)?)?;
    m.add_function(wrap_pyfunction!(predict, &m)?)?;
    Ok(m)
}
