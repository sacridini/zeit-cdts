//! zeit._rust.sma: the bindings of zeit._core.sma.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use zeit_core::sma;

use crate::arrays::c_order;

/// Fractions of one spectrum, then the RMSE.
#[pyfunction]
#[pyo3(signature = (y, endmembers, k, sum_to_one = true, nonneg = true, delta = 1000.0))]
fn unmix_pixel(
    y: Vec<f64>,
    endmembers: Vec<f64>,
    k: i64,
    sum_to_one: bool,
    nonneg: bool,
    delta: f64,
) -> PyResult<Vec<f64>> {
    let k = usize::try_from(k).unwrap_or(0);
    sma::unmix_pixel(&y, &endmembers, k, sum_to_one, nonneg, delta).map_err(PyRuntimeError::new_err)
}

/// Fractions of (pixels, bands) spectra, then the RMSE: (pixels, k + 1).
#[pyfunction]
#[pyo3(signature = (values, endmembers, sum_to_one = true, nonneg = true, delta = 1000.0, n_jobs = -1))]
fn unmix_batch<'py>(
    py: Python<'py>,
    values: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    endmembers: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    sum_to_one: bool,
    nonneg: bool,
    delta: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (v, e) = (values.as_array(), endmembers.as_array());
    if v.ndim() != 2 || e.ndim() != 2 {
        return Err(PyRuntimeError::new_err(
            "unmix: values (pixels, bands), endmembers (k, bands)",
        ));
    }
    let (n, bands, k) = (v.shape()[0], v.shape()[1], e.shape()[0]);
    if e.shape()[1] != bands {
        return Err(PyRuntimeError::new_err(
            "unmix: the spectra and the endmembers must have the same bands",
        ));
    }
    let (v, e) = (c_order(&v), c_order(&e));
    let (v, e): (&[f64], &[f64]) = (&v, &e);
    let out = py.detach(|| sma::unmix_batch(v, n, bands, e, k, sum_to_one, nonneg, delta, n_jobs));
    PyArray1::from_vec(py, out).reshape([n, k + 1])
}

/// Lawson-Hanson non-negative least squares (A row-major, m x n).
#[pyfunction]
#[pyo3(signature = (A, b, m, n, max_iter = -1))]
#[allow(non_snake_case)]
fn nnls(A: Vec<f64>, b: Vec<f64>, m: i64, n: i64, max_iter: i64) -> PyResult<Vec<f64>> {
    let (m, n) = (
        usize::try_from(m).unwrap_or(usize::MAX),
        usize::try_from(n).unwrap_or(usize::MAX),
    );
    sma::nnls(&A, &b, m, n, max_iter).map_err(PyRuntimeError::new_err)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "sma")?;
    m.add("__doc__", "Linear spectral mixture analysis")?;
    m.add_function(wrap_pyfunction!(unmix_pixel, &m)?)?;
    m.add_function(wrap_pyfunction!(unmix_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(nnls, &m)?)?;
    Ok(m)
}
