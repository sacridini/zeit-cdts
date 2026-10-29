//! zeit._core.knndm: the Python module of zeit-core's kNNDM distances (Linnenbrink et al. 2024).

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::knndm;

use crate::arrays::c_order;

/// The distance from each query point (m, p) to the nearest train point (n, p).
#[pyfunction]
#[pyo3(signature = (train, query, n_jobs = -1))]
fn nearest<'py>(
    py: Python<'py>,
    train: PyArrayLike2<'py, f64, AllowTypeChange>,
    query: PyArrayLike2<'py, f64, AllowTypeChange>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let (tv, qv) = (train.as_array(), query.as_array());
    let ((n, p), (m, q)) = (tv.dim(), qv.dim());
    if p != q {
        return Err(PyValueError::new_err(format!(
            "the query points have {q} dimensions, the train points {p}"
        )));
    }
    let (t, x) = (c_order(&tv), c_order(&qv));
    let (t, x): (&[f64], &[f64]) = (&t, &x);
    let d = py.detach(|| knndm::nearest(t, n, p, x, m, n_jobs));
    Ok(PyArray1::from_vec(py, d))
}

/// The distance from each point (n, p) to the nearest point of another fold
/// (folds: one per point, negative: in none); NaN without one.
#[pyfunction]
#[pyo3(signature = (points, folds, n_jobs = -1))]
fn nearest_other<'py>(
    py: Python<'py>,
    points: PyArrayLike2<'py, f64, AllowTypeChange>,
    folds: PyArrayLike1<'py, i64, AllowTypeChange>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let pv = points.as_array();
    let (n, p) = pv.dim();
    let folds = folds.as_array().to_vec();
    if folds.len() != n {
        return Err(PyValueError::new_err(format!("{} folds for {n} points", folds.len())));
    }
    let x = c_order(&pv);
    let x: &[f64] = &x;
    let d = py.detach(|| knndm::nearest_other(x, n, p, &folds, n_jobs));
    Ok(PyArray1::from_vec(py, d))
}

/// W, the integral of the absolute difference of the empirical distribution
/// functions of a and b (NaN left out).
#[pyfunction]
fn wasserstein(a: PyArrayLike1<'_, f64, AllowTypeChange>, b: PyArrayLike1<'_, f64, AllowTypeChange>) -> f64 {
    knndm::wasserstein(&a.as_array().to_vec(), &b.as_array().to_vec())
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "knndm")?;
    m.add(
        "__doc__",
        "The nearest-neighbour distances of kNNDM (Linnenbrink et al. 2024) and their W statistic",
    )?;
    m.add_function(wrap_pyfunction!(nearest, &m)?)?;
    m.add_function(wrap_pyfunction!(nearest_other, &m)?)?;
    m.add_function(wrap_pyfunction!(wasserstein, &m)?)?;
    Ok(m)
}
