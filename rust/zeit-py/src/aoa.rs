//! zeit._core.aoa: the Python module of zeit-core's area of applicability (CAST's trainDI and aoa).

use numpy::{AllowTypeChange, PyArray1, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::aoa::{self, TrainDi};

use crate::arrays::c_order;

/// CAST's trainDI of train (points, predictors) with weights and optional
/// folds (one per point, negative: none) -> dict of center, scale, weight,
/// train (scaled and weighted), mean_distance, train_di, threshold.
#[pyfunction]
#[pyo3(signature = (train, weight, folds = None, n_jobs = -1))]
fn train_di<'py>(
    py: Python<'py>,
    train: PyArrayLike2<'py, f64, AllowTypeChange>,
    weight: PyArrayLike1<'py, f64, AllowTypeChange>,
    folds: Option<PyArrayLike1<'py, i64, AllowTypeChange>>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let view = train.as_array();
    let (n, p) = view.dim();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let weight: Vec<f64> = weight.as_array().to_vec();
    let folds: Option<Vec<i64>> = folds.map(|f| f.as_array().to_vec());
    let td = py
        .detach(|| aoa::train_di(values, n, p, &weight, folds.as_deref(), n_jobs))
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    d.set_item("center", PyArray1::from_vec(py, td.center))?;
    d.set_item("scale", PyArray1::from_vec(py, td.scale))?;
    d.set_item("weight", PyArray1::from_vec(py, td.weight))?;
    d.set_item("train", PyArray1::from_vec(py, td.train).reshape([n, p])?)?;
    d.set_item("mean_distance", td.mean_distance)?;
    d.set_item("train_di", PyArray1::from_vec(py, td.train_di))?;
    d.set_item("threshold", td.threshold)?;
    Ok(d)
}

/// The DI of data (pixels, predictors) against a trainDI's parts, and with
/// lpd the local point density -> (di, lpd); lpd empty without.
#[pyfunction]
#[pyo3(signature = (data, center, scale, weight, train, mean_distance, threshold, lpd = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn di<'py>(
    py: Python<'py>,
    data: PyArrayLike2<'py, f64, AllowTypeChange>,
    center: PyArrayLike1<'py, f64, AllowTypeChange>,
    scale: PyArrayLike1<'py, f64, AllowTypeChange>,
    weight: PyArrayLike1<'py, f64, AllowTypeChange>,
    train: PyArrayLike2<'py, f64, AllowTypeChange>,
    mean_distance: f64,
    threshold: f64,
    lpd: bool,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let tv = train.as_array();
    let (n, p) = tv.dim();
    let dv = data.as_array();
    let (m, q) = dv.dim();
    if q != p {
        return Err(PyValueError::new_err(format!(
            "the data have {q} predictors, the training data {p}"
        )));
    }
    let td = TrainDi {
        n,
        p,
        center: center.as_array().to_vec(),
        scale: scale.as_array().to_vec(),
        weight: weight.as_array().to_vec(),
        train: c_order(&tv).into_owned(),
        mean_distance,
        train_di: vec![],
        threshold,
    };
    if td.center.len() != p || td.scale.len() != p || td.weight.len() != p {
        return Err(PyValueError::new_err(
            "center, scale and weight need one value per predictor",
        ));
    }
    let values = c_order(&dv);
    let values: &[f64] = &values;
    let (d, l) = py.detach(|| aoa::di(values, m, &td, lpd, n_jobs));
    Ok((PyArray1::from_vec(py, d), PyArray1::from_vec(py, l)))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "aoa")?;
    m.add(
        "__doc__",
        "Area of applicability (Meyer & Pebesma 2021), as CAST's trainDI and aoa",
    )?;
    m.add_function(wrap_pyfunction!(train_di, &m)?)?;
    m.add_function(wrap_pyfunction!(di, &m)?)?;
    Ok(m)
}
