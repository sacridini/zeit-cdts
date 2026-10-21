//! zeit._rust.coded: the bindings of zeit._core.coded.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use zeit_core::coded;

use crate::arrays::c_order;

#[pyclass(name = "Params", module = "zeit._rust.coded", get_all, set_all)]
#[derive(Clone)]
pub struct Params {
    train_start: f64,
    train_years: f64,
    consec: i32,
    thresh: f64,
    min_years: f64,
    min_obs: i32,
    loss_only: bool,
    max_events: i32,
}

#[pymethods]
impl Params {
    #[new]
    fn new() -> Self {
        let p = coded::Params::default();
        Params {
            train_start: p.train_start,
            train_years: p.train_years,
            consec: p.consec,
            thresh: p.thresh,
            min_years: p.min_years,
            min_obs: p.min_obs,
            loss_only: p.loss_only,
            max_events: p.max_events,
        }
    }
}

impl Params {
    fn core(&self) -> coded::Params {
        coded::Params {
            train_start: self.train_start,
            train_years: self.train_years,
            consec: self.consec,
            thresh: self.thresh,
            min_years: self.min_years,
            min_obs: self.min_obs,
            loss_only: self.loss_only,
            max_events: self.max_events,
        }
    }
}

/// CODED monitoring of one pixel.
#[pyfunction]
fn coded_pixel(
    t: Vec<f64>,
    ndfi: Vec<f64>,
    features: Vec<f64>,
    n_features: i64,
    params: PyRef<'_, Params>,
) -> PyResult<Vec<f64>> {
    let n_features = usize::try_from(n_features).map_err(|_| PyRuntimeError::new_err("coded: n_features < 0"))?;
    coded::coded_pixel(&t, &ndfi, &features, n_features, &params.core()).map_err(PyRuntimeError::new_err)
}

/// CODED monitoring of (pixels, time) series: (pixels, coded_size).
#[pyfunction]
#[pyo3(signature = (t, ndfi, features, params, n_jobs = -1))]
fn coded_batch<'py>(
    py: Python<'py>,
    t: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    ndfi: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    features: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    params: PyRef<'py, Params>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (tv, nd, fe) = (t.as_array(), ndfi.as_array(), features.as_array());
    let shape_error =
        || PyRuntimeError::new_err("coded: expected t (time), ndfi (pixels, time), features (pixels, n, time)");
    if tv.ndim() != 1 || nd.ndim() != 2 || fe.ndim() != 3 {
        return Err(shape_error());
    }
    let (n_time, pixels, n_features) = (tv.len(), nd.shape()[0], fe.shape()[1]);
    if nd.shape()[1] != n_time || fe.shape()[0] != pixels || fe.shape()[2] != n_time {
        return Err(shape_error());
    }
    let (tv, nd, fe) = (c_order(&tv), c_order(&nd), c_order(&fe));
    let (tv, nd, fe): (&[f64], &[f64], &[f64]) = (&tv, &nd, &fe);
    let p = params.core();
    let size = coded::coded_size(n_features as i64, p.max_events as i64).max(0) as usize;
    let out = py
        .detach(|| coded::coded_batch(tv, nd, fe, pixels, n_features, &p, n_jobs))
        .map_err(PyRuntimeError::new_err)?;
    PyArray1::from_vec(py, out).reshape([pixels, size])
}

#[pyfunction]
fn coded_size(n_features: i64, max_events: i64) -> i64 {
    coded::coded_size(n_features, max_events)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "coded")?;
    m.add("__doc__", "Continuous Degradation Detection (CODED) monitoring")?;
    m.add_class::<Params>()?;
    m.add_function(wrap_pyfunction!(coded_pixel, &m)?)?;
    m.add_function(wrap_pyfunction!(coded_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(coded_size, &m)?)?;
    Ok(m)
}
