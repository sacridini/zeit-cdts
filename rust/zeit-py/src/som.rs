//! zeit._rust.som: the bindings of zeit._core.som.

use numpy::{AllowTypeChange, PyArray1, PyArray3, PyArrayLikeDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyIndexError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use zeit_core::som::{self, Grid, Schedule};

use crate::arrays::c_order;

/// The weights (x, y, D), float64 and C-contiguous: updated in place.
fn weights_of<'py>(weights: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let w = weights
        .cast::<PyArray3<f64>>()
        .map_err(|_| PyValueError::new_err("weights must have shape (x, y, input_len)"))?
        .clone();
    if !w.is_c_contiguous() {
        return Err(PyTypeError::new_err(
            "weights must be a C-contiguous float64 array (updated in place)",
        ));
    }
    Ok(w)
}

/// The data (N, D) of `d` features, at least one sample.
fn data_of(data: &PyArrayLikeDyn<'_, f64, AllowTypeChange>, d: usize) -> PyResult<(Vec<f64>, usize)> {
    let v = data.as_array();
    if v.ndim() != 2 || v.shape()[1] != d {
        return Err(PyValueError::new_err("data must have shape (n_samples, input_len)"));
    }
    let n = v.shape()[0];
    Ok((c_order(&v).into_owned(), n))
}

#[allow(clippy::too_many_arguments)]
fn schedule(learning_rate: f64, sigma: f64, lr_decay: i32, sigma_decay: i32, neighborhood: i32) -> Schedule {
    Schedule {
        learning_rate,
        sigma,
        lr_decay,
        sigma_decay,
        neighborhood,
    }
}

fn grid_of(
    w: &Bound<'_, PyArray3<f64>>,
    xx: &PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    yy: &PyArrayLikeDyn<'_, f64, AllowTypeChange>,
) -> PyResult<Grid> {
    let s = w.shape();
    let (xx, yy) = (xx.as_array(), yy.as_array());
    Grid::new(s[0], s[1], &c_order(&xx), &c_order(&yy)).map_err(PyValueError::new_err)
}

/// Online SOM training (port of MiniSom.train); updates weights in place.
#[pyfunction]
#[pyo3(signature = (weights, data, order, num_iteration, use_epochs, learning_rate, sigma, lr_decay, sigma_decay,
                    neighborhood, xx, yy))]
#[allow(clippy::too_many_arguments)]
fn train_online(
    py: Python<'_>,
    weights: &Bound<'_, PyAny>,
    data: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    order: PyArrayLikeDyn<'_, i64, AllowTypeChange>,
    num_iteration: i64,
    use_epochs: bool,
    learning_rate: f64,
    sigma: f64,
    lr_decay: i32,
    sigma_decay: i32,
    neighborhood: i32,
    xx: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    yy: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
) -> PyResult<()> {
    let w = weights_of(weights)?;
    let grid = grid_of(&w, &xx, &yy)?;
    let d = w.shape()[2];
    let (data, n) = data_of(&data, d)?;
    if n < 1 {
        return Err(PyValueError::new_err("data must contain at least one sample"));
    }
    let order: Vec<usize> = order
        .as_array()
        .iter()
        .map(|&i| usize::try_from(i).ok().filter(|&i| i < n))
        .collect::<Option<_>>()
        .ok_or_else(|| PyIndexError::new_err("sample index out of range"))?;
    if num_iteration < 1 {
        return Err(PyValueError::new_err("num_iteration must be >= 1"));
    }
    let s = schedule(learning_rate, sigma, lr_decay, sigma_decay, neighborhood);
    let mut rw = w.readwrite();
    let wslice = rw.as_slice_mut()?;
    py.detach(|| som::train_online(wslice, &data, d, &order, num_iteration as usize, use_epochs, &s, &grid));
    Ok(())
}

/// Batch SOM training (port of MiniSom.train_batch_offline); updates weights in place.
#[pyfunction]
#[pyo3(signature = (weights, data, num_iteration, learning_rate, sigma, lr_decay, sigma_decay, neighborhood, xx, yy,
                    n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn train_batch(
    py: Python<'_>,
    weights: &Bound<'_, PyAny>,
    data: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    num_iteration: i64,
    learning_rate: f64,
    sigma: f64,
    lr_decay: i32,
    sigma_decay: i32,
    neighborhood: i32,
    xx: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    yy: PyArrayLikeDyn<'_, f64, AllowTypeChange>,
    n_jobs: i32,
) -> PyResult<()> {
    let w = weights_of(weights)?;
    let grid = grid_of(&w, &xx, &yy)?;
    let d = w.shape()[2];
    let (data, n) = data_of(&data, d)?;
    if n < 1 {
        return Err(PyValueError::new_err("data must contain at least one sample"));
    }
    if num_iteration < 1 {
        return Err(PyValueError::new_err("num_iteration must be >= 1"));
    }
    let s = schedule(learning_rate, sigma, lr_decay, sigma_decay, neighborhood);
    let mut rw = w.readwrite();
    let wslice = rw.as_slice_mut()?;
    py.detach(|| som::train_batch(wslice, &data, d, num_iteration as usize, &s, &grid, n_jobs));
    Ok(())
}

/// Find BMU for samples.
#[pyfunction]
#[pyo3(signature = (data, weights, n_jobs = -1))]
fn predict_bmus<'py>(
    py: Python<'py>,
    data: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    weights: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray1<i32>>> {
    let w = weights.as_array();
    if w.ndim() != 3 {
        return Err(PyValueError::new_err("weights must have shape (x, y, input_len)"));
    }
    let (k, d) = (w.shape()[0] * w.shape()[1], w.shape()[2]);
    let (data, _) = data_of(&data, d)?;
    let wv = c_order(&w);
    let wv: &[f64] = &wv;
    let out = py.detach(|| som::predict_bmus(&data, d, wv, k, n_jobs));
    Ok(PyArray1::from_vec(py, out))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "som")?;
    m.add("__doc__", "SOM kernels")?;
    m.add_function(wrap_pyfunction!(train_online, &m)?)?;
    m.add_function(wrap_pyfunction!(train_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(predict_bmus, &m)?)?;
    Ok(m)
}
