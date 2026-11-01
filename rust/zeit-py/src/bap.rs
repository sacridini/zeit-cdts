//! zeit._core.bap: the Python module.

use numpy::{AllowTypeChange, IntoPyArray, PyArray2, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::bap::{self, Date, Options};

/// The best date of each pixel of a window: states [dates, rows, cols] uint8 (0 no data, 1 cloud, 2 clear), the
/// dates' day of the year, sensor score and cover score. Returns (date int32, -1 none; score float32, NaN none;
/// clear dates uint16), each [rows, cols].
#[pyfunction]
#[pyo3(signature = (states, doy, sensor, cover, target_doy = 196.0, doy_sigma = 30.0, cloud_min = 1.0,
    cloud_max = 50.0, w_doy = 0.3, w_cloud = 0.5, w_sensor = 0.1, w_cover = 0.1, n_jobs = -1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn best_dates<'py>(
    py: Python<'py>,
    states: PyArrayLikeDyn<'py, u8, AllowTypeChange>,
    doy: Vec<f64>,
    sensor: Vec<f64>,
    cover: Vec<f64>,
    target_doy: f64,
    doy_sigma: f64,
    cloud_min: f64,
    cloud_max: f64,
    w_doy: f64,
    w_cloud: f64,
    w_sensor: f64,
    w_cover: f64,
    n_jobs: i32,
) -> PyResult<(
    Bound<'py, PyArray2<i32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<u16>>,
)> {
    let a = states.as_array();
    if a.ndim() != 3 {
        return Err(PyValueError::new_err("states must be 3D [dates, rows, cols]"));
    }
    let (t, rows, cols) = (a.shape()[0], a.shape()[1], a.shape()[2]);
    if doy.len() != t || sensor.len() != t || cover.len() != t {
        return Err(PyValueError::new_err(format!(
            "doy, sensor and cover need {t} values, one a date"
        )));
    }
    let values: Vec<u8> = a.iter().copied().collect();
    let dates: Vec<Date> = (0..t)
        .map(|i| Date {
            doy: doy[i],
            sensor: sensor[i],
            cover: cover[i],
        })
        .collect();
    let opts = Options {
        target_doy,
        doy_sigma,
        cloud_min,
        cloud_max,
        w_doy,
        w_cloud,
        w_sensor,
        w_cover,
    };
    let b = py
        .detach(|| bap::best(&values, rows, cols, &dates, &opts, n_jobs))
        .map_err(PyValueError::new_err)?;
    let shape = (rows, cols);
    let err = |e: numpy::ndarray::ShapeError| PyValueError::new_err(e.to_string());
    let date = numpy::ndarray::Array2::from_shape_vec(shape, b.date).map_err(err)?;
    let score = numpy::ndarray::Array2::from_shape_vec(shape, b.score).map_err(err)?;
    let clear = numpy::ndarray::Array2::from_shape_vec(shape, b.clear).map_err(err)?;
    Ok((date.into_pyarray(py), score.into_pyarray(py), clear.into_pyarray(py)))
}

/// The Euclidean distance (pixels) of each pixel of a [rows, cols] mask to the nearest true pixel, at most `cap`.
#[pyfunction]
#[pyo3(signature = (mask, cap = f64::INFINITY))]
fn distance_to<'py>(
    py: Python<'py>,
    mask: PyArrayLikeDyn<'py, bool, AllowTypeChange>,
    cap: f64,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let a = mask.as_array();
    if a.ndim() != 2 {
        return Err(PyValueError::new_err("mask must be 2D [rows, cols]"));
    }
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let m: Vec<bool> = a.iter().copied().collect();
    let cap = if cap.is_finite() {
        cap
    } else {
        (rows as f64).hypot(cols as f64) + 1.0
    };
    let d = py.detach(|| bap::distance_to(|i| m[i], rows, cols, cap));
    let arr =
        numpy::ndarray::Array2::from_shape_vec((rows, cols), d).map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "bap")?;
    m.add(
        "__doc__",
        "Best available pixel composites: each pixel's best date by its quality's score",
    )?;
    m.add_function(wrap_pyfunction!(best_dates, &m)?)?;
    m.add_function(wrap_pyfunction!(distance_to, &m)?)?;
    Ok(m)
}
