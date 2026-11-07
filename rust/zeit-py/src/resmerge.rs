//! zeit._core.resmerge: the Python module (Sentinel-2 resolution merge, FORCE's two methods).

use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::resmerge::{self, Scene, NT};

fn slice<'a, D: numpy::ndarray::Dimension, T: numpy::Element>(
    a: &'a numpy::PyReadonlyArray<'_, T, D>,
    what: &str,
) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))
}

/// The six 20 m bands `targets` `(6, y, x)` float32 predicted at the pixels that are neither `off` nor `bad`, from
/// `green`, `red` and `bnir` `(y, x)` float32, all on the 10 m grid; the values divided by `denom` are reflectance.
/// `method` is "regression" or "improphe"; `force` is FORCE's arithmetic (truncated integers, the regression in
/// place on one thread). `(6, y, x)` float32.
#[pyfunction]
#[pyo3(signature = (green, red, bnir, targets, off, bad, method, denom = 10000.0, force = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn resmerge_batch<'py>(
    py: Python<'py>,
    green: PyReadonlyArray2<'py, f32>,
    red: PyReadonlyArray2<'py, f32>,
    bnir: PyReadonlyArray2<'py, f32>,
    targets: PyReadonlyArray3<'py, f32>,
    off: PyReadonlyArray2<'py, bool>,
    bad: PyReadonlyArray2<'py, bool>,
    method: &str,
    denom: f64,
    force: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f32>>> {
    let [ny, nx] = *green.shape() else { unreachable!() };
    for (shape, what) in [
        (red.shape(), "red"),
        (bnir.shape(), "bnir"),
        (off.shape(), "off"),
        (bad.shape(), "bad"),
    ] {
        if shape != [ny, nx] {
            return Err(PyValueError::new_err(format!(
                "{what} must be (y, x) = ({ny}, {nx}), got {shape:?}"
            )));
        }
    }
    if targets.shape() != [NT, ny, nx] {
        return Err(PyValueError::new_err(format!(
            "targets must be (6, y, x) = (6, {ny}, {nx}), got {:?}",
            targets.shape()
        )));
    }
    if !(denom > 0.0) || !denom.is_finite() {
        return Err(PyValueError::new_err("denom must be positive"));
    }
    let run: fn(&Scene, f64, bool, i32) -> Vec<f32> = match method {
        "regression" => resmerge::regression,
        "improphe" => resmerge::improphe,
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'regression' or 'improphe', got {other:?}"
            )))
        }
    };
    let sc = Scene {
        nx,
        ny,
        green: slice(&green, "green")?,
        red: slice(&red, "red")?,
        bnir: slice(&bnir, "bnir")?,
        targets: slice(&targets, "targets")?,
        off: slice(&off, "off")?,
        bad: slice(&bad, "bad")?,
    };
    let out = py.detach(|| run(&sc, denom, force, n_jobs));
    let arr = PyArray3::<f32>::zeros(py, [NT, ny, nx], false);
    {
        let mut w = arr.readwrite();
        w.as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?
            .copy_from_slice(&out);
    }
    Ok(arr)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "resmerge")?;
    m.add(
        "__doc__",
        "Sentinel-2 resolution merge (FORCE's regression and ImproPhe methods)",
    )?;
    m.add_function(wrap_pyfunction!(resmerge_batch, &m)?)?;
    Ok(m)
}
