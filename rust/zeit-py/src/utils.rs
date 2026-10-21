//! zeit._rust.utils: the bindings of zeit._core.utils.

use numpy::{AllowTypeChange, PyArray1, PyArrayDyn, PyArrayLikeDyn, PyArrayMethods};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use zeit_core::medoid;

use crate::arrays::c_order;

/// Computes the multidimensional medoid composite over the time axis:
/// (y, x, time, band) -> (y, x, band).
#[pyfunction]
#[pyo3(signature = (input_array, no_data_value = -9999.0))]
fn compute_medoid<'py>(
    py: Python<'py>,
    input_array: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    no_data_value: f64,
) -> PyResult<Bound<'py, PyArrayDyn<f64>>> {
    let view = input_array.as_array();
    if view.ndim() != 4 {
        return Err(PyRuntimeError::new_err(
            "Input array must have exactly 4 dimensions: [Y, X, Time, Band]",
        ));
    }
    let s = view.shape().to_vec();
    let values = c_order(&view);
    let values: &[f64] = &values;
    let out = py.detach(|| medoid::medoid(values, s[0] * s[1], s[2], s[3], no_data_value));
    PyArray1::from_vec(py, out).reshape(numpy::IxDyn(&[s[0], s[1], s[3]]))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "utils")?;
    m.add("__doc__", "Geospatial utilities and processing")?;
    m.add_function(wrap_pyfunction!(compute_medoid, &m)?)?;
    Ok(m)
}
