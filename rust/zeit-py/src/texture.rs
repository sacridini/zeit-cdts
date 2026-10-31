//! zeit._core.texture: the Python module.

use numpy::{AllowTypeChange, IntoPyArray, PyArray3, PyArrayLikeDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::texture::{self, Feature, Options};

/// The texture features of a [rows, cols] image (NaN none): [features, rows, cols] float64.
#[pyfunction]
#[pyo3(signature = (image, features, window = 7, levels = 32, distance = 1, lo = f64::NAN, hi = f64::NAN, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn texture_features<'py>(
    py: Python<'py>,
    image: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    features: Vec<String>,
    window: usize,
    levels: usize,
    distance: usize,
    lo: f64,
    hi: f64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let a = image.as_array();
    if a.ndim() != 2 {
        return Err(PyValueError::new_err("image must be 2D [rows, cols]"));
    }
    let (rows, cols) = (a.shape()[0], a.shape()[1]);
    let values: Vec<f64> = a.iter().copied().collect();
    let feats: Vec<Feature> =
        features.iter().map(|f| Feature::parse(f)).collect::<Result<_, _>>().map_err(PyValueError::new_err)?;
    let opts = Options { window, levels, distance, lo, hi };
    let out = py
        .detach(|| texture::texture(&values, rows, cols, &feats, &opts, n_jobs))
        .map_err(PyValueError::new_err)?;
    let arr = numpy::ndarray::Array3::from_shape_vec((feats.len(), rows, cols), out)
        .map_err(|e| PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "texture")?;
    m.add("__doc__", "Texture: GLCM properties and window statistics of an image")?;
    m.add_function(wrap_pyfunction!(texture_features, &m)?)?;
    Ok(m)
}
