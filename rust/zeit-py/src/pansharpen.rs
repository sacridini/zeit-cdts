//! zeit._core.pansharpen: the Python module (pansharpening of multispectral bands with a finer pan).

use numpy::{PyArray3, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray3, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::pansharpen::{self, Method, Options};

/// `ms` `(K, y, x)` float32 (the bands on the pan's grid) fused with `pan` `(y, x)` float32 by `method` ("brovey",
/// "gsa", "hpf" or "pca"); `ratio` is the pan cells per ms cell on a side (any real >= 1); `weights` the Brovey intensity's (fitted when
/// None), `select` the indices of the bands that enter the estimate of the intensity (all when None). Returns `(fused (K, y, x) float32,
/// weights, gains)` (what the method estimated, see the core).
#[pyfunction]
#[pyo3(signature = (ms, pan, method, ratio, weights = None, select = None, kernel = 0, gain = 1.0, gain_regression = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn pansharpen_batch<'py>(
    py: Python<'py>,
    ms: PyReadonlyArray3<'py, f32>,
    pan: PyReadonlyArray2<'py, f32>,
    method: &str,
    ratio: f64,
    weights: Option<Vec<f64>>,
    select: Option<Vec<usize>>,
    kernel: usize,
    gain: f64,
    gain_regression: bool,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray3<f32>>, Vec<f64>, Vec<f64>)> {
    let [nb, ny, nx] = *ms.shape() else { unreachable!() };
    if pan.shape() != [ny, nx] {
        return Err(PyValueError::new_err(format!(
            "pan must be (y, x) = ({ny}, {nx}), got {:?}",
            pan.shape()
        )));
    }
    let method = Method::parse(method).ok_or_else(|| {
        PyValueError::new_err(format!("method must be 'brovey', 'gsa', 'hpf' or 'pca', got {method:?}"))
    })?;
    let ms_s = ms
        .as_slice()
        .map_err(|_| PyValueError::new_err("ms must be C-contiguous"))?;
    let pan_s = pan
        .as_slice()
        .map_err(|_| PyValueError::new_err("pan must be C-contiguous"))?;
    let opt = Options {
        ratio,
        weights: weights.unwrap_or_default(),
        select: select.unwrap_or_default(),
        kernel,
        gain,
        gain_regression,
        n_jobs,
    };
    let fusion = py
        .detach(|| pansharpen::pansharpen(ms_s, pan_s, nb, ny, nx, method, &opt))
        .map_err(PyValueError::new_err)?;
    let arr = PyArray3::<f32>::zeros(py, [nb, ny, nx], false);
    {
        let mut w = arr.readwrite();
        w.as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?
            .copy_from_slice(&fusion.data);
    }
    Ok((arr, fusion.weights, fusion.gains))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "pansharpen")?;
    m.add(
        "__doc__",
        "Pansharpening (Brovey, GSA, HPF and PCA)",
    )?;
    m.add_function(wrap_pyfunction!(pansharpen_batch, &m)?)?;
    Ok(m)
}
