//! zeit._core.specadjust: the Python module (FORCE's spectral adjustment of Landsat and MODIS to
//! Sentinel-2A).

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::specadjust::{self, Quirk, Sensor, N_DST, N_SRC};

use crate::arrays::c_order;

/// The (pixels, 6) values: a float32 array is read as it is, any other dtype is converted to float64.
#[derive(FromPyObject)]
enum Values<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyArrayLike2<'py, f64, AllowTypeChange>),
}

/// The ten Sentinel-2A bands (BLUE GREEN RED REDEDGE1 REDEDGE2 REDEDGE3 BROADNIR NIR SWIR1
/// SWIR2) of every pixel of `values` (pixels, 6: BLUE GREEN RED NIR SWIR1 SWIR2, reflectance x
/// 10000) of `sensor` (LND04, LND05, LND07, LND08, LND09, MOD01, MOD02): (pixels, 10), x 10000,
/// not truncated. A pixel with a NaN band is NaN. `force_first_search`: FORCE's literal first search
/// (cluster 9 skipped, the global cluster a candidate), for comparison.
#[pyfunction]
#[pyo3(signature = (values, sensor, n_jobs = -1, force_first_search = false))]
fn specadjust_batch<'py>(
    py: Python<'py>,
    values: Values<'py>,
    sensor: &str,
    n_jobs: i32,
    force_first_search: bool,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let sensor = Sensor::parse(sensor).map_err(PyValueError::new_err)?;
    let quirk = if force_first_search { Quirk::Force } else { Quirk::Fixed };
    let (n_pixels, n_bands) = match &values {
        Values::F32(a) => (a.shape()[0], a.shape()[1]),
        Values::F64(a) => a.as_array().dim(),
    };
    if n_bands != N_SRC {
        return Err(PyValueError::new_err(format!(
            "values must be (pixels, {N_SRC}): BLUE GREEN RED NIR SWIR1 SWIR2"
        )));
    }
    let mut out = vec![0.0f64; n_pixels * N_DST];
    match &values {
        Values::F32(a) => {
            let view = a.as_array();
            let c = c_order(&view);
            let v: &[f32] = &c;
            py.detach(|| specadjust::spectral_adjust_batch_as(v, sensor, &mut out, n_jobs, quirk));
        }
        Values::F64(a) => {
            let view = a.as_array();
            let c = c_order(&view);
            let v: &[f64] = &c;
            py.detach(|| specadjust::spectral_adjust_batch_as(v, sensor, &mut out, n_jobs, quirk));
        }
    }
    PyArray1::from_vec(py, out).reshape([n_pixels, N_DST])
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "specadjust")?;
    m.add("__doc__", "FORCE's spectral adjustment of Landsat and MODIS to Sentinel-2A")?;
    m.add_function(wrap_pyfunction!(specadjust_batch, &m)?)?;
    Ok(m)
}
