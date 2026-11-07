//! zeit._core.coreg: the Python module (global co-registration of an image to a reference, AROSICS' `COREG`).

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::coreg::{self, Image, Params};

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray2<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))
}

/// The shift of `target` `(y, x)` float32 to `reference`, both on the same grid with their validity masks
/// `(y, x)` bool: a dict with `success`, `reason`, `x_shift` and `y_shift` (pixels, the amount the target has to
/// be moved, x to the right and y down), `x_int`, `y_int`, `x_sub`, `y_sub`, `reliability`, `ssim_before`,
/// `ssim_after`, `window` (x0, y0, x1, y1), `fft_size` and `iterations`.
#[pyfunction]
#[pyo3(signature = (reference, ref_valid, target, tgt_valid, window_size = 256, window_pos = None, max_shift = 5.0,
                    max_iter = 5, integer_range = None, nodata = None, fill = 0.0))]
#[allow(clippy::too_many_arguments)]
fn coregister<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray2<'py, f32>,
    ref_valid: PyReadonlyArray2<'py, bool>,
    target: PyReadonlyArray2<'py, f32>,
    tgt_valid: PyReadonlyArray2<'py, bool>,
    window_size: usize,
    window_pos: Option<(f64, f64)>,
    max_shift: f64,
    max_iter: usize,
    integer_range: Option<(f64, f64)>,
    nodata: Option<f64>,
    fill: f32,
) -> PyResult<Bound<'py, PyDict>> {
    let [ny, nx] = *reference.shape() else { unreachable!() };
    for (shape, what) in [
        (ref_valid.shape(), "ref_valid"),
        (target.shape(), "target"),
        (tgt_valid.shape(), "tgt_valid"),
    ] {
        if shape != [ny, nx] {
            return Err(PyValueError::new_err(format!(
                "{what} must be (y, x) = ({ny}, {nx}), got {shape:?}"
            )));
        }
    }
    if window_size < 16 {
        return Err(PyValueError::new_err("window_size must be at least 16"));
    }
    if !(max_shift > 0.0) || max_iter < 1 {
        return Err(PyValueError::new_err(
            "max_shift must be positive and max_iter at least 1",
        ));
    }
    let p = Params {
        window_size,
        window_pos,
        max_shift,
        max_iter,
        integer_range,
        nodata,
        fill,
    };
    let r = Image {
        data: slice(&reference, "reference")?,
        valid: slice(&ref_valid, "ref_valid")?,
        ny,
        nx,
    };
    let t = Image {
        data: slice(&target, "target")?,
        valid: slice(&tgt_valid, "tgt_valid")?,
        ny,
        nx,
    };
    let o = py.detach(|| coreg::coregister(&r, &t, &p));
    let d = PyDict::new(py);
    d.set_item("success", o.success)?;
    d.set_item("reason", o.failure.map(|f| f.as_str()))?;
    d.set_item("x_shift", o.x_shift)?;
    d.set_item("y_shift", o.y_shift)?;
    d.set_item("x_int", o.x_int)?;
    d.set_item("y_int", o.y_int)?;
    d.set_item("x_sub", o.x_sub)?;
    d.set_item("y_sub", o.y_sub)?;
    d.set_item("reliability", o.reliability)?;
    d.set_item("ssim_before", o.ssim_before)?;
    d.set_item("ssim_after", o.ssim_after)?;
    d.set_item("window", o.window.to_vec())?;
    d.set_item("fft_size", o.fft_size)?;
    d.set_item("iterations", o.iterations)?;
    Ok(d)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "coreg")?;
    m.add(
        "__doc__",
        "Global co-registration of an image to a reference (AROSICS' COREG: phase correlation of one window)",
    )?;
    m.add_function(wrap_pyfunction!(coregister, &m)?)?;
    Ok(m)
}
