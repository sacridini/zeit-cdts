//! zeit._core.coreg: the Python module (global co-registration of an image to a reference, AROSICS' `COREG`).

use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::coreg::{self, Image, Params};
use zeit_core::coreg_local::{self, Geo, LocalParams};

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
                    max_iter = 5, integer_range = None, nodata = None, fill = 0.0, cell = None))]
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
    cell: Option<(f64, f64)>,
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
    let cell = cell.unwrap_or((30.0, 30.0));
    if !(cell.0 > 0.0 && cell.1 > 0.0) {
        return Err(PyValueError::new_err("cell must be positive"));
    }
    let p = Params {
        window_size,
        window_pos,
        max_shift,
        max_iter,
        integer_range,
        nodata,
        fill,
        cell: [cell.0, cell.1],
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

/// The tie points of a pair of images on the same grid (the overlap of the two, `(y, x)` float32 with their
/// validity masks `(y, x)` bool), AROSICS' `COREG_LOCAL` grid: `points` are `(x, y)` pixel coordinates of the
/// target's whole image (the corner of its first pixel is `(0, 0)`), `offset` the `(x, y)` pixels of that image
/// before the overlap's first one, `geo` its `(x0, dx, y0, dy)`. Returns a dict of lists, one entry per point
/// kept (on data of both images): `index` (of `points`), `x_map`, `y_map`, `x_shift_px`, `y_shift_px`,
/// `x_shift_map`, `y_shift_map`, `abs_shift`, `angle`, `reliability`, `ssim_before`, `ssim_after`,
/// `ssim_improved`, `success`, `reason`, `window` (x0, y0, x1, y1 in the overlap),
/// `fft_size`, `l1`, `l2`, `l3` and `outlier`.
#[pyfunction]
#[pyo3(signature = (reference, ref_valid, target, tgt_valid, points, offset, geo, window_size = 256,
                    max_shift = 5.0, max_iter = 5, integer_range = None, nodata = None, fill = 0.0,
                    filter_level = 3, min_reliability = 60.0, rs_max_outlier = 10.0, rs_tolerance = 2.5,
                    rs_max_iter = 15, rs_seed = Some(0), rs_arosics_quirk = false, n_jobs = -1, cell = None))]
#[allow(clippy::too_many_arguments)]
fn tie_points<'py>(
    py: Python<'py>,
    reference: PyReadonlyArray2<'py, f32>,
    ref_valid: PyReadonlyArray2<'py, bool>,
    target: PyReadonlyArray2<'py, f32>,
    tgt_valid: PyReadonlyArray2<'py, bool>,
    points: PyReadonlyArray2<'py, f64>,
    offset: (f64, f64),
    geo: (f64, f64, f64, f64),
    window_size: usize,
    max_shift: f64,
    max_iter: usize,
    integer_range: Option<(f64, f64)>,
    nodata: Option<f64>,
    fill: f32,
    filter_level: u8,
    min_reliability: f64,
    rs_max_outlier: f64,
    rs_tolerance: f64,
    rs_max_iter: usize,
    rs_seed: Option<u64>,
    rs_arosics_quirk: bool,
    n_jobs: i32,
    cell: Option<(f64, f64)>,
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
    if points.shape()[1] != 2 {
        return Err(PyValueError::new_err("points must be (n, 2): x and y"));
    }
    if window_size < 16 {
        return Err(PyValueError::new_err("window_size must be at least 16"));
    }
    if !(max_shift > 0.0) || max_iter < 1 {
        return Err(PyValueError::new_err(
            "max_shift must be positive and max_iter at least 1",
        ));
    }
    if filter_level > 3 {
        return Err(PyValueError::new_err("filter_level must be 0, 1, 2 or 3"));
    }
    if !(0.0 < rs_max_outlier && rs_max_outlier < 100.0) {
        return Err(PyValueError::new_err("rs_max_outlier must be between 0 and 100"));
    }
    let cell = cell.unwrap_or((geo.1.abs(), geo.3.abs()));
    if !(cell.0 > 0.0 && cell.1 > 0.0) {
        return Err(PyValueError::new_err("the cell size must be positive"));
    }
    let p = LocalParams {
        coreg: Params {
            window_size,
            window_pos: None,
            max_shift,
            max_iter,
            integer_range,
            nodata,
            fill,
            cell: [cell.0, cell.1],
        },
        filter_level,
        min_reliability,
        rs_max_outlier,
        rs_tolerance,
        rs_max_iter,
        rs_seed,
        rs_arosics_quirk,
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
    let pts: Vec<(f64, f64)> = points
        .as_array()
        .rows()
        .into_iter()
        .map(|row| (row[0], row[1]))
        .collect();
    let g = Geo {
        x0: geo.0,
        dx: geo.1,
        y0: geo.2,
        dy: geo.3,
    };
    let tps = py.detach(|| coreg_local::tie_points(&r, &t, &pts, offset, &g, &p, n_jobs));
    let d = PyDict::new(py);
    macro_rules! col {
        ($name:expr, $f:expr) => {
            d.set_item($name, tps.iter().map($f).collect::<Vec<_>>())?
        };
    }
    col!("index", |t| t.index);
    col!("x_map", |t| t.x_map);
    col!("y_map", |t| t.y_map);
    col!("x_shift_px", |t| t.x_shift_px);
    col!("y_shift_px", |t| t.y_shift_px);
    col!("x_shift_map", |t| t.x_shift_map);
    col!("y_shift_map", |t| t.y_shift_map);
    col!("abs_shift", |t| t.abs_shift);
    col!("angle", |t| t.angle);
    col!("reliability", |t| t.outcome.reliability);
    col!("ssim_before", |t| t.outcome.ssim_before);
    col!("ssim_after", |t| t.outcome.ssim_after);
    col!("ssim_improved", |t| t.ssim_improved);
    col!("success", |t| t.outcome.success);
    col!("reason", |t| t.outcome.failure.map(|f| f.as_str()));
    col!("window", |t| t.outcome.window.to_vec());
    col!("fft_size", |t| t.outcome.fft_size);
    col!("l1", |t| t.l1);
    col!("l2", |t| t.l2);
    col!("l3", |t| t.l3);
    col!("outlier", |t| t.outlier);
    Ok(d)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "coreg")?;
    m.add(
        "__doc__",
        "Co-registration of an image to a reference (AROSICS' COREG: phase correlation of a window; COREG_LOCAL: of \
         the windows of a grid, with the filters of its tie points)",
    )?;
    m.add_function(wrap_pyfunction!(coregister, &m)?)?;
    m.add_function(wrap_pyfunction!(tie_points, &m)?)?;
    Ok(m)
}
