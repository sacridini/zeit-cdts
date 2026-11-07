//! zeit._core.parallax: the Python module (Sentinel-2 parallax cloud test and the buffer of a binary mask).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::parallax::{self, Scene};

fn slice<'a, T: numpy::Element>(a: &'a PyReadonlyArray2<'_, T>, what: &str) -> PyResult<&'a [T]> {
    a.as_slice()
        .map_err(|_| PyValueError::new_err(format!("{what} must be C-contiguous")))
}

macro_rules! check_shapes {
    ($ny:expr, $nx:expr, $( ($arr:expr, $what:expr) ),+ ) => {
        $(
            if $arr.shape() != [$ny, $nx] {
                return Err(PyValueError::new_err(format!(
                    "{} must be (y, x) = ({}, {}), got {:?}", $what, $ny, $nx, $arr.shape()
                )));
            }
        )+
    };
}

/// The Cloud Displacement Index `(y / 2, x / 2)` float32 (-9999 where a block has none) of one date, and, when
/// `clouds` is true, the cloud mask after the parallax test `(y, x)` bool (else an empty `(0, 0)` one).
/// `re3`, `bnir`, `nir`, `var` are int16 (reflectance times 10000; `var` the variability probability times
/// 10000), `pcp`, `off`, `bright`, `saturated` are bool, all `(y, x)`.
#[pyfunction]
#[pyo3(signature = (re3, bnir, nir, var, pcp, off, bright, saturated, clouds = true, n_jobs = -1))]
#[allow(clippy::too_many_arguments, clippy::type_complexity)]
fn parallax_batch<'py>(
    py: Python<'py>,
    re3: PyReadonlyArray2<'py, i16>,
    bnir: PyReadonlyArray2<'py, i16>,
    nir: PyReadonlyArray2<'py, i16>,
    var: PyReadonlyArray2<'py, i16>,
    pcp: PyReadonlyArray2<'py, bool>,
    off: PyReadonlyArray2<'py, bool>,
    bright: PyReadonlyArray2<'py, bool>,
    saturated: PyReadonlyArray2<'py, bool>,
    clouds: bool,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray2<bool>>, Bound<'py, PyArray2<f32>>)> {
    let [ny, nx] = *re3.shape() else { unreachable!() };
    check_shapes!(
        ny,
        nx,
        (bnir, "bnir"),
        (nir, "nir"),
        (var, "var"),
        (pcp, "pcp"),
        (off, "off"),
        (bright, "bright"),
        (saturated, "saturated")
    );
    let sc = Scene {
        nx,
        ny,
        re3: slice(&re3, "re3")?,
        bnir: slice(&bnir, "bnir")?,
        nir: slice(&nir, "nir")?,
        var: slice(&var, "var")?,
        pcp: slice(&pcp, "pcp")?,
        off: slice(&off, "off")?,
    };
    let (bright, saturated) = (slice(&bright, "bright")?, slice(&saturated, "saturated")?);
    let (cld, cdi) = py.detach(|| {
        if clouds {
            parallax::parallax_clouds(&sc, bright, saturated, n_jobs)
        } else {
            (Vec::new(), parallax::displacement_index(&sc, n_jobs))
        }
    });
    let (ny_, nx_) = (ny / 2, nx / 2);
    let cdi_arr = PyArray1::from_vec(py, cdi).reshape([ny_, nx_])?;
    let cld_arr = if clouds {
        PyArray1::from_vec(py, cld).reshape([ny, nx])?
    } else {
        PyArray1::from_vec(py, Vec::<bool>::new()).reshape([0, 0])?
    };
    Ok((cld_arr, cdi_arr))
}

/// The true pixels of `mask` `(y, x)` bool buffered by `radius` pixels with FORCE's midpoint circle; `force` is
/// FORCE's code to the letter (pixels within `radius` of an edge are lost).
#[pyfunction]
#[pyo3(signature = (mask, radius, force = false, n_jobs = -1))]
fn buffer_batch<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'py, bool>,
    radius: usize,
    force: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let [ny, nx] = *mask.shape() else { unreachable!() };
    let m = slice(&mask, "mask")?;
    let out = py.detach(|| parallax::buffer(m, nx, ny, radius, force, n_jobs));
    Ok(PyArray1::from_vec(py, out).reshape([ny, nx])?)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "parallax")?;
    m.add(
        "__doc__",
        "Sentinel-2 parallax cloud test (Cloud Displacement Index) and the buffer of a binary mask (FORCE)",
    )?;
    m.add_function(wrap_pyfunction!(parallax_batch, &m)?)?;
    m.add_function(wrap_pyfunction!(buffer_batch, &m)?)?;
    Ok(m)
}
