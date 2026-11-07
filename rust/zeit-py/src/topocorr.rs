//! zeit._core.topocorr: the Python module (slope and aspect by Horn, C-correction and SCS+C).

use numpy::{PyArray1, PyArray2, PyArray4, PyArrayMethods, PyReadonlyArray2, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::topocorr::{self, Method};

/// Slope and aspect, in radians, of the DEM `z` `(y, x)` float64: `dx` the easting step per column and `dy` the
/// northing step per row (negative for north-up), in the unit of `z`. `(slope, aspect)` float32, NaN slope where the
/// 3x3 window has a non-finite value, aspect 0 where flat, clockwise from north.
#[pyfunction]
#[pyo3(signature = (z, dx, dy, n_jobs = -1))]
fn slope_aspect<'py>(
    py: Python<'py>,
    z: PyReadonlyArray2<'py, f64>,
    dx: f64,
    dy: f64,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
    if !(dx.is_finite() && dy.is_finite() && dx != 0.0 && dy != 0.0) {
        return Err(PyValueError::new_err("dx and dy must be finite and non-zero"));
    }
    let [ny, nx] = *z.shape() else { unreachable!() };
    let za = z.as_array();
    let zs = za
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("the DEM must be C-contiguous"))?;
    let (s, a) = py.detach(|| topocorr::slope_aspect(zs, ny, nx, dx, dy, n_jobs));
    Ok((
        PyArray1::from_vec(py, s).reshape([ny, nx])?,
        PyArray1::from_vec(py, a).reshape([ny, nx])?,
    ))
}

/// The topographic correction of `x` `(time, band, y, x)` float32 (see `zeit_core::topocorr::topo_correct`):
/// `slope`, `aspect` `(y, x)` float32 radians; `zenith`, `azimuth` `(time, ns)` float32 degrees, `ns` 1 or `y * x`;
/// `sample` `(1 or time, y * x)` uint8 or None; `method` `"c"` or `"scs+c"`. `(corrected, fits)`, `fits`
/// `(time, band, 5)` float64: c, slope m, intercept b, sample size, pixels left uncorrected by the guards.
#[pyfunction]
#[pyo3(signature = (x, slope, aspect, zenith, azimuth, sample = None, method = "c", min_slope = 0.0, min_sample = 100, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn correct<'py>(
    py: Python<'py>,
    x: PyReadonlyArray4<'py, f32>,
    slope: PyReadonlyArray2<'py, f32>,
    aspect: PyReadonlyArray2<'py, f32>,
    zenith: PyReadonlyArray2<'py, f32>,
    azimuth: PyReadonlyArray2<'py, f32>,
    sample: Option<PyReadonlyArray2<'py, u8>>,
    method: &str,
    min_slope: f32,
    min_sample: usize,
    n_jobs: i32,
) -> PyResult<(Bound<'py, PyArray4<f32>>, Bound<'py, PyArray2<f64>>)> {
    let [nt, nb, ny, nx] = *x.shape() else { unreachable!() };
    let npix = ny * nx;
    let method = match method {
        "c" => Method::C,
        "scs+c" => Method::ScsC,
        other => {
            return Err(PyValueError::new_err(format!(
                "method must be 'c' or 'scs+c', got {other:?}"
            )))
        }
    };
    if slope.shape() != [ny, nx] || aspect.shape() != [ny, nx] {
        return Err(PyValueError::new_err("slope and aspect must be (y, x) like the cube"));
    }
    let zs = zenith.shape();
    if zs[0] != nt || (zs[1] != 1 && zs[1] != npix) || azimuth.shape() != zs {
        return Err(PyValueError::new_err(
            "zenith and azimuth must be (time, 1) or (time, y * x) and of one shape",
        ));
    }
    let ns = zs[1];
    let xa = x.as_array();
    let xs = xa
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("the cube must be C-contiguous"))?;
    let (sa, aa, za, wa) = (
        slope.as_array(),
        aspect.as_array(),
        zenith.as_array(),
        azimuth.as_array(),
    );
    let need = || PyValueError::new_err("arrays must be C-contiguous");
    let (sl, asp, zen, azi) = (
        sa.to_slice().ok_or_else(need)?,
        aa.to_slice().ok_or_else(need)?,
        za.to_slice().ok_or_else(need)?,
        wa.to_slice().ok_or_else(need)?,
    );
    let sm = sample.as_ref().map(|m| m.as_array());
    let sample_arg = match &sm {
        Some(m) => {
            let sh = m.shape();
            if sh[1] != npix || (sh[0] != 1 && sh[0] != nt) {
                return Err(PyValueError::new_err("sample must be (1 or time, y * x)"));
            }
            Some((m.to_slice().ok_or_else(need)?, sh[0]))
        }
        None => None,
    };
    // SAFETY: the core writes every value.
    let out = unsafe { PyArray4::<f32>::new(py, [nt, nb, ny, nx], false) };
    let fits = {
        let mut w = out.readwrite();
        let o = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        py.detach(|| {
            topocorr::topo_correct(
                xs, nt, nb, npix, sl, asp, zen, azi, ns, sample_arg, method, min_slope, min_sample, o, n_jobs,
            )
        })
    };
    let flat: Vec<f64> = fits
        .iter()
        .flat_map(|f| [f.c, f.slope, f.intercept, f.n as f64, f.skipped as f64])
        .collect();
    Ok((out, PyArray1::from_vec(py, flat).reshape([nt * nb, 5])?))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "topocorr")?;
    m.add(
        "__doc__",
        "Topographic correction: Horn's slope and aspect, C-correction and SCS+C",
    )?;
    m.add_function(wrap_pyfunction!(slope_aspect, &m)?)?;
    m.add_function(wrap_pyfunction!(correct, &m)?)?;
    Ok(m)
}
