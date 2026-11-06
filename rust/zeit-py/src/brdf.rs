//! zeit._core.brdf: the Python module (NBAR by the c-factor, Roy et al. 2016, 2017).

use numpy::{Element, PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::brdf::{self, Geometry, Real, RossLi};

/// The (bands, pixels) reflectance, or the (4, pixels) angles: float32 as it is, float64 as it is.
#[derive(FromPyObject)]
enum Values<'py> {
    F32(PyReadonlyArray2<'py, f32>),
    F64(PyReadonlyArray2<'py, f64>),
}

fn coefs_of(coefs: &PyReadonlyArray2<'_, f64>, bands: usize) -> PyResult<Vec<RossLi>> {
    let c = coefs.as_array();
    if c.dim() != (bands, 3) {
        return Err(PyValueError::new_err(format!(
            "coefs must be (bands, 3) = ({bands}, 3): f_iso, f_geo, f_vol of each band"
        )));
    }
    Ok(c.rows().into_iter().map(|r| RossLi::new(r[0], r[1], r[2])).collect())
}

#[allow(clippy::too_many_arguments)]
fn run<'py, T: Real + Element>(
    py: Python<'py>,
    refl: Option<&PyReadonlyArray2<'py, T>>,
    angles: &PyReadonlyArray2<'py, T>,
    coefs: &[RossLi],
    sza_norm: f64,
    n_jobs: i32,
    with_c: bool,
) -> PyResult<(Bound<'py, PyArray2<T>>, Option<Bound<'py, PyArray2<T>>>)> {
    let a = angles.as_array();
    let n = a.dim().1;
    let rows: Vec<&[T]> = (0..4)
        .map(|i| {
            a.row(i)
                .to_slice()
                .ok_or_else(|| PyValueError::new_err("angles must be C-contiguous"))
        })
        .collect::<PyResult<_>>()?;
    let geo = Geometry { sza: rows[0], saa: rows[1], vza: rows[2], vaa: rows[3] };
    let nb = coefs.len();
    // SAFETY: every value is written by the core before anything reads it.
    let out = unsafe { PyArray2::<T>::new(py, [nb, n], false) };
    let c_out = with_c.then(|| unsafe { PyArray2::<T>::new(py, [nb, n], false) });
    {
        let mut o = out.readwrite();
        let o = o.as_slice_mut().map_err(|_| PyValueError::new_err("output not contiguous"))?;
        let mut cw = c_out.as_ref().map(|c| c.readwrite());
        let cs = match cw.as_mut() {
            Some(w) => Some(w.as_slice_mut().map_err(|_| PyValueError::new_err("output not contiguous"))?),
            None => None,
        };
        match refl {
            Some(r) => {
                let ra = r.as_array();
                let r = ra
                    .to_slice()
                    .ok_or_else(|| PyValueError::new_err("refl must be C-contiguous"))?;
                py.detach(|| brdf::nbar_bands(r, &geo, coefs, sza_norm, o, cs, n_jobs));
            }
            None => py.detach(|| brdf::cfactor_bands(&geo, coefs, sza_norm, o, n_jobs)),
        }
    }
    Ok((out, c_out))
}

fn check(refl_bands: Option<usize>, refl_pixels: Option<usize>, angles: (usize, usize)) -> PyResult<()> {
    if angles.0 != 4 {
        return Err(PyValueError::new_err("angles must be (4, pixels): SZA, SAA, VZA, VAA in degrees"));
    }
    if let (Some(_), Some(p)) = (refl_bands, refl_pixels) {
        if p != angles.1 {
            return Err(PyValueError::new_err("refl and angles must have the same pixels"));
        }
    }
    Ok(())
}

/// The NBAR of every band of `refl` (bands, pixels) from `angles` (4, pixels: SZA, SAA, VZA, VAA, degrees; the
/// same dtype as `refl`, float32 or float64) with `coefs` (bands, 3: f_iso, f_geo, f_vol) and the normalized solar
/// zenith `sza_norm` (degrees): `(bands, pixels)`, the reflectance times its c-factor, NaN where either is NaN or
/// the geometry is outside the model's domain. With `return_cfactor`, `(nbar, c)`.
#[pyfunction]
#[pyo3(signature = (refl, angles, coefs, sza_norm, n_jobs = -1, return_cfactor = false))]
fn nbar_bands<'py>(
    py: Python<'py>,
    refl: Values<'py>,
    angles: Values<'py>,
    coefs: PyReadonlyArray2<'py, f64>,
    sza_norm: f64,
    n_jobs: i32,
    return_cfactor: bool,
) -> PyResult<Py<PyAny>> {
    match (&refl, &angles) {
        (Values::F32(r), Values::F32(a)) => {
            check(Some(r.shape()[0]), Some(r.shape()[1]), (a.shape()[0], a.shape()[1]))?;
            let cf = coefs_of(&coefs, r.shape()[0])?;
            let (o, c) = run(py, Some(r), a, &cf, sza_norm, n_jobs, return_cfactor)?;
            Ok(pack(py, o, c))
        }
        (Values::F64(r), Values::F64(a)) => {
            check(Some(r.shape()[0]), Some(r.shape()[1]), (a.shape()[0], a.shape()[1]))?;
            let cf = coefs_of(&coefs, r.shape()[0])?;
            let (o, c) = run(py, Some(r), a, &cf, sza_norm, n_jobs, return_cfactor)?;
            Ok(pack(py, o, c))
        }
        _ => Err(PyValueError::new_err("refl and angles must both be float32 or both float64")),
    }
}

fn pack<'py, T: Element>(py: Python<'py>, o: Bound<'py, PyArray2<T>>, c: Option<Bound<'py, PyArray2<T>>>) -> Py<PyAny> {
    match c {
        Some(c) => (o, c).into_pyobject(py).unwrap().into_any().unbind(),
        None => o.into_any().unbind(),
    }
}

/// The c-factor of every pixel for each row of `coefs` (bands, 3): `(bands, pixels)`.
#[pyfunction]
#[pyo3(signature = (angles, coefs, sza_norm, n_jobs = -1))]
fn cfactor_bands<'py>(
    py: Python<'py>,
    angles: Values<'py>,
    coefs: PyReadonlyArray2<'py, f64>,
    sza_norm: f64,
    n_jobs: i32,
) -> PyResult<Py<PyAny>> {
    let bands = coefs.shape()[0];
    match &angles {
        Values::F32(a) => {
            check(None, None, (a.shape()[0], a.shape()[1]))?;
            let cf = coefs_of(&coefs, bands)?;
            Ok(run(py, None, a, &cf, sza_norm, n_jobs, false)?.0.into_any().unbind())
        }
        Values::F64(a) => {
            check(None, None, (a.shape()[0], a.shape()[1]))?;
            let cf = coefs_of(&coefs, bands)?;
            Ok(run(py, None, a, &cf, sza_norm, n_jobs, false)?.0.into_any().unbind())
        }
    }
}

/// RossThick and LiSparse-Reciprocal kernels `(k_vol, k_geo)` for angles in degrees (arrays of one length): the sun
/// zenith, the view zenith and the relative azimuth. NaN outside the model's domain.
#[pyfunction]
fn kernels<'py>(
    py: Python<'py>,
    sza: PyReadonlyArray1<'py, f64>,
    vza: PyReadonlyArray1<'py, f64>,
    raa: PyReadonlyArray1<'py, f64>,
) -> PyResult<(Bound<'py, PyArray1<f64>>, Bound<'py, PyArray1<f64>>)> {
    let (s, v, r) = (sza.as_array(), vza.as_array(), raa.as_array());
    if s.len() != v.len() || s.len() != r.len() {
        return Err(PyValueError::new_err("sza, vza and raa must have the same length"));
    }
    let (kv, kg): (Vec<f64>, Vec<f64>) = s
        .iter()
        .zip(v.iter())
        .zip(r.iter())
        .map(|((s, v), r)| brdf::kernels(s.to_radians(), v.to_radians(), r.to_radians()))
        .unzip();
    Ok((PyArray1::from_vec(py, kv), PyArray1::from_vec(py, kg)))
}

/// The model parameters `(f_iso, f_geo, f_vol)` of a role (`coastal`, `blue`, `green`, `red`, `rededge1-3`, `nir`,
/// `nir08`, `swir1`, `swir2`), or None.
#[pyfunction]
fn role(name: &str) -> Option<(f64, f64, f64)> {
    brdf::role(name).map(|p| (p.f_iso, p.f_geo, p.f_vol))
}

/// The parameters `(f_iso, f_geo, f_vol)` at a wavelength (nm) by the linear interpolation between MODIS's red and
/// NIR (Roy, Li and Zhang 2017, Eq. 3).
#[pyfunction]
fn rededge(lambda_nm: f64) -> (f64, f64, f64) {
    let p = brdf::rededge(lambda_nm);
    (p.f_iso, p.f_geo, p.f_vol)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "brdf")?;
    m.add("__doc__", "NBAR by the c-factor (Roy et al. 2016, 2017)")?;
    m.add("ROLES", brdf::ROLES.to_vec())?;
    m.add_function(wrap_pyfunction!(nbar_bands, &m)?)?;
    m.add_function(wrap_pyfunction!(cfactor_bands, &m)?)?;
    m.add_function(wrap_pyfunction!(kernels, &m)?)?;
    m.add_function(wrap_pyfunction!(role, &m)?)?;
    m.add_function(wrap_pyfunction!(rededge, &m)?)?;
    Ok(m)
}
