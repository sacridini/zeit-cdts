//! zeit._core.tasseled: the Python module (Tasseled Cap, FORCE's float32 arithmetic).

use numpy::{Element, PyArray4, PyArrayMethods, PyReadonlyArray3, PyReadonlyArray4, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::tasseled::{self, Sample32};

/// The cube `(time, band, y, x)`, C-contiguous: int16, uint16, int32, float32 or float64.
#[derive(FromPyObject)]
enum Cube<'py> {
    I16(PyReadonlyArray4<'py, i16>),
    U16(PyReadonlyArray4<'py, u16>),
    I32(PyReadonlyArray4<'py, i32>),
    F32(PyReadonlyArray4<'py, f32>),
    F64(PyReadonlyArray4<'py, f64>),
}

#[allow(clippy::too_many_arguments)]
fn run<'py, T: Sample32 + Element>(
    py: Python<'py>,
    x: &PyReadonlyArray4<'py, T>,
    src: [usize; 6],
    matrices: &PyReadonlyArray3<'py, f32>,
    rows: &[usize],
    disturbance: Option<[usize; 3]>,
    missing: &[f64],
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    let [nt, nb, ny, nx] = *x.shape() else { unreachable!() };
    let ms = matrices.shape();
    if ms[0] != nt || ms[2] != 6 {
        return Err(PyValueError::new_err(format!(
            "matrices must be (time, rows, 6) = ({nt}, rows, 6), got {ms:?}"
        )));
    }
    let nm = ms[1];
    if src.iter().any(|&b| b >= nb) {
        return Err(PyValueError::new_err("src names a band the cube does not have"));
    }
    if rows.iter().chain(disturbance.iter().flatten()).any(|&r| r >= nm) {
        return Err(PyValueError::new_err("rows names a row the matrices do not have"));
    }
    let xa = x.as_array();
    let xs = xa
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("the cube must be C-contiguous"))?;
    let ma = matrices.as_array();
    let ms = ma
        .to_slice()
        .ok_or_else(|| PyValueError::new_err("matrices must be C-contiguous"))?;
    let ncomp = rows.len() + usize::from(disturbance.is_some());
    // SAFETY: the core writes every value before anything reads it.
    let out = unsafe { PyArray4::<f32>::new(py, [nt, ncomp, ny, nx], false) };
    {
        let mut w = out.readwrite();
        let o = w
            .as_slice_mut()
            .map_err(|_| PyValueError::new_err("output not contiguous"))?;
        py.detach(|| tasseled::tasseled_batch(xs, nb, ny * nx, src, ms, nm, rows, disturbance, missing, o, n_jobs));
    }
    Ok(out)
}

/// The Tasseled Cap of `x` `(time, band, y, x)` (C-contiguous): `matrices` `(time, rows, 6)` float32, the bands
/// `src` (six indices: blue, green, red, NIR, SWIR1, SWIR2), the components `rows` (rows of the matrices) and,
/// with `disturbance` (three rows: brightness, greenness, wetness), `B - G - W` last. `missing` are the values
/// that mean NaN. `(time, component, y, x)` float32.
#[pyfunction]
#[pyo3(signature = (x, src, matrices, rows, disturbance = None, missing = Vec::new(), n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn tasseled_batch<'py>(
    py: Python<'py>,
    x: Cube<'py>,
    src: [usize; 6],
    matrices: PyReadonlyArray3<'py, f32>,
    rows: Vec<usize>,
    disturbance: Option<[usize; 3]>,
    missing: Vec<f64>,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyArray4<f32>>> {
    match &x {
        Cube::I16(a) => run(py, a, src, &matrices, &rows, disturbance, &missing, n_jobs),
        Cube::U16(a) => run(py, a, src, &matrices, &rows, disturbance, &missing, n_jobs),
        Cube::I32(a) => run(py, a, src, &matrices, &rows, disturbance, &missing, n_jobs),
        Cube::F32(a) => run(py, a, src, &matrices, &rows, disturbance, &missing, n_jobs),
        Cube::F64(a) => run(py, a, src, &matrices, &rows, disturbance, &missing, n_jobs),
    }
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "tasseled")?;
    m.add("__doc__", "Tasseled Cap (FORCE's float32 arithmetic)")?;
    m.add_function(wrap_pyfunction!(tasseled_batch, &m)?)?;
    Ok(m)
}
