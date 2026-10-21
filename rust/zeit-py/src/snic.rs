//! zeit._rust.snic: the bindings of zeit._core.snic.

use numpy::{AllowTypeChange, PyArray1, PyArrayDyn, PyArrayLikeDyn, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use zeit_core::snic::{self, Feature, Image, Segments};

use crate::arrays::c_order;

#[allow(clippy::too_many_arguments)]
fn run<T: Feature + numpy::Element>(
    py: Python<'_>,
    data: &[T],
    shape: &[usize],
    seeds: &[(i64, i64)],
    compactness: f64,
    tile_height: i64,
    tile_width: i64,
    n_jobs: i32,
) -> PyResult<Segments> {
    let img = Image {
        data,
        features: shape[0],
        rows: shape[1],
        cols: shape[2],
    };
    py.detach(|| snic::segment(img, seeds, compactness, tile_height, tile_width, n_jobs))
        .map_err(PyValueError::new_err)
}

/// Run SNIC on a planar [features, rows, cols] image from [n, 2] (row, col)
/// seeds, tiles in parallel: (labels [rows, cols] int32, means [n, features],
/// centroids [n, 2] (row, col), sizes [n] int64).
#[pyfunction]
#[pyo3(signature = (data, seeds, compactness = 10.0, tile_height = 0, tile_width = 0, n_jobs = -1))]
fn snic_segment<'py>(
    py: Python<'py>,
    data: &Bound<'py, PyAny>,
    seeds: PyArrayLikeDyn<'py, i32, AllowTypeChange>,
    compactness: f64,
    tile_height: i64,
    tile_width: i64,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let seeds = seeds.as_array();
    if seeds.ndim() != 2 || seeds.shape()[1] != 2 {
        return Err(PyValueError::new_err("seeds must be [n_seeds, 2] (row, col)"));
    }
    let seeds: Vec<(i64, i64)> = seeds.rows().into_iter().map(|r| (r[0] as i64, r[1] as i64)).collect();

    // float32 cubes are segmented without a float64 copy; the accumulators
    // and distances are float64 either way
    let (seg, shape) = if let Ok(a) = data.cast::<PyArrayDyn<f32>>() {
        if a.ndim() != 3 {
            return Err(PyValueError::new_err("data must be 3D [features, rows, cols]"));
        }
        let ro = a.readonly();
        let view = ro.as_array();
        let shape = view.shape().to_vec();
        let values = view
            .as_slice()
            .map(std::borrow::Cow::Borrowed)
            .unwrap_or_else(|| view.iter().copied().collect::<Vec<_>>().into());
        (
            run(
                py,
                &values,
                &shape,
                &seeds,
                compactness,
                tile_height,
                tile_width,
                n_jobs,
            )?,
            shape,
        )
    } else {
        let a: PyArrayLikeDyn<'py, f64, AllowTypeChange> = data.extract()?;
        let view = a.as_array();
        if view.ndim() != 3 {
            return Err(PyValueError::new_err("data must be 3D [features, rows, cols]"));
        }
        let shape = view.shape().to_vec();
        let values = c_order(&view);
        (
            run(
                py,
                &values,
                &shape,
                &seeds,
                compactness,
                tile_height,
                tile_width,
                n_jobs,
            )?,
            shape,
        )
    };

    let (f, rows, cols, k) = (shape[0], shape[1], shape[2], seeds.len());
    let labels = PyArray1::from_vec(py, seg.labels).reshape([rows, cols])?;
    let means = PyArray1::from_vec(py, seg.means).reshape([k, f])?;
    let centroids = PyArray1::from_vec(py, seg.centroids).reshape([k, 2])?;
    let sizes = PyArray1::from_vec(py, seg.sizes);
    PyTuple::new(
        py,
        [
            labels.into_any(),
            means.into_any(),
            centroids.into_any(),
            sizes.into_any(),
        ],
    )
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "snic")?;
    m.add("__doc__", "SNIC: Simple Non-Iterative Clustering superpixels")?;
    m.add_function(wrap_pyfunction!(snic_segment, &m)?)?;
    Ok(m)
}
