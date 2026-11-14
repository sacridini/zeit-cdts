//! zeit._core.fields: the Python module of zeit-core's field-boundary building blocks (private;
//! `zeit/_fields_core.py` wraps it). Arrays come in with the exact dtype asked (the wrapper casts).

use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use zeit_core::fields::{intensity, label, morph, points, watershed as ws};

use crate::arrays::c_order;

fn err(e: String) -> PyErr {
    PyValueError::new_err(e)
}

/// Marker-based watershed equal to skimage.segmentation.watershed -> labels int32 (y, x).
#[pyfunction]
#[pyo3(signature = (image, markers, connectivity = 1, mask = None))]
fn watershed<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    markers: PyReadonlyArray2<'py, i32>,
    connectivity: usize,
    mask: Option<PyReadonlyArray2<'py, bool>>,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let (ny, nx) = image.as_array().dim();
    let (iv, mv) = (image.as_array(), markers.as_array());
    let (img, mk) = (c_order(&iv), c_order(&mv));
    let mview = mask.as_ref().map(|m| m.as_array());
    let mvals = mview.as_ref().map(c_order);
    let out = py
        .detach(|| ws::watershed(&img, &mk, ny, nx, connectivity, mvals.as_deref()))
        .map_err(err)?;
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// Grayscale morphology of an image (y, x) float64 with an odd boolean footprint; `op` is one of
/// dilation, erosion, opening, closing, white_tophat, black_tophat.
#[pyfunction]
#[pyo3(signature = (op, image, footprint, mode = "reflect", cval = 0.0))]
fn grey_morphology<'py>(
    py: Python<'py>,
    op: &str,
    image: PyReadonlyArray2<'py, f64>,
    footprint: PyReadonlyArray2<'py, bool>,
    mode: &str,
    cval: f64,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let (ny, nx) = image.as_array().dim();
    let (fr, fc) = footprint.as_array().dim();
    let (iv, fv) = (image.as_array(), footprint.as_array());
    let (img, fpv) = (c_order(&iv), c_order(&fv));
    let fp = morph::Footprint::from_mask(&fpv, fr, fc).map_err(err)?;
    let border = morph::Border::parse(mode, cval).map_err(err)?;
    type MorphFn = fn(&[f64], usize, usize, &morph::Footprint, morph::Border) -> Result<Vec<f64>, String>;
    let f: MorphFn = match op {
        "dilation" => morph::dilation,
        "erosion" => morph::erosion,
        "opening" => morph::opening,
        "closing" => morph::closing,
        "white_tophat" => morph::white_tophat,
        "black_tophat" => morph::black_tophat,
        _ => return Err(PyValueError::new_err(format!("unknown morphology operation {op:?}"))),
    };
    let out = py.detach(|| f(&img, ny, nx, &fp, border)).map_err(err)?;
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// np.percentile (linear) of a 1-D float64 array at each q in 0..100.
#[pyfunction]
fn percentiles<'py>(py: Python<'py>, values: PyReadonlyArray1<'py, f64>, qs: Vec<f64>) -> PyResult<Vec<f64>> {
    let v = values.as_array();
    let v = c_order(&v);
    Ok(py.detach(|| intensity::percentiles(&v, &qs)))
}

/// np.percentile (linear, float64 result) of a 1-D float32 array at each q in 0..100.
#[pyfunction]
fn percentiles_f32<'py>(py: Python<'py>, values: PyReadonlyArray1<'py, f32>, qs: Vec<f64>) -> PyResult<Vec<f64>> {
    let v = values.as_array();
    let v = c_order(&v);
    Ok(py.detach(|| intensity::percentiles_f32(&v, &qs)))
}

/// rescale_intensity of a 1-D float64 array.
#[pyfunction]
fn rescale_intensity<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f64>,
    imin: f64,
    imax: f64,
    omin: f64,
    omax: f64,
) -> Bound<'py, PyArray1<f64>> {
    let v = values.as_array();
    let v = c_order(&v);
    let out = py.detach(|| intensity::rescale_intensity(&v, imin, imax, omin, omax));
    PyArray1::from_vec(py, out)
}

/// rescale_intensity of a 1-D float32 array, in float32 arithmetic as numpy does.
#[pyfunction]
fn rescale_intensity_f32<'py>(
    py: Python<'py>,
    values: PyReadonlyArray1<'py, f32>,
    imin: f64,
    imax: f64,
    omin: f64,
    omax: f64,
) -> Bound<'py, PyArray1<f32>> {
    let v = values.as_array();
    let v = c_order(&v);
    let out = py.detach(|| intensity::rescale_intensity_f32(&v, imin, imax, omin, omax));
    PyArray1::from_vec(py, out)
}

/// apply_hysteresis_threshold of an image (y, x) float64 -> bool (y, x).
#[pyfunction]
fn hysteresis_threshold<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, f64>,
    low: f64,
    high: f64,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let (ny, nx) = image.as_array().dim();
    let v = image.as_array();
    let v = c_order(&v);
    let out = py
        .detach(|| label::hysteresis_threshold(&v, ny, nx, low, high))
        .map_err(err)?;
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// find_boundaries(labels, connectivity, mode="thick") of int32 labels (y, x) -> bool (y, x).
#[pyfunction]
#[pyo3(signature = (labels, connectivity = 1))]
fn find_boundaries<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'py, i32>,
    connectivity: usize,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let (ny, nx) = labels.as_array().dim();
    let v = labels.as_array();
    let v = c_order(&v);
    let out = py
        .detach(|| label::find_boundaries(&v, ny, nx, connectivity))
        .map_err(err)?;
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// Components of equal value (same_value) or of any non-zero value of int32 (y, x) -> (labels, n).
#[pyfunction]
#[pyo3(signature = (image, connectivity = 1, same_value = false))]
fn label_ints<'py>(
    py: Python<'py>,
    image: PyReadonlyArray2<'py, i32>,
    connectivity: usize,
    same_value: bool,
) -> PyResult<(Bound<'py, PyArray2<i32>>, usize)> {
    let (ny, nx) = image.as_array().dim();
    let v = image.as_array();
    let v = c_order(&v);
    let (out, n) = py
        .detach(|| label::label_ints(&v, ny, nx, connectivity, same_value))
        .map_err(err)?;
    Ok((PyArray1::from_vec(py, out).reshape([ny, nx])?, n))
}

/// relabel_sequential of int32 labels (y, x) -> (labels, old values in the order of the new ones).
#[pyfunction]
#[pyo3(signature = (labels, offset = 1))]
fn relabel_sequential<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'py, i32>,
    offset: i32,
) -> PyResult<(Bound<'py, PyArray2<i32>>, Vec<i32>)> {
    let (ny, nx) = labels.as_array().dim();
    let v = labels.as_array();
    let v = c_order(&v);
    let (out, inv) = py.detach(|| label::relabel_sequential(&v, offset)).map_err(err)?;
    Ok((PyArray1::from_vec(py, out).reshape([ny, nx])?, inv))
}

/// The labels of int32 (y, x) with max_size pixels or fewer set to 0.
#[pyfunction]
fn remove_small_labels<'py>(
    py: Python<'py>,
    labels: PyReadonlyArray2<'py, i32>,
    max_size: usize,
) -> PyResult<Bound<'py, PyArray2<i32>>> {
    let (ny, nx) = labels.as_array().dim();
    let v = labels.as_array();
    let v = c_order(&v);
    let out = py.detach(|| label::remove_small_labels(&v, max_size)).map_err(err)?;
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// The components of a bool (y, x) with max_size pixels or fewer cleared.
#[pyfunction]
#[pyo3(signature = (mask, max_size, connectivity = 1))]
fn remove_small_objects_mask<'py>(
    py: Python<'py>,
    mask: PyReadonlyArray2<'py, bool>,
    max_size: usize,
    connectivity: usize,
) -> PyResult<Bound<'py, PyArray2<bool>>> {
    let (ny, nx) = mask.as_array().dim();
    let v = mask.as_array();
    let v = c_order(&v);
    let out = py
        .detach(|| label::remove_small_objects_mask(&v, ny, nx, max_size, connectivity))
        .map_err(err)?;
    PyArray1::from_vec(py, out).reshape([ny, nx])
}

/// The pairs (i, j), i < j, of points within `distance`, sorted.
#[pyfunction]
fn close_pairs<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    distance: f64,
) -> PyResult<Vec<(u32, u32)>> {
    let (xv, yv) = (x.as_array(), y.as_array());
    let (xs, ys) = (c_order(&xv), c_order(&yv));
    py.detach(|| points::close_pairs(&xs, &ys, distance)).map_err(err)
}

/// The indices kept by the notebook's greedy thinning over `pairs` in the given order.
#[pyfunction]
fn trim_close_points_pairs(n: usize, pairs: Vec<(u32, u32)>) -> PyResult<Vec<u32>> {
    if pairs.iter().any(|&(i, j)| i as usize >= n || j as usize >= n) {
        return Err(PyValueError::new_err("a pair indexes a point out of range"));
    }
    Ok(points::trim_close_points_pairs(n, &pairs))
}

/// The indices kept by the greedy thinning with the pairs sorted by (i, j).
#[pyfunction]
fn trim_close_points<'py>(
    py: Python<'py>,
    x: PyReadonlyArray1<'py, f64>,
    y: PyReadonlyArray1<'py, f64>,
    distance: f64,
) -> PyResult<Vec<u32>> {
    let (xv, yv) = (x.as_array(), y.as_array());
    let (xs, ys) = (c_order(&xv), c_order(&yv));
    py.detach(|| points::trim_close_points(&xs, &ys, distance)).map_err(err)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "fields")?;
    m.add(
        "__doc__",
        "Field-boundary building blocks: watershed, grey morphology, hysteresis, labels, marker thinning",
    )?;
    m.add_function(wrap_pyfunction!(watershed, &m)?)?;
    m.add_function(wrap_pyfunction!(grey_morphology, &m)?)?;
    m.add_function(wrap_pyfunction!(percentiles, &m)?)?;
    m.add_function(wrap_pyfunction!(percentiles_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(rescale_intensity, &m)?)?;
    m.add_function(wrap_pyfunction!(rescale_intensity_f32, &m)?)?;
    m.add_function(wrap_pyfunction!(hysteresis_threshold, &m)?)?;
    m.add_function(wrap_pyfunction!(find_boundaries, &m)?)?;
    m.add_function(wrap_pyfunction!(label_ints, &m)?)?;
    m.add_function(wrap_pyfunction!(relabel_sequential, &m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_labels, &m)?)?;
    m.add_function(wrap_pyfunction!(remove_small_objects_mask, &m)?)?;
    m.add_function(wrap_pyfunction!(close_pairs, &m)?)?;
    m.add_function(wrap_pyfunction!(trim_close_points_pairs, &m)?)?;
    m.add_function(wrap_pyfunction!(trim_close_points, &m)?)?;
    Ok(m)
}
