//! zeit._core.warp: warp, coords and lattice_checks, the Python module of the
//! warp kernel.

use numpy::ndarray::Dimension;
use numpy::{
    AllowTypeChange, Element, PyArray1, PyArray3, PyArrayDyn, PyArrayLikeDyn, PyArrayMethods, PyUntypedArray,
    PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyTuple;
use std::borrow::Cow;
use zeit_core::parallel;
use zeit_core::warp::{self as zw, Coords, Raster, Spec, TransformerFunc, CELL};

/// The values of an array-like converted to T, in C order.
fn values<'a, T: Element + Copy>(a: &'a PyArrayLikeDyn<'_, T, AllowTypeChange>) -> Cow<'a, [T]> {
    let view = a.as_array();
    match view.to_slice() {
        Some(s) => Cow::Borrowed(s),
        None => Cow::Owned(view.iter().copied().collect()),
    }
}

/// The thread count of a call: n > 0, or every processor zeit may use (none kept free).
fn threads_of(n_threads: i32) -> usize {
    if n_threads > 0 {
        n_threads as usize
    } else {
        parallel::all_processors().max(1) as usize
    }
}

/// The coordinate arguments, as zeit/_warp.py passes them.
struct CoordArgs<'py> {
    kind: i32,
    offset: f64,
    gt: [f64; 6],
    inv: [f64; 6],
    level: PyArrayLikeDyn<'py, i8, AllowTypeChange>,
    grid0: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    node_off: PyArrayLikeDyn<'py, i64, AllowTypeChange>,
    nodes: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    cells: [i64; 3],
    dense: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    dense_row0: i64,
    dense_col0: i64,
    transformer: usize,
    transformer_args: PyArrayLikeDyn<'py, usize, AllowTypeChange>,
}

/// The arrays of the coordinates, in C order.
struct CoordData<'a> {
    level: Cow<'a, [i8]>,
    grid0: Cow<'a, [f64]>,
    node_off: Cow<'a, [i64]>,
    nodes: Cow<'a, [f64]>,
    dense: Cow<'a, [f64]>,
    args: Cow<'a, [usize]>,
}

impl CoordArgs<'_> {
    fn data(&self) -> CoordData<'_> {
        CoordData {
            level: values(&self.level),
            grid0: values(&self.grid0),
            node_off: values(&self.node_off),
            nodes: values(&self.nodes),
            dense: values(&self.dense),
            args: values(&self.transformer_args),
        }
    }

    /// The source coordinates of the warp kernel's samples (checked): samples
    /// rows r0 .. last_row - 1, columns c0 .. last_col - 1 must be covered.
    fn coords<'a>(
        &self,
        data: &'a CoordData<'a>,
        r0: i64,
        c0: i64,
        last_row: i64,
        last_col: i64,
    ) -> PyResult<Coords<'a>> {
        let err = |m: &str| Err(PyValueError::new_err(m.to_string()));
        let mut c = Coords {
            kind: self.kind,
            offset: self.offset,
            gt: self.gt,
            inv: self.inv,
            ..Default::default()
        };
        if self.kind == zw::LATTICE {
            let [cells_x, cell_x0, cell_y0] = self.cells;
            let n_level = data.level.len() as i64;
            if cells_x < 1 || cell_x0 < 0 || cell_y0 < 0 || n_level % cells_x != 0 {
                return err("bad lattice");
            }
            let cells_y = n_level / cells_x;
            if cell_x0 * CELL > c0
                || cell_y0 * CELL > r0
                || (cell_x0 + cells_x) * CELL < last_col
                || (cell_y0 + cells_y) * CELL < last_row
            {
                return err("the lattice does not cover the destination region");
            }
            if data.node_off.len() != data.level.len() {
                return err("one node offset per lattice cell");
            }
            if data.grid0.len() as i64 != (cells_x + 1) * (cells_y + 1) * 2 {
                return err("grid0 must hold the corners of every lattice cell");
            }
            let n_nodes = data.nodes.len() as i64 / 2;
            for (&lev, &off) in data.level.iter().zip(data.node_off.iter()) {
                if !(0..=4).contains(&lev) {
                    return err("lattice levels go from 0 to 4");
                }
                if lev == 0 {
                    continue;
                }
                let side = CELL / (CELL >> lev) + 1;
                if off < 0 || off + side * side > n_nodes {
                    return err("lattice nodes out of range");
                }
            }
            c.cells_x = cells_x;
            c.cells_y = cells_y;
            c.cell_x0 = cell_x0;
            c.cell_y0 = cell_y0;
            c.level = &data.level;
            c.grid0 = &data.grid0;
            c.node_off = &data.node_off;
            c.nodes = &data.nodes;
        } else if self.kind == zw::DENSE {
            let shape = self.dense.as_array().raw_dim();
            let shape = shape.slice();
            if shape.len() != 3 || shape[2] != 2 {
                return err("dense coordinates must be (rows, columns, 2)");
            }
            if self.dense_row0 > r0
                || self.dense_col0 > c0
                || self.dense_row0 + (shape[0] as i64) < last_row
                || self.dense_col0 + (shape[1] as i64) < last_col
            {
                return err("the dense coordinates do not cover the destination region");
            }
            c.dense = &data.dense;
            c.dense_row0 = self.dense_row0;
            c.dense_col0 = self.dense_col0;
            c.dense_stride = shape[1] as i64;
        } else if self.kind != zw::AFFINE && self.kind != zw::EXACT {
            return err("unknown coordinate kind");
        }
        if self.transformer != 0 {
            if data.args.is_empty() {
                return err("one transformer argument per thread");
            }
            // SAFETY: the caller gives the address of a GDALTransformerFunc.
            c.transformer = Some(unsafe { std::mem::transmute::<usize, TransformerFunc>(self.transformer) });
            c.transformer_args = &data.args;
        } else if self.kind == zw::EXACT {
            return err("exact coordinates need the transformer");
        }
        Ok(c)
    }
}

/// Warps the typed source, if it is of type T, into a new array of its type.
fn run<'py, T: Raster + Element>(
    py: Python<'py>,
    src: &Bound<'py, PyUntypedArray>,
    coords: &Coords,
    sp: &Spec,
    threads: usize,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Ok(a) = src.as_any().cast::<PyArrayDyn<T>>() else {
        return Ok(None);
    };
    let ro = a.readonly();
    let data = ro.as_slice()?;
    let (bands, h, w) = (sp.bands as usize, sp.dst_h as usize, sp.dst_w as usize);
    let n = bands * h * w;
    // Not zeroed first (one thread through the whole array): warp_uninit
    // writes every cell, on the threads that compute them.
    let mut out: Vec<T> = Vec::with_capacity(n);
    py.detach(|| zw::warp_uninit(data, &mut out.spare_capacity_mut()[..n], coords, sp, threads));
    // SAFETY: warp_uninit wrote the n cells.
    unsafe { out.set_len(n) };
    let out: Bound<'py, PyArray3<T>> = PyArray1::from_vec(py, out).reshape([bands, h, w])?;
    Ok(Some(out.into_any()))
}

/// GDAL's warp kernel on a region of the destination: (bands, rows, columns).
#[pyfunction]
#[pyo3(signature = (src, method, src_shape, window, scales, has_src_nodata, src_nodata, dst_nodata, region, kind,
                    offset, gt, inv, level, grid0, node_off, nodes, cells, dense, dense_row0, dense_col0,
                    transformer, transformer_args, per_band = true, n_threads = 0))]
#[allow(clippy::too_many_arguments)]
fn warp<'py>(
    py: Python<'py>,
    src: &Bound<'py, PyAny>,
    method: i32,
    src_shape: [i32; 2],
    window: [i32; 2],
    scales: [f64; 2],
    has_src_nodata: bool,
    src_nodata: f64,
    dst_nodata: f64,
    region: [i32; 4],
    kind: i32,
    offset: f64,
    gt: [f64; 6],
    inv: [f64; 6],
    level: PyArrayLikeDyn<'py, i8, AllowTypeChange>,
    grid0: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    node_off: PyArrayLikeDyn<'py, i64, AllowTypeChange>,
    nodes: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    cells: [i64; 3],
    dense: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    dense_row0: i64,
    dense_col0: i64,
    transformer: usize,
    transformer_args: PyArrayLikeDyn<'py, usize, AllowTypeChange>,
    per_band: bool,
    n_threads: i32,
) -> PyResult<Bound<'py, PyAny>> {
    let src = match src.cast::<PyUntypedArray>() {
        Ok(a) => a.clone(),
        Err(_) => py
            .import("numpy")?
            .call_method1("asarray", (src,))?
            .cast_into::<PyUntypedArray>()?,
    };
    if src.ndim() != 3 {
        return Err(PyValueError::new_err("src must be (bands, rows, columns)"));
    }
    if !src.is_c_contiguous() {
        return Err(PyValueError::new_err("src must be C-contiguous"));
    }
    let shape = src.shape();
    let sp = Spec {
        method,
        bands: shape[0] as i32,
        src_h: src_shape[0],
        src_w: src_shape[1],
        win_r0: window[0],
        win_c0: window[1],
        win_h: shape[1] as i32,
        win_w: shape[2] as i32,
        xscale: scales[0],
        yscale: scales[1],
        has_src_nodata,
        per_band,
        src_nodata,
        dst_nodata,
        dst_r0: region[0],
        dst_c0: region[1],
        dst_h: region[2],
        dst_w: region[3],
    };
    if !zw::valid_method(method) {
        return Err(PyValueError::new_err("unknown warp method"));
    }
    if sp.bands < 1 || sp.dst_h < 0 || sp.dst_w < 0 || sp.dst_r0 < 0 || sp.dst_c0 < 0 {
        return Err(PyValueError::new_err("bad destination region"));
    }
    if sp.win_r0 < 0
        || sp.win_c0 < 0
        || sp.win_r0 as i64 + sp.win_h as i64 > sp.src_h as i64
        || sp.win_c0 as i64 + sp.win_w as i64 > sp.src_w as i64
    {
        return Err(PyValueError::new_err("the source window must lie in the source"));
    }
    if !(scales[0] > 0.0 && scales[1] > 0.0) {
        return Err(PyValueError::new_err("scales must be positive"));
    }
    // Samples needed: rows r0 .. r0 + h (+ 1 for the corners of the area methods).
    let extra = i64::from(zw::by_area(method));
    let last_row = sp.dst_r0 as i64 + sp.dst_h as i64 + extra; // exclusive
    let last_col = sp.dst_c0 as i64 + sp.dst_w as i64 + extra;
    let args = CoordArgs {
        kind,
        offset,
        gt,
        inv,
        level,
        grid0,
        node_off,
        nodes,
        cells,
        dense,
        dense_row0,
        dense_col0,
        transformer,
        transformer_args,
    };
    let data = args.data();
    let c = args.coords(&data, sp.dst_r0 as i64, sp.dst_c0 as i64, last_row, last_col)?;
    let threads = threads_of(n_threads);
    if c.transformer.is_some() && c.transformer_args.len() < threads {
        return Err(PyValueError::new_err("one transformer argument per thread"));
    }
    macro_rules! try_type {
        ($($t:ty),*) => {
            $(if let Some(out) = run::<$t>(py, &src, &c, &sp, threads)? {
                return Ok(out);
            })*
        };
    }
    try_type!(u64, u8, i8, u16, i16, u32, i32, i64, f32, f64);
    Err(PyTypeError::new_err(
        "unsupported raster type: use (u)int8/16/32, int64, float32 or float64",
    ))
}

/// The source coordinates the warp kernel takes for the samples of a region: (rows, columns, 2).
#[pyfunction]
#[pyo3(signature = (kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense, dense_row0, dense_col0,
                    transformer, transformer_args, region))]
#[allow(clippy::too_many_arguments)]
fn coords<'py>(
    py: Python<'py>,
    kind: i32,
    offset: f64,
    gt: [f64; 6],
    inv: [f64; 6],
    level: PyArrayLikeDyn<'py, i8, AllowTypeChange>,
    grid0: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    node_off: PyArrayLikeDyn<'py, i64, AllowTypeChange>,
    nodes: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    cells: [i64; 3],
    dense: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    dense_row0: i64,
    dense_col0: i64,
    transformer: usize,
    transformer_args: PyArrayLikeDyn<'py, usize, AllowTypeChange>,
    region: [i64; 4],
) -> PyResult<Bound<'py, PyArray3<f64>>> {
    let [r0, c0, nr, nc] = region;
    if r0 < 0 || c0 < 0 || nr < 0 || nc < 0 || nc > i32::MAX as i64 {
        return Err(PyValueError::new_err("bad region"));
    }
    let args = CoordArgs {
        kind,
        offset,
        gt,
        inv,
        level,
        grid0,
        node_off,
        nodes,
        cells,
        dense,
        dense_row0,
        dense_col0,
        transformer,
        transformer_args,
    };
    let data = args.data();
    let c = args.coords(&data, r0, c0, r0 + nr, c0 + nc)?;
    let (nr, nc) = (nr as usize, nc as usize);
    let out = py.detach(|| {
        let mut out = vec![0.0; nr * nc * 2];
        let (mut x, mut y) = (vec![0.0; nc], vec![0.0; nc]);
        for j in 0..nr {
            c.row(r0 + j as i64, c0, &mut x, &mut y);
            let o = &mut out[2 * j * nc..2 * (j + 1) * nc];
            for i in 0..nc {
                o[2 * i] = x[i];
                o[2 * i + 1] = y[i];
            }
        }
        out
    });
    PyArray1::from_vec(py, out).reshape([nr, nc, 2])
}

/// Checks of lattice cells: (err, bad, outside, bbox), one per cell of every grid.
#[pyfunction]
#[pyo3(signature = (g, src_h, src_w, n_threads = 0))]
fn lattice_checks<'py>(
    py: Python<'py>,
    g: PyArrayLikeDyn<'py, f64, AllowTypeChange>,
    src_h: i32,
    src_w: i32,
    n_threads: i32,
) -> PyResult<Bound<'py, PyTuple>> {
    let shape = g.as_array().shape().to_vec();
    if shape.len() != 4 || shape[3] != 2 || shape[1] % 2 != 1 || shape[2] % 2 != 1 || shape[1] < 3 || shape[2] < 3 {
        return Err(PyValueError::new_err("g must be (n, 2 my + 1, 2 mx + 1, 2)"));
    }
    let (n, my, mx) = (shape[0], (shape[1] - 1) / 2, (shape[2] - 1) / 2);
    let gv = values(&g);
    let cells = n * my * mx;
    let threads = threads_of(n_threads);
    let (err, bad, outside, bbox) = py.detach(|| {
        let (mut err, mut bad, mut outside, mut bbox) = (
            vec![0.0; cells],
            vec![false; cells],
            vec![false; cells],
            vec![0.0; 4 * cells],
        );
        zw::lattice_checks(
            &gv,
            n,
            my,
            mx,
            src_h,
            src_w,
            &mut err,
            &mut bad,
            &mut outside,
            &mut bbox,
            threads,
        );
        (err, bad, outside, bbox)
    });
    let err = PyArray1::from_vec(py, err).reshape([n, my, mx])?;
    let bad = PyArray1::from_vec(py, bad).reshape([n, my, mx])?;
    let outside = PyArray1::from_vec(py, outside).reshape([n, my, mx])?;
    let bbox = PyArray1::from_vec(py, bbox).reshape([n, my, mx, 4])?;
    PyTuple::new(
        py,
        [err.into_any(), bad.into_any(), outside.into_any(), bbox.into_any()],
    )
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "warp")?;
    m.add("__doc__", "GDAL's warp kernel on fixed source coordinates")?;
    m.add_function(wrap_pyfunction!(warp, &m)?)?;
    m.add_function(wrap_pyfunction!(coords, &m)?)?;
    m.add_function(wrap_pyfunction!(lattice_checks, &m)?)?;
    Ok(m)
}
