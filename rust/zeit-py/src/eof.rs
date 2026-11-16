//! zeit._core.eof: the Python module of zeit-core's temporal EOF (principal
//! components of a cube): the decomposition, the maps, the PCs of the dates
//! left out of the fit, North's test and the varimax rotation, in one call.

use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike1, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicUsize, Ordering};
use zeit_core::eof::{self, Blocks, Dense, Settings, Solver};
use zeit_core::parallel;

use crate::arrays::c_order;

/// What a call of `decompose` gives, in the layout of the Python side.
struct Out {
    modes: usize,
    dates: usize,
    n: usize,
    singular: Vec<f64>,
    explained: Vec<f64>,
    eigenvalues: Vec<f64>,
    /// modes x dates (all the dates of the cube, NaN where there is no PC).
    pcs: Vec<f64>,
    /// modes x pixels of the cube, NaN where the pixel was not kept.
    eof: Vec<f64>,
    corr: Option<Vec<f64>>,
    north_error: Vec<f64>,
    separable: Vec<bool>,
    pc_filled: Vec<bool>,
    pixel_kept: Vec<bool>,
    date_kept: Vec<bool>,
    total_variance: f64,
    solver: &'static str,
    rotated: Option<Rotated>,
}

struct Rotated {
    m: usize,
    eof: Vec<f64>,
    pcs: Vec<f64>,
    variance: Vec<f64>,
    rotation: Vec<f64>,
    iterations: usize,
}

/// The maps of every block (`V_b`, and with `want_corr` the correlation of
/// each pixel with each PC), the blocks shared among `threads` threads.
fn block_maps(dense: &Dense, d: &eof::Decomposition, threads: usize, want_corr: bool) -> Vec<(usize, Vec<f64>, Vec<f64>)> {
    let nb = dense.block_count();
    let m = d.modes();
    let next = AtomicUsize::new(0);
    let work = || {
        let mut got = Vec::new();
        let mut buf = Vec::new();
        loop {
            let b = next.fetch_add(1, Ordering::Relaxed);
            if b >= nb {
                break;
            }
            let range = dense.block(b, &mut buf);
            let cols = range.len();
            let mut v = vec![0.0; cols * m];
            eof::loadings(d, &buf, cols, &mut v);
            let mut corr = Vec::new();
            if want_corr {
                corr = vec![f64::NAN; cols * m];
                let norm: Vec<f64> = buf
                    .chunks(d.dates)
                    .map(|c| c.iter().map(|x| x * x).sum::<f64>().sqrt())
                    .collect();
                for k in 0..m {
                    for j in 0..cols {
                        if norm[j] > 0.0 {
                            corr[k * cols + j] = v[k * cols + j] * d.singular[k] / norm[j];
                        }
                    }
                }
            }
            got.push((b, v, corr));
        }
        got
    };
    let mut all: Vec<(usize, Vec<f64>, Vec<f64>)> = if threads <= 1 || nb <= 1 {
        work()
    } else {
        std::thread::scope(|s| {
            let hs: Vec<_> = (0..threads.min(nb)).map(|_| s.spawn(&work)).collect();
            hs.into_iter().flat_map(|h| h.join().expect("an EOF worker panicked")).collect()
        })
    };
    all.sort_by_key(|x| x.0);
    all
}

/// The sign rule of `eof::fix_signs` for one unit-variance PC over the kept dates.
fn flip_needed(pc: &[f64], weighted_sum: &[f64], total: f64) -> bool {
    let norm_m = weighted_sum.iter().map(|v| v * v).sum::<f64>().sqrt();
    let noise = norm_m <= 1e-10 * total.max(0.0).sqrt();
    let norm_pc = pc.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dot: f64 = pc.iter().zip(weighted_sum).map(|(a, b)| a * b).sum();
    if noise || dot.abs() < 1e-6 * norm_m * norm_pc {
        pc.iter().fold(0.0f64, |m, &v| if v.abs() > m.abs() { v } else { m }) < 0.0
    } else {
        dot < 0.0
    }
}

#[allow(clippy::too_many_arguments)]
fn run(
    x: &[f64],
    t: usize,
    n: usize,
    weights: Option<&[f64]>,
    s: &Settings,
    min_valid: f64,
    min_date_valid: f64,
    units: bool,
    rotate: usize,
    kaiser: bool,
    autocorrelation: bool,
    want_corr: bool,
    n_jobs: i32,
) -> Result<Out, String> {
    let dense = Dense::new(x, t, n, weights, min_valid, min_date_valid)?;
    let mut d = eof::decompose(&dense, s, n_jobs, &|_| {})?;
    eof::fix_signs(&mut d);
    let m = d.modes();
    if rotate == 1 || rotate > m {
        return Err(format!("rotate must be 0 or between 2 and the {m} modes computed, got {rotate}"));
    }
    let tk = d.dates;
    let f = ((tk.max(2) - 1) as f64).sqrt();
    let kept = dense.kept_pixels();
    let nk = kept.len();
    let sqrt_w = dense.sqrt_weights_all();
    let threads = parallel::threads(n_jobs);

    // the maps: V of the kept pixels, then the cube's grid
    let blocks = block_maps(&dense, &d, threads, want_corr);
    let mut v = vec![0.0; nk * m];
    let mut corr_out = want_corr.then(|| vec![f64::NAN; m * n]);
    let mut at = 0;
    for (_, vb, cb) in &blocks {
        let cols = vb.len() / m.max(1);
        for k in 0..m {
            v[k * nk + at..k * nk + at + cols].copy_from_slice(&vb[k * cols..(k + 1) * cols]);
            if let Some(c) = corr_out.as_mut() {
                for j in 0..cols {
                    c[k * n + kept[at + j]] = cb[k * cols + j];
                }
            }
        }
        at += cols;
    }
    drop(blocks);
    let mut eof_map = vec![f64::NAN; m * n];
    for k in 0..m {
        let scale = d.singular[k] / f;
        for p in 0..nk {
            eof_map[k * n + kept[p]] = if units {
                if sqrt_w[p] > 0.0 {
                    v[k * nk + p] * scale / sqrt_w[p]
                } else {
                    f64::NAN
                }
            } else {
                v[k * nk + p]
            };
        }
    }

    // the PCs of every date: the fit's, and the gappy projection for the others
    let unit_kept = d.pcs(true);
    let mut unit_all = vec![f64::NAN; m * t];
    let mut date_kept = vec![false; t];
    for (i, &ti) in dense.kept_dates().iter().enumerate() {
        date_kept[ti] = true;
        for k in 0..m {
            unit_all[k * t + ti] = unit_kept[k * tk + i];
        }
    }
    let mut pc_filled = vec![false; t];
    let mut row = Vec::new();
    for ti in (0..t).filter(|&ti| !date_kept[ti]) {
        dense.weighted_row(ti, &mut row);
        let coef = eof::project_gappy(&v, m, &row);
        for k in 0..m {
            if d.singular[k] > 0.0 {
                unit_all[k * t + ti] = coef[k] / d.singular[k] * f;
            }
        }
        pc_filled[ti] = coef.iter().all(|c| c.is_finite());
    }
    let pcs = if units {
        unit_all.clone()
    } else {
        let mut p = unit_all.clone();
        for (k, col) in p.chunks_mut(t).enumerate() {
            let c = d.singular[k] / f;
            col.iter_mut().for_each(|x| *x *= c);
        }
        p
    };

    // North's test
    let eigen = d.eigenvalues();
    let n_star: Vec<f64> = (0..m)
        .map(|k| {
            if autocorrelation {
                eof::effective_dates(tk, eof::lag1(&unit_kept[k * tk..(k + 1) * tk])).max(1e-9)
            } else {
                tk as f64
            }
        })
        .collect();
    let next_eigen = d.next_singular * d.next_singular / (tk.max(2) - 1) as f64;
    let nr = eof::north(&eigen, &n_star, next_eigen);

    // varimax
    let rotated = if rotate >= 2 {
        let mut l = vec![0.0; nk * rotate];
        for k in 0..rotate {
            for p in 0..nk {
                l[k * nk + p] = v[k * nk + p] * d.singular[k] / f;
            }
        }
        let (r, iterations) = eof::varimax(&mut l, nk, rotate, kaiser, 1e-8, 1000);
        let total_var = d.total / (tk.max(2) - 1) as f64;
        let var: Vec<f64> = (0..rotate)
            .map(|k| l[k * nk..(k + 1) * nk].iter().map(|a| a * a).sum::<f64>() / total_var)
            .collect();
        let mut order: Vec<usize> = (0..rotate).collect();
        order.sort_by(|&a, &b| var[b].total_cmp(&var[a]));
        let z = eof::rotate(&unit_all[..rotate * t], t, rotate, &r);
        let mut rpcs = vec![0.0; rotate * t];
        let mut reof = vec![f64::NAN; rotate * n];
        let mut rrot = vec![0.0; rotate * rotate];
        let mut rvar = vec![0.0; rotate];
        for (new, &old) in order.iter().enumerate() {
            let mut col = z[old * t..(old + 1) * t].to_vec();
            let on_kept: Vec<f64> = dense.kept_dates().iter().map(|&ti| col[ti]).collect();
            let u: Vec<f64> = on_kept.iter().map(|c| c / f).collect();
            let sign = if flip_needed(&u, &d.weighted_sum, d.total) { -1.0 } else { 1.0 };
            col.iter_mut().for_each(|c| *c *= sign);
            rpcs[new * t..(new + 1) * t].copy_from_slice(&col);
            for p in 0..nk {
                reof[new * n + kept[p]] = if sqrt_w[p] > 0.0 {
                    sign * l[old * nk + p] / sqrt_w[p]
                } else {
                    f64::NAN
                };
            }
            for i in 0..rotate {
                rrot[new * rotate + i] = sign * r[old * rotate + i];
            }
            rvar[new] = var[old];
        }
        Some(Rotated {
            m: rotate,
            eof: reof,
            pcs: rpcs,
            variance: rvar,
            rotation: rrot,
            iterations,
        })
    } else {
        None
    };

    let mut pixel_kept = vec![false; n];
    kept.iter().for_each(|&j| pixel_kept[j] = true);
    Ok(Out {
        modes: m,
        dates: t,
        n,
        explained: d.explained(),
        eigenvalues: eigen,
        singular: d.singular.clone(),
        pcs,
        eof: eof_map,
        corr: corr_out,
        north_error: nr.iter().map(|x| x.0).collect(),
        separable: nr.iter().map(|x| x.1).collect(),
        pc_filled,
        pixel_kept,
        date_kept,
        total_variance: d.total / (tk.max(2) - 1) as f64,
        solver: match d.solver_used {
            Solver::Randomized { .. } => "randomized",
            _ => "exact",
        },
        rotated,
    })
}

/// The temporal EOF of an (already anomalized) cube `values` (dates, pixels),
/// NaN missing, with the weights (pixels, mean 1) or None. Returns a dict:
/// `singular`, `explained`, `eigenvalues`, `north_error`, `separable` (modes),
/// `pcs` (modes, dates), `eof` (modes, pixels, NaN where the pixel was not
/// kept), `corr` (with `correlation_maps`), `pc_filled`, `date_kept`,
/// `pixel_kept`, `total_variance`, `solver`, and with `rotate` the `rotated_*`
/// ones (in the units scaling, unit-variance PCs).
#[pyfunction]
#[pyo3(signature = (values, weights = None, modes = 10, solver = "auto", oversampling = 10, power_iterations = 4,
    seed = 42, min_valid = 0.8, min_date_valid = 0.5, scaling = "units", rotate = 0, kaiser = true,
    north_n = "dates", correlation_maps = false, n_jobs = -1))]
#[allow(clippy::too_many_arguments)]
fn decompose<'py>(
    py: Python<'py>,
    values: PyArrayLike2<'py, f64, AllowTypeChange>,
    weights: Option<PyArrayLike1<'py, f64, AllowTypeChange>>,
    modes: usize,
    solver: &str,
    oversampling: usize,
    power_iterations: usize,
    seed: u64,
    min_valid: f64,
    min_date_valid: f64,
    scaling: &str,
    rotate: usize,
    kaiser: bool,
    north_n: &str,
    correlation_maps: bool,
    n_jobs: i32,
) -> PyResult<Bound<'py, PyDict>> {
    let solver = match solver {
        "auto" => Solver::Auto,
        "exact" => Solver::Exact,
        "randomized" => Solver::Randomized {
            oversampling,
            power_iterations,
        },
        other => return Err(PyValueError::new_err(format!("solver must be 'auto', 'exact' or 'randomized', got {other:?}"))),
    };
    let units = match scaling {
        "units" => true,
        "none" => false,
        other => return Err(PyValueError::new_err(format!("scaling must be 'units' or 'none', got {other:?}"))),
    };
    let autocorrelation = match north_n {
        "dates" => false,
        "autocorrelation" => true,
        other => {
            return Err(PyValueError::new_err(format!(
                "north_n must be 'dates' or 'autocorrelation', got {other:?}"
            )))
        }
    };
    if !(min_valid >= 0.0 && min_valid <= 1.0 && min_date_valid >= 0.0 && min_date_valid <= 1.0) {
        return Err(PyValueError::new_err("min_valid and min_date_valid must be in [0, 1]"));
    }
    let view = values.as_array();
    let (t, n) = view.dim();
    let v = c_order(&view);
    let v: &[f64] = &v;
    let w_view = weights.as_ref().map(|w| w.as_array());
    let w_cow = w_view.as_ref().map(c_order);
    let w: Option<&[f64]> = w_cow.as_deref();
    let s = Settings { modes, solver, seed };
    let out = py
        .detach(|| run(v, t, n, w, &s, min_valid, min_date_valid, units, rotate, kaiser, autocorrelation, correlation_maps, n_jobs))
        .map_err(PyValueError::new_err)?;
    let d = PyDict::new(py);
    let (m, nt, np) = (out.modes, out.dates, out.n);
    d.set_item("singular", PyArray1::from_vec(py, out.singular))?;
    d.set_item("explained", PyArray1::from_vec(py, out.explained))?;
    d.set_item("eigenvalues", PyArray1::from_vec(py, out.eigenvalues))?;
    d.set_item("north_error", PyArray1::from_vec(py, out.north_error))?;
    d.set_item("separable", PyArray1::from_vec(py, out.separable))?;
    d.set_item("pcs", PyArray1::from_vec(py, out.pcs).reshape([m, nt])?)?;
    d.set_item("eof", PyArray1::from_vec(py, out.eof).reshape([m, np])?)?;
    if let Some(c) = out.corr {
        d.set_item("corr", PyArray1::from_vec(py, c).reshape([m, np])?)?;
    }
    d.set_item("pc_filled", PyArray1::from_vec(py, out.pc_filled))?;
    d.set_item("date_kept", PyArray1::from_vec(py, out.date_kept))?;
    d.set_item("pixel_kept", PyArray1::from_vec(py, out.pixel_kept))?;
    d.set_item("total_variance", out.total_variance)?;
    d.set_item("solver", out.solver)?;
    if let Some(r) = out.rotated {
        d.set_item("rotated_eof", PyArray1::from_vec(py, r.eof).reshape([r.m, np])?)?;
        d.set_item("rotated_pcs", PyArray1::from_vec(py, r.pcs).reshape([r.m, nt])?)?;
        d.set_item("rotated_variance", PyArray1::from_vec(py, r.variance))?;
        d.set_item("rotation", PyArray1::from_vec(py, r.rotation).reshape([r.m, r.m])?)?;
        d.set_item("rotation_iterations", r.iterations)?;
    }
    Ok(d)
}

/// The varimax rotation of a loadings matrix `l` (n, m) -> (rotated (n, m), R (m, m)).
#[pyfunction]
#[pyo3(signature = (loadings, kaiser = true))]
fn varimax<'py>(
    py: Python<'py>,
    loadings: PyArrayLike2<'py, f64, AllowTypeChange>,
    kaiser: bool,
) -> PyResult<(Bound<'py, PyArray2<f64>>, Bound<'py, PyArray2<f64>>)> {
    let view = loadings.as_array();
    let (n, m) = view.dim();
    // the core's layout is column-major
    let mut l = vec![0.0; n * m];
    for ((j, k), v) in view.indexed_iter() {
        l[k * n + j] = *v;
    }
    let (r, _) = py.detach(|| eof::varimax(&mut l, n, m, kaiser, 1e-8, 1000));
    let mut rot = vec![0.0; n * m];
    let mut rr = vec![0.0; m * m];
    for k in 0..m {
        for j in 0..n {
            rot[j * m + k] = l[k * n + j];
        }
        for i in 0..m {
            rr[i * m + k] = r[k * m + i];
        }
    }
    Ok((
        PyArray1::from_vec(py, rot).reshape([n, m])?,
        PyArray1::from_vec(py, rr).reshape([m, m])?,
    ))
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "eof")?;
    m.add("__doc__", "Temporal EOF: exact and randomized SVD over pixel blocks, North, varimax")?;
    m.add_function(wrap_pyfunction!(decompose, &m)?)?;
    m.add_function(wrap_pyfunction!(varimax, &m)?)?;
    Ok(m)
}
