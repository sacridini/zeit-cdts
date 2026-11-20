//! STARFM (Gao, Masek, Schwaller and Hall 2006, IEEE TGRS 44(8):2207-2218), written from the paper and its open
//! restatements: every band on its own, a weighted mean over the spectrally similar neighbours of a window of the
//! fine value of a pair plus the coarse change since the pair's date. See `docs/api/preprocessing.md` for the
//! equations and the options.

use super::*;
use std::collections::HashMap;

/// Result of one pixel and band.
pub(crate) struct Px {
    pub value: f64,
    pub n: u32,
    pub src: u16,
}

/// What `pixel` reads for one band.
pub(crate) struct Bands<'a> {
    pub nx: usize,
    pub ny: usize,
    /// Fine image of each pair of the target.
    pub f: Vec<&'a [f32]>,
    /// Resampled coarse image of each pair's date.
    pub m: Vec<&'a [f32]>,
    /// Resampled coarse image of the predicted date.
    pub mp: &'a [f32],
    /// `2 sigma / classes` of each pair (sigma of the image).
    pub theta: Vec<f64>,
    /// Pure coarse cells of each pair (`pure = "cells"`), and the cell of each fine pixel.
    pub pure: Option<Vec<&'a [bool]>>,
    pub cell: &'a [i32],
}

#[derive(Clone, Copy, Default)]
struct Acc {
    w: f64,
    we: f64,
    n: u32,
}

impl Acc {
    #[inline]
    fn add(&mut self, w: f64, e: f64) {
        self.w += w;
        self.we += w * e;
        self.n += 1;
    }
}

/// The weight `1 / C` of a candidate with spectral distance `s`, temporal distance `t` and spatial term `d`.
#[inline]
fn base_weight(weighting: Weighting, s: f64, t: f64, d: f64) -> f64 {
    match weighting {
        Weighting::Direct => 1.0 / ((s + EPS) * (t + EPS) * d),
        Weighting::Log => 1.0 / (((s + EPS) * LOG_B + 1.0).ln() * ((t + EPS) * LOG_B + 1.0).ln() * d),
    }
}

#[inline]
fn one_dist(weighting: Weighting, x: f64, d: f64) -> f64 {
    match weighting {
        Weighting::Direct => 1.0 / (x * d),
        Weighting::Log => 1.0 / ((x * LOG_B + 1.0).ln() * d),
    }
}

/// STARFM at pixel `(x, y)` of one band.
pub(crate) fn pixel(p: &StarfmParams, bd: &Bands, x: usize, y: usize) -> Px {
    let (nx, ny) = (bd.nx, bd.ny);
    let c = y * nx + x;
    let mpc = bd.mp[c];
    let nk = bd.f.len();
    // Pairs usable at the centre.
    let mut usable: Vec<usize> = Vec::with_capacity(nk);
    let (mut smax, mut tmax) = (0.0f64, 0.0f64);
    for k in 0..nk {
        let (f, m) = (bd.f[k][c], bd.m[k][c]);
        if f.is_finite() && m.is_finite() && mpc.is_finite() {
            usable.push(k);
            smax = smax.max(f64::from((f - m).abs()));
            tmax = tmax.max(f64::from((m - mpc).abs()));
        }
    }
    if usable.is_empty() {
        return if p.common.fallback == Fallback::Coarse && mpc.is_finite() {
            Px {
                value: f64::from(mpc),
                n: 0,
                src: SRC_COARSE,
            }
        } else {
            Px {
                value: f64::NAN,
                n: 0,
                src: SRC_MISSING,
            }
        };
    }
    let sfc = (p.fine_unc * p.fine_unc + p.coarse_unc * p.coarse_unc).sqrt();
    let scc = std::f64::consts::SQRT_2 * p.coarse_unc;
    let r = p.common.radius_px;
    let (x0, x1, y0, y1) = window(x, y, r, nx, ny);

    let mut all = Acc::default();
    let mut tiers = [Acc::default(); 3];
    let mut pure_acc = Acc::default();
    for &k in &usable {
        let (fk, mk) = (bd.f[k], bd.m[k]);
        let fc = f64::from(fk[c]);
        let theta = match p.common.sigma {
            SigmaScope::Image => bd.theta[k],
            SigmaScope::Window => {
                let (mut n, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0f64);
                for yy in y0..y1 {
                    for &v in &fk[yy * nx + x0..yy * nx + x1] {
                        if v.is_finite() {
                            let v = f64::from(v);
                            n += 1.0;
                            s1 += v;
                            s2 += v * v;
                        }
                    }
                }
                let var = (s2 / n - (s1 / n) * (s1 / n)).max(0.0);
                2.0 * var.sqrt() / p.classes
            }
        };
        let pure_k = bd.pure.as_ref().map(|v| v[k]);
        for yy in y0..y1 {
            for xx in x0..x1 {
                let q = yy * nx + xx;
                let (f, m, mq) = (fk[q], mk[q], bd.mp[q]);
                if !(f.is_finite() && m.is_finite() && mq.is_finite()) {
                    continue;
                }
                let centre = q == c;
                if !centre && !((f64::from(f) - fc).abs() <= theta) {
                    continue;
                }
                let s = f64::from((f - m).abs());
                let t = f64::from((m - mq).abs());
                if !centre {
                    let keep = match p.filter {
                        Filter::Both => s < smax + sfc && t < tmax + scc,
                        Filter::Either => s < smax + sfc || t < tmax + scc,
                        Filter::None => true,
                    };
                    if !keep {
                        continue;
                    }
                }
                let (dx, dy) = (xx as f64 - x as f64, yy as f64 - y as f64);
                let d = 1.0 + (dx * dx + dy * dy).sqrt() / p.spatial_scale_px;
                let e = f64::from(mq) + f64::from(f) - f64::from(m);
                match p.pure {
                    Pure::Epsilon => all.add(base_weight(p.weighting, s, t, d), e),
                    Pure::Tiers => {
                        let (zs, zt) = (s <= p.zero_tol, t <= p.zero_tol);
                        if zs && zt {
                            tiers[2].add(1.0 / d, e);
                        } else if zs || zt {
                            tiers[1].add(one_dist(p.weighting, if zs { t } else { s }, d), e);
                        } else {
                            tiers[0].add(base_weight(p.weighting, s, t, d), e);
                        }
                    }
                    Pure::Cells => {
                        let w = base_weight(p.weighting, s, t, d);
                        all.add(w, e);
                        let cq = bd.cell[q];
                        if cq >= 0 && pure_k.is_some_and(|pk| pk[cq as usize]) {
                            pure_acc.add(w, e);
                        }
                    }
                }
            }
        }
    }
    let sel = match p.pure {
        Pure::Epsilon => all,
        Pure::Tiers => {
            if tiers[2].n > 0 {
                tiers[2]
            } else if tiers[1].n > 0 {
                tiers[1]
            } else {
                tiers[0]
            }
        }
        Pure::Cells => {
            if pure_acc.n > 0 {
                pure_acc
            } else {
                all
            }
        }
    };
    Px {
        value: sel.we / sel.w,
        n: sel.n,
        src: if usable.len() == nk {
            SRC_TWO_PAIRS
        } else {
            SRC_ONE_PAIR
        },
    }
}

/// For each cell, whether the valid fine pixels of the image `plane` in it span at most `theta` (and at least half
/// of the cell's fine pixels of the extent are valid).
pub(crate) fn pure_cells(plane: &[f32], cell: &[i32], ncell: usize, theta: f64) -> Vec<bool> {
    let mut lo = vec![f32::INFINITY; ncell];
    let mut hi = vec![f32::NEG_INFINITY; ncell];
    let mut valid = vec![0u32; ncell];
    let mut total = vec![0u32; ncell];
    for (i, &c) in cell.iter().enumerate() {
        if c < 0 {
            continue;
        }
        let c = c as usize;
        total[c] += 1;
        let v = plane[i];
        if v.is_finite() {
            valid[c] += 1;
            lo[c] = lo[c].min(v);
            hi[c] = hi[c].max(v);
        }
    }
    (0..ncell)
        .map(|c| valid[c] > 0 && 2 * valid[c] >= total[c] && f64::from(hi[c]) - f64::from(lo[c]) <= theta)
        .collect()
}

/// STARFM over `region`. `sigma` is `image_sigma`'s `(pair, band)` array. `out` is `(target, band, region pixel)`,
/// `quality` (if given) `(target, 2, region pixel)` with `n_similar` (minimum over the bands) and `source`.
#[allow(clippy::too_many_arguments)]
pub fn starfm(
    fine: &Fine,
    coarse: &Coarse,
    plan: &Plan,
    sigma: &[f64],
    p: &StarfmParams,
    region: &Region,
    out: &mut [f32],
    mut quality: Option<&mut [u16]>,
    n_jobs: i32,
) -> Result<(), String> {
    check_inputs(fine, coarse, plan, region)?;
    let (nb, nx, ny) = (fine.nb, fine.nx, fine.ny);
    let nt = plan.targets.len();
    let rn = region.nx * region.ny;
    if out.len() != nt * nb * rn {
        return Err(format!("out must hold {} values, has {}", nt * nb * rn, out.len()));
    }
    if quality.as_ref().is_some_and(|q| q.len() != nt * 2 * rn) {
        return Err("quality must be (target, 2, y, x)".into());
    }
    if sigma.len() != fine.npairs() * nb {
        return Err("sigma must be (pair, band)".into());
    }
    if p.classes <= 0.0 || p.spatial_scale_px <= 0.0 {
        return Err("classes and spatial_scale must be positive".into());
    }
    let cells = if p.pure == Pure::Cells {
        Some(cell_map(coarse, nx, ny))
    } else {
        None
    };
    let empty: Vec<i32> = Vec::new();
    let cell: &[i32] = cells.as_deref().unwrap_or(&empty);
    // Pure cells of each (pair, band), computed once.
    let mut pure: HashMap<(usize, usize), Vec<bool>> = HashMap::new();
    if p.pure == Pure::Cells {
        for k in plan.fine_used() {
            for b in 0..nb {
                let th = 2.0 * sigma[k * nb + b] / p.classes;
                pure.insert((k, b), pure_cells(fine.plane(k, b), cell, coarse.ncell(), th));
            }
        }
    }
    let mut cache: HashMap<usize, Vec<f32>> = HashMap::new();
    let npix = nx * ny;
    for (ti, t) in plan.targets.iter().enumerate() {
        for &(_, cj) in &t.pairs {
            cache
                .entry(cj)
                .or_insert_with(|| resample_date(coarse, cj, nx, ny, p.common.resampling));
        }
        let own;
        let mp_all: &[f32] = match cache.get(&t.coarse) {
            Some(v) => v,
            None => {
                own = resample_date(coarse, t.coarse, nx, ny, p.common.resampling);
                &own
            }
        };
        let obs = t.observed;
        let res = rows(region.ny, n_jobs, |ry| {
            let y = region.y0 + ry;
            let mut vals = vec![f32::NAN; nb * region.nx];
            let mut q = vec![0u16; 2 * region.nx];
            let mut bands: Vec<Bands> = Vec::with_capacity(nb);
            for b in 0..nb {
                bands.push(Bands {
                    nx,
                    ny,
                    f: t.pairs.iter().map(|&(k, _)| fine.plane(k, b)).collect(),
                    m: t.pairs
                        .iter()
                        .map(|&(_, cj)| &cache[&cj][b * npix..(b + 1) * npix])
                        .collect(),
                    mp: &mp_all[b * npix..(b + 1) * npix],
                    theta: t
                        .pairs
                        .iter()
                        .map(|&(k, _)| 2.0 * sigma[k * nb + b] / p.classes)
                        .collect(),
                    pure: if p.pure == Pure::Cells {
                        Some(t.pairs.iter().map(|&(k, _)| pure[&(k, b)].as_slice()).collect())
                    } else {
                        None
                    },
                    cell,
                });
            }
            for rx in 0..region.nx {
                let x = region.x0 + rx;
                let c = y * nx + x;
                if let Some(fi) = obs {
                    if (0..nb).all(|b| fine.plane(fi, b)[c].is_finite()) {
                        for b in 0..nb {
                            vals[b * region.nx + rx] = fine.plane(fi, b)[c];
                        }
                        q[rx] = 0;
                        q[region.nx + rx] = SRC_OBSERVED;
                        continue;
                    }
                }
                let (mut nmin, mut src, mut any) = (u32::MAX, 0u16, false);
                for b in 0..nb {
                    let px = pixel(p, &bands[b], x, y);
                    vals[b * region.nx + rx] =
                        if px.src == SRC_COARSE { px.value as f32 } else { clip(px.value, p.common.clip) as f32 };
                    nmin = nmin.min(px.n);
                    if px.src != SRC_MISSING {
                        any = true;
                        src = src.max(px.src);
                    }
                }
                q[rx] = nmin.min(u32::from(u16::MAX)) as u16;
                q[region.nx + rx] = if any { src } else { SRC_MISSING };
            }
            (vals, q)
        });
        for (ry, (vals, q)) in res.into_iter().enumerate() {
            for b in 0..nb {
                let o = ((ti * nb + b) * region.ny + ry) * region.nx;
                out[o..o + region.nx].copy_from_slice(&vals[b * region.nx..(b + 1) * region.nx]);
            }
            if let Some(qa) = quality.as_deref_mut() {
                for j in 0..2 {
                    let o = ((ti * 2 + j) * region.ny + ry) * region.nx;
                    qa[o..o + region.nx].copy_from_slice(&q[j * region.nx..(j + 1) * region.nx]);
                }
            }
        }
        // Keep only the pair dates in the cache.
        let keep: Vec<usize> = plan.targets.iter().flat_map(|t| t.pairs.iter().map(|p| p.1)).collect();
        cache.retain(|k, _| keep.contains(k));
    }
    Ok(())
}
