//! The naive per-pixel formulas of STARFM and ESTARFM, straight from the equations, in f64, sampling the coarse
//! images point by point. They are the oracle of the tests (the optimised code must agree to 1e-6) and are not meant
//! to be fast.

use super::*;

/// The coarse value seen by fine pixel `(x, y)` of the data extent.
pub fn sample(coarse: &Coarse, date: usize, band: usize, x: usize, y: usize, nx: usize, how: Resampling) -> f64 {
    let (u, v) = coarse.uv(x, y, nx);
    let plane = coarse.plane(date, band);
    match how {
        Resampling::Nearest => {
            let (i, j) = (u.floor(), v.floor());
            if i < 0.0 || j < 0.0 || i >= coarse.ncx as f64 || j >= coarse.ncy as f64 {
                f64::NAN
            } else {
                f64::from(plane[j as usize * coarse.ncx + i as usize])
            }
        }
        Resampling::Bilinear => f64::from(bilinear(plane, coarse.ncx as i64, coarse.ncy as i64, u, v)),
    }
}

fn cell_of(coarse: &Coarse, x: usize, y: usize, nx: usize) -> Option<usize> {
    let (u, v) = coarse.uv(x, y, nx);
    let (i, j) = (u.floor(), v.floor());
    if i < 0.0 || j < 0.0 || i >= coarse.ncx as f64 || j >= coarse.ncy as f64 {
        None
    } else {
        Some(j as usize * coarse.ncx + i as usize)
    }
}

struct Cand {
    s: f64,
    t: f64,
    d: f64,
    e: f64,
    pure: bool,
}

/// STARFM for one pixel and band: `(value, n_similar, source)`.
#[allow(clippy::too_many_arguments)]
pub fn starfm_pixel(
    fine: &Fine,
    coarse: &Coarse,
    target: &Target,
    band: usize,
    x: usize,
    y: usize,
    sigma: &[f64],
    p: &StarfmParams,
) -> (f64, u32, u16) {
    let (nx, ny, nb) = (fine.nx, fine.ny, fine.nb);
    let how = p.common.resampling;
    let fval = |k: usize, xx: usize, yy: usize| f64::from(fine.plane(k, band)[yy * nx + xx]);
    let mval = |cj: usize, xx: usize, yy: usize| sample(coarse, cj, band, xx, yy, nx, how);
    let ok = |k: usize, cj: usize, xx: usize, yy: usize| {
        fval(k, xx, yy).is_finite() && mval(cj, xx, yy).is_finite() && mval(target.coarse, xx, yy).is_finite()
    };
    let usable: Vec<(usize, usize)> = target
        .pairs
        .iter()
        .copied()
        .filter(|&(k, cj)| ok(k, cj, x, y))
        .collect();
    if usable.is_empty() {
        let m = mval(target.coarse, x, y);
        return if p.common.fallback == Fallback::Coarse && m.is_finite() {
            (m, 0, SRC_COARSE)
        } else {
            (f64::NAN, 0, SRC_MISSING)
        };
    }
    let smax = usable
        .iter()
        .map(|&(k, cj)| (fval(k, x, y) - mval(cj, x, y)).abs())
        .fold(0.0, f64::max);
    let tmax = usable
        .iter()
        .map(|&(_, cj)| (mval(cj, x, y) - mval(target.coarse, x, y)).abs())
        .fold(0.0, f64::max);
    let sfc = (p.fine_unc.powi(2) + p.coarse_unc.powi(2)).sqrt();
    let scc = 2f64.sqrt() * p.coarse_unc;
    let r = p.common.radius_px as i64;
    let mut cands: Vec<Cand> = Vec::new();
    for &(k, cj) in &usable {
        let theta = match p.common.sigma {
            SigmaScope::Image => 2.0 * sigma[k * nb + band] / p.classes,
            SigmaScope::Window => {
                let mut v = Vec::new();
                for yy in (y as i64 - r).max(0)..=(y as i64 + r).min(ny as i64 - 1) {
                    for xx in (x as i64 - r).max(0)..=(x as i64 + r).min(nx as i64 - 1) {
                        let f = fval(k, xx as usize, yy as usize);
                        if f.is_finite() {
                            v.push(f);
                        }
                    }
                }
                let mean = v.iter().sum::<f64>() / v.len() as f64;
                2.0 * (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt() / p.classes
            }
        };
        // Purity of the cells of this pair and band (spec 4.7, "cells").
        let pure_cell = |cell: usize| -> bool {
            let (mut lo, mut hi, mut valid, mut total) = (f64::INFINITY, f64::NEG_INFINITY, 0usize, 0usize);
            for yy in 0..ny {
                for xx in 0..nx {
                    if cell_of(coarse, xx, yy, nx) == Some(cell) {
                        total += 1;
                        let f = fval(k, xx, yy);
                        if f.is_finite() {
                            valid += 1;
                            lo = lo.min(f);
                            hi = hi.max(f);
                        }
                    }
                }
            }
            valid > 0 && 2 * valid >= total && hi - lo <= 2.0 * sigma[k * nb + band] / p.classes
        };
        for yy in (y as i64 - r).max(0)..=(y as i64 + r).min(ny as i64 - 1) {
            for xx in (x as i64 - r).max(0)..=(x as i64 + r).min(nx as i64 - 1) {
                let (xx, yy) = (xx as usize, yy as usize);
                if !ok(k, cj, xx, yy) {
                    continue;
                }
                let centre = xx == x && yy == y;
                if !centre && (fval(k, xx, yy) - fval(k, x, y)).abs() > theta {
                    continue;
                }
                let s = (fval(k, xx, yy) - mval(cj, xx, yy)).abs();
                let t = (mval(cj, xx, yy) - mval(target.coarse, xx, yy)).abs();
                if !centre {
                    let a = s < smax + sfc;
                    let b = t < tmax + scc;
                    let keep = match p.filter {
                        Filter::Both => a && b,
                        Filter::Either => a || b,
                        Filter::None => true,
                    };
                    if !keep {
                        continue;
                    }
                }
                let dist = ((xx as f64 - x as f64).powi(2) + (yy as f64 - y as f64).powi(2)).sqrt();
                let e = mval(target.coarse, xx, yy) + fval(k, xx, yy) - mval(cj, xx, yy);
                let pure = p.pure == Pure::Cells && cell_of(coarse, xx, yy, nx).is_some_and(pure_cell);
                cands.push(Cand {
                    s,
                    t,
                    d: 1.0 + dist / p.spatial_scale_px,
                    e,
                    pure,
                });
            }
        }
    }
    let direct = |c: &Cand| match p.weighting {
        Weighting::Direct => 1.0 / ((c.s + EPS) * (c.t + EPS) * c.d),
        Weighting::Log => 1.0 / (((c.s + EPS) * LOG_B + 1.0).ln() * ((c.t + EPS) * LOG_B + 1.0).ln() * c.d),
    };
    let one = |x: f64, d: f64| match p.weighting {
        Weighting::Direct => 1.0 / (x * d),
        Weighting::Log => 1.0 / ((x * LOG_B + 1.0).ln() * d),
    };
    let mut weighted: Vec<(f64, f64)> = Vec::new(); // (weight, estimate)
    match p.pure {
        Pure::Epsilon => weighted.extend(cands.iter().map(|c| (direct(c), c.e))),
        Pure::Cells => {
            let any = cands.iter().any(|c| c.pure);
            weighted.extend(cands.iter().filter(|c| !any || c.pure).map(|c| (direct(c), c.e)));
        }
        Pure::Tiers => {
            let tier = |c: &Cand| (c.s <= p.zero_tol) as u8 + (c.t <= p.zero_tol) as u8;
            let top = cands.iter().map(tier).max().unwrap();
            for c in cands.iter().filter(|c| tier(c) == top) {
                let w = match top {
                    2 => 1.0 / c.d,
                    1 => one(if c.s <= p.zero_tol { c.t } else { c.s }, c.d),
                    _ => direct(c),
                };
                weighted.push((w, c.e));
            }
        }
    }
    let sw: f64 = weighted.iter().map(|w| w.0).sum();
    let v = weighted.iter().map(|w| w.0 * w.1).sum::<f64>() / sw;
    (
        v,
        weighted.len() as u32,
        if usable.len() == target.pairs.len() {
            SRC_TWO_PAIRS
        } else {
            SRC_ONE_PAIR
        },
    )
}

/// ESTARFM for one pixel, all bands: `(values, n_similar, source)`. Two pairs.
pub fn estarfm_pixel(
    fine: &Fine,
    coarse: &Coarse,
    target: &Target,
    x: usize,
    y: usize,
    sigma: &[f64],
    p: &EstarfmParams,
) -> (Vec<f64>, u32, u16) {
    let (nx, ny, nb) = (fine.nx, fine.ny, fine.nb);
    let how = p.common.resampling;
    let fval = |k: usize, b: usize, xx: usize, yy: usize| f64::from(fine.plane(k, b)[yy * nx + xx]);
    let mval = |cj: usize, b: usize, xx: usize, yy: usize| sample(coarse, cj, b, xx, yy, nx, how);
    let valid = |pair: (usize, usize), xx: usize, yy: usize| {
        (0..nb).all(|b| fval(pair.0, b, xx, yy).is_finite() && mval(pair.1, b, xx, yy).is_finite())
    };
    let pairs: Vec<(usize, usize)> = target.pairs.iter().copied().filter(|&pr| valid(pr, x, y)).collect();
    let usepairs: Vec<(usize, usize)> =
        if pairs.len() == 2 || (pairs.len() == 1 && p.single_pair == SinglePair::Fallback) {
            pairs.clone()
        } else {
            vec![]
        };
    if usepairs.is_empty() {
        let vals: Vec<f64> = (0..nb)
            .map(|b| {
                let m = mval(target.coarse, b, x, y);
                if p.common.fallback == Fallback::Coarse {
                    m
                } else {
                    f64::NAN
                }
            })
            .collect();
        let any = vals.iter().any(|v| v.is_finite());
        return (vals, 0, if any { SRC_COARSE } else { SRC_MISSING });
    }
    let two = usepairs.len() == 2;
    let r = p.common.radius_px as i64;
    let rx = |c: usize, rad: i64, n: usize| (c as i64 - rad).max(0)..=(c as i64 + rad).min(n as i64 - 1);
    let theta: Vec<Vec<f64>> = usepairs
        .iter()
        .map(|&(k, _)| {
            (0..nb)
                .map(|b| {
                    let s = match p.common.sigma {
                        SigmaScope::Image => sigma[k * nb + b],
                        SigmaScope::Window => {
                            let mut v = Vec::new();
                            for yy in rx(y, r, ny) {
                                for xx in rx(x, r, nx) {
                                    let f = fval(k, b, xx as usize, yy as usize);
                                    if f.is_finite() {
                                        v.push(f);
                                    }
                                }
                            }
                            let mean = v.iter().sum::<f64>() / v.len() as f64;
                            (v.iter().map(|a| (a - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt()
                        }
                    };
                    2.0 * s / p.classes
                })
                .collect()
        })
        .collect();
    let similar = |xx: usize, yy: usize| -> bool {
        if xx == x && yy == y {
            return true;
        }
        usepairs.iter().enumerate().all(|(ki, &pr)| {
            valid(pr, xx, yy) && (0..nb).all(|b| (fval(pr.0, b, xx, yy) - fval(pr.0, b, x, y)).abs() <= theta[ki][b])
        })
    };
    let ext = if two && p.v_scope == VScope::Cell {
        cell_size_px_ref(coarse, nx, ny).ceil() as i64 + 1
    } else {
        0
    };
    let mut list: Vec<(usize, usize)> = Vec::new();
    for yy in rx(y, r, ny) {
        for xx in rx(x, r, nx) {
            if similar(xx as usize, yy as usize) {
                list.push((xx as usize, yy as usize));
            }
        }
    }
    // Weights.
    let corr: Vec<f64> = list
        .iter()
        .map(|&(xx, yy)| {
            let mut a = Vec::new();
            let mut c = Vec::new();
            for &pr in &usepairs {
                for b in 0..nb {
                    a.push(fval(pr.0, b, xx, yy));
                    c.push(mval(pr.1, b, xx, yy));
                }
            }
            let n = a.len() as f64;
            let (ma, mc) = (a.iter().sum::<f64>() / n, c.iter().sum::<f64>() / n);
            let cov: f64 = a.iter().zip(&c).map(|(u, v)| (u - ma) * (v - mc)).sum();
            let va: f64 = a.iter().map(|u| (u - ma).powi(2)).sum();
            let vc: f64 = c.iter().map(|v| (v - mc).powi(2)).sum();
            if va <= 0.0 || vc <= 0.0 {
                0.0
            } else {
                (cov / (va * vc).sqrt()).clamp(-1.0, 1.0)
            }
        })
        .collect();
    let dist: Vec<f64> = list
        .iter()
        .map(|&(xx, yy)| 1.0 + (((xx as f64 - x as f64).powi(2) + (yy as f64 - y as f64).powi(2)).sqrt()) / r as f64)
        .collect();
    let n = list.len();
    let base: Vec<f64> = if n == 1 {
        vec![1.0]
    } else {
        let nperf = corr.iter().filter(|&&c| c >= 1.0 - 1e-6).count();
        if nperf >= 1 {
            corr.iter()
                .map(|&c| if c >= 1.0 - 1e-6 { 1.0 / nperf as f64 } else { 0.0 })
                .collect()
        } else {
            let inv: Vec<f64> = (0..n).map(|i| 1.0 / ((1.0 - corr[i]) * dist[i])).collect();
            let s: f64 = inv.iter().sum();
            inv.iter().map(|v| v / s).collect()
        }
    };
    // Conversion coefficient of band b for similar pixel i.
    let mut forced = false;
    let regress = |pts: &[(f64, f64)]| -> (f64, bool) {
        let n = pts.len() as f64;
        // The mean coarse change between the pairs over the group's pixels (points alternate pair 0, pair 1).
        let (mut m0, mut m1) = (0.0, 0.0);
        for (i, q) in pts.iter().enumerate() {
            if i % 2 == 0 { m0 += q.0 } else { m1 += q.0 }
        }
        if n > 0.0 && (m0 - m1).abs() / (n / 2.0) <= p.min_change {
            return (1.0, true);
        }
        if n < 3.0 {
            return (1.0, false);
        }
        let xm = pts.iter().map(|q| q.0).sum::<f64>() / n;
        let ym = pts.iter().map(|q| q.1).sum::<f64>() / n;
        let sxx: f64 = pts.iter().map(|q| (q.0 - xm).powi(2)).sum();
        let sxy: f64 = pts.iter().map(|q| (q.0 - xm) * (q.1 - ym)).sum();
        let syy: f64 = pts.iter().map(|q| (q.1 - ym).powi(2)).sum();
        if !(sxx > 1e-12 * pts.iter().map(|q| q.0 * q.0).sum::<f64>().max(1e-30)) {
            return (1.0, false);
        }
        let v = sxy / sxx;
        let s2 = ((syy - v * sxy) / (n - 2.0)).max(0.0);
        let se = (s2 / sxx).sqrt();
        let ok = se == 0.0 || crate::dist::t_two_sided(v / se, n - 2.0) <= p.alpha;
        if ok && (v > 0.0 || p.allow_negative) { (v, true) } else { (1.0, false) }
    };
    let points = |b: usize, xx: usize, yy: usize| -> Vec<(f64, f64)> {
        usepairs
            .iter()
            .map(|&pr| (mval(pr.1, b, xx, yy), fval(pr.0, b, xx, yy)))
            .collect()
    };
    let mut vwin = vec![1.0; nb];
    if two && p.v_scope == VScope::Window {
        for b in 0..nb {
            let mut pts = Vec::new();
            for &(xx, yy) in &list {
                pts.extend(points(b, xx, yy));
            }
            let (v, built) = regress(&pts);
            vwin[b] = v;
            forced |= !built;
        }
    }
    let vfor = |b: usize, i: usize, forced: &mut bool| -> f64 {
        if !two {
            1.0
        } else if p.v_scope == VScope::Window {
            vwin[b]
        } else {
            let (xi, yi) = list[i];
            let ci = cell_of(coarse, xi, yi, nx);
            let mut pts = Vec::new();
            for yy in rx(y, r + ext, ny) {
                for xx in rx(x, r + ext, nx) {
                    let (xx, yy) = (xx as usize, yy as usize);
                    if cell_of(coarse, xx, yy, nx) == ci && similar(xx, yy) {
                        pts.extend(points(b, xx, yy));
                    }
                }
            }
            let (v, built) = regress(&pts);
            *forced |= !built;
            v
        }
    };
    if two && p.v_scope == VScope::Cell {
        for b in 0..nb {
            for i in 0..n {
                vfor(b, i, &mut forced);
            }
        }
    }
    let mut vals = vec![f64::NAN; nb];
    for b in 0..nb {
        if !mval(target.coarse, b, x, y).is_finite() {
            continue;
        }
        let keep: Vec<usize> = (0..n)
            .filter(|&i| mval(target.coarse, b, list[i].0, list[i].1).is_finite())
            .collect();
        let sw: f64 = keep.iter().map(|&i| base[i]).sum();
        let w: Vec<f64> = if sw > 0.0 {
            keep.iter().map(|&i| base[i] / sw).collect()
        } else {
            let s: f64 = keep.iter().map(|&i| 1.0 / dist[i]).sum();
            keep.iter().map(|&i| 1.0 / dist[i] / s).collect()
        };
        let mut pred = Vec::new();
        for &(k, cj) in &usepairs {
            let mut acc = 0.0;
            for (j, &i) in keep.iter().enumerate() {
                let (xx, yy) = list[i];
                let vi = vfor(b, i, &mut forced);
                acc += w[j] * vi * (mval(target.coarse, b, xx, yy) - mval(cj, b, xx, yy));
            }
            pred.push(fval(k, b, x, y) + acc);
        }
        vals[b] = if two {
            let mut delta = [0.0; 2];
            for (ki, &(_, cj)) in usepairs.iter().enumerate() {
                let (mut sk, mut sp) = (0.0, 0.0);
                for yy in rx(y, r, ny) {
                    for xx in rx(x, r, nx) {
                        let (a, c) = (
                            mval(cj, b, xx as usize, yy as usize),
                            mval(target.coarse, b, xx as usize, yy as usize),
                        );
                        if a.is_finite() && c.is_finite() {
                            sk += a;
                            sp += c;
                        }
                    }
                }
                delta[ki] = (sk - sp).abs();
            }
            let (tm, tn) = estarfm::temporal_weights(delta[0], delta[1]);
            tm * pred[0] + tn * pred[1]
        } else {
            pred[0]
        };
    }
    let src = if !two {
        SRC_ONE_PAIR
    } else if forced {
        SRC_V_ONE
    } else {
        SRC_TWO_PAIRS
    };
    let src = if vals.iter().all(|v| v.is_nan()) {
        SRC_MISSING
    } else {
        src
    };
    (vals, n as u32, src)
}

fn cell_size_px_ref(coarse: &Coarse, nx: usize, ny: usize) -> f64 {
    estarfm::cell_size_px(coarse, nx, ny)
}
