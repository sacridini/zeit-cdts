//! ESTARFM (Zhu, Chen, Gao, Chen and Masek 2010, Remote Sensing of Environment 114(11):2610-2623), written from the
//! paper: two pairs, all bands together; neighbours similar in every band on both pair dates; weights from the
//! correlation of the fine and coarse spectra and the distance (eqs. 14-17); the coarse change scaled by a
//! conversion coefficient (a regression of fine on coarse, sec. 3.4); the two predictions combined by how much the
//! coarse window changed since each pair (eqs. 18-19).

use super::*;
use std::collections::HashMap;

/// What `pixel` reads. `f[pair][band]`, `m[pair][band]`, `mp[band]` are planes of the data extent.
pub(crate) struct Data<'a> {
    pub nx: usize,
    pub ny: usize,
    pub nb: usize,
    pub f: Vec<Vec<&'a [f32]>>,
    pub m: Vec<Vec<&'a [f32]>>,
    pub mp: Vec<&'a [f32]>,
    /// `sigma[pair][band]` of the image.
    pub sigma: Vec<Vec<f64>>,
    pub cell: &'a [i32],
    pub ncell: usize,
    /// Pixels the per-cell regression reaches beyond the window.
    pub ext: usize,
}

pub(crate) struct Est {
    pub vals: Vec<f64>,
    pub n: u32,
    pub src: u16,
}

struct Group {
    cell: i32,
    /// Per band: n, sx, sy, sxx, sxy, syy (shifted by the centre's first-pair values).
    sums: Vec<f64>,
}

/// Reused between pixels.
pub(crate) struct Scratch {
    slot: Vec<i32>,
    groups: Vec<Group>,
    list: Vec<(usize, f64)>,
}

impl Scratch {
    pub fn new(ncell: usize) -> Self {
        Scratch {
            slot: vec![-1; ncell + 1],
            groups: Vec::new(),
            list: Vec::new(),
        }
    }
}

/// The slope of the regression of `y` on `x` from centred sums, set to 1 when it cannot be built or is not
/// significant at `alpha`. Returns `(slope, built)`.
pub fn slope(n: f64, sx: f64, sy: f64, sxx: f64, sxy: f64, syy: f64, alpha: f64) -> (f64, bool) {
    if n < 3.0 {
        return (1.0, false);
    }
    let cxx = sxx - sx * sx / n;
    let cxy = sxy - sx * sy / n;
    let cyy = syy - sy * sy / n;
    if !(cxx > 1e-12 * sxx.max(1e-30)) {
        return (1.0, false);
    }
    let v = cxy / cxx;
    let s2 = ((cyy - v * cxy) / (n - 2.0)).max(0.0);
    let se = (s2 / cxx).sqrt();
    let significant = if se == 0.0 {
        true
    } else {
        crate::dist::t_two_sided(v / se, n - 2.0) <= alpha
    };
    if significant {
        (v, true)
    } else {
        (1.0, false)
    }
}

/// The conversion coefficient `V` of a group from its sums (`n`, `sx`, `sy`, `sxx`, `sxy`, `syy`, then the sums of `x`
/// of each pair, `x` shifted as the others): the regression slope when the coarse reflectance changed between the two
/// pairs by more than `min_change` on average over the group, the slope is positive (unless `allow_negative`) and it is
/// significant; otherwise 1. The second value is false only when a regression was due and could not be built.
pub fn coefficient(s: &[f64], p: &EstarfmParams) -> (f64, bool) {
    let m = s[0] / 2.0; // pixels of the group
    if m > 0.0 && (s[6] - s[7]).abs() / m <= p.min_change {
        return (1.0, true);
    }
    let (v, built) = slope(s[0], s[1], s[2], s[3], s[4], s[5], p.alpha);
    if built && v <= 0.0 && !p.allow_negative {
        return (1.0, false);
    }
    (v, built)
}

/// Pearson correlation of two vectors, 0 when one has no variance, clamped to [-1, 1].
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for (x, y) in a.iter().zip(b) {
        sab += (x - ma) * (y - mb);
        saa += (x - ma) * (x - ma);
        sbb += (y - mb) * (y - mb);
    }
    if saa <= 0.0 || sbb <= 0.0 {
        return 0.0;
    }
    (sab / (saa * sbb).sqrt()).clamp(-1.0, 1.0)
}

/// Temporal weights of eq. 18 from the two window changes.
pub fn temporal_weights(dm: f64, dn: f64) -> (f64, f64) {
    match (dm == 0.0, dn == 0.0) {
        (true, true) => (0.5, 0.5),
        (true, false) => (1.0, 0.0),
        (false, true) => (0.0, 1.0),
        _ => {
            let (a, b) = (1.0 / dm, 1.0 / dn);
            (a / (a + b), b / (a + b))
        }
    }
}

/// Weights of eq. 17 with the special case of sec. 3.3.
pub fn similar_weights(r: &[f64], d: &[f64]) -> Vec<f64> {
    let perfect: Vec<bool> = r.iter().map(|&x| x >= 1.0 - 1e-6).collect();
    let p = perfect.iter().filter(|&&b| b).count();
    if p >= 1 {
        return perfect.iter().map(|&b| if b { 1.0 / p as f64 } else { 0.0 }).collect();
    }
    let inv: Vec<f64> = r.iter().zip(d).map(|(&ri, &di)| 1.0 / ((1.0 - ri) * di)).collect();
    let s: f64 = inv.iter().sum();
    inv.iter().map(|w| w / s).collect()
}

fn window_sigma(plane: &[f32], nx: usize, w: (usize, usize, usize, usize)) -> f64 {
    let (mut n, mut s1, mut s2) = (0.0f64, 0.0f64, 0.0f64);
    for yy in w.2..w.3 {
        for &v in &plane[yy * nx + w.0..yy * nx + w.1] {
            if v.is_finite() {
                let v = f64::from(v);
                n += 1.0;
                s1 += v;
                s2 += v * v;
            }
        }
    }
    ((s2 / n - (s1 / n) * (s1 / n)).max(0.0)).sqrt()
}

/// ESTARFM at pixel `(x, y)`, all bands.
pub(crate) fn pixel(p: &EstarfmParams, d: &Data, sc: &mut Scratch, x: usize, y: usize) -> Est {
    let (nx, ny, nb) = (d.nx, d.ny, d.nb);
    let c = y * nx + x;
    let valid_at = |k: usize, q: usize| (0..nb).all(|b| d.f[k][b][q].is_finite() && d.m[k][b][q].is_finite());
    let (v0, v1) = (valid_at(0, c), valid_at(1, c));
    let ks: Vec<usize> = if v0 && v1 {
        vec![0, 1]
    } else if (v0 || v1) && p.single_pair == SinglePair::Fallback {
        vec![if v0 { 0 } else { 1 }]
    } else {
        vec![]
    };
    if ks.is_empty() {
        let mut any = false;
        let vals = (0..nb)
            .map(|b| {
                let m = d.mp[b][c];
                if p.common.fallback == Fallback::Coarse && m.is_finite() {
                    any = true;
                    f64::from(m)
                } else {
                    f64::NAN
                }
            })
            .collect();
        return Est {
            vals,
            n: 0,
            src: if any { SRC_COARSE } else { SRC_MISSING },
        };
    }
    let two = ks.len() == 2;
    let r = p.common.radius_px;
    let win = window(x, y, r, nx, ny);
    let ext = if two && p.v_scope == VScope::Cell { d.ext } else { 0 };
    let big = window(x, y, r + ext, nx, ny);
    // Thresholds of each used pair and band.
    let theta: Vec<Vec<f64>> = ks
        .iter()
        .map(|&k| {
            (0..nb)
                .map(|b| {
                    let s = match p.common.sigma {
                        SigmaScope::Image => d.sigma[k][b],
                        SigmaScope::Window => window_sigma(d.f[k][b], nx, win),
                    };
                    2.0 * s / p.classes
                })
                .collect()
        })
        .collect();
    let x0s: Vec<f64> = (0..nb).map(|b| f64::from(d.m[ks[0]][b][c])).collect();
    let y0s: Vec<f64> = (0..nb).map(|b| f64::from(d.f[ks[0]][b][c])).collect();

    sc.list.clear();
    for g in sc.groups.drain(..) {
        sc.slot[if g.cell < 0 { d.ncell } else { g.cell as usize }] = -1;
    }
    let cell_scope = two && p.v_scope == VScope::Cell;
    for yy in big.2..big.3 {
        for xx in big.0..big.1 {
            let q = yy * nx + xx;
            if q != c {
                if !ks.iter().all(|&k| valid_at(k, q)) {
                    continue;
                }
                let similar = ks.iter().enumerate().all(|(ki, &k)| {
                    (0..nb).all(|b| (f64::from(d.f[k][b][q]) - f64::from(d.f[k][b][c])).abs() <= theta[ki][b])
                });
                if !similar {
                    continue;
                }
            }
            if cell_scope {
                let key = if d.cell[q] < 0 { d.ncell } else { d.cell[q] as usize };
                if sc.slot[key] < 0 {
                    sc.slot[key] = sc.groups.len() as i32;
                    sc.groups.push(Group {
                        cell: d.cell[q],
                        sums: vec![0.0; nb * 8],
                    });
                }
                let g = &mut sc.groups[sc.slot[key] as usize];
                for b in 0..nb {
                    for k in 0..2 {
                        let (xv, yv) = (f64::from(d.m[k][b][q]) - x0s[b], f64::from(d.f[k][b][q]) - y0s[b]);
                        let s = &mut g.sums[b * 8..b * 8 + 8];
                        s[0] += 1.0;
                        s[1] += xv;
                        s[2] += yv;
                        s[3] += xv * xv;
                        s[4] += xv * yv;
                        s[5] += yv * yv;
                        s[6 + k] += xv;
                    }
                }
            }
            if xx >= win.0 && xx < win.1 && yy >= win.2 && yy < win.3 {
                let (dx, dy) = (xx as f64 - x as f64, yy as f64 - y as f64);
                sc.list.push((q, (dx * dx + dy * dy).sqrt()));
            }
        }
    }
    // Weights of the similar pixels (eqs. 14-17).
    let np = sc.list.len();
    let (mut rr, mut dd) = (Vec::with_capacity(np), Vec::with_capacity(np));
    let mut fv = vec![0.0f64; ks.len() * nb];
    let mut cv = vec![0.0f64; ks.len() * nb];
    for &(q, dist) in &sc.list {
        for (ki, &k) in ks.iter().enumerate() {
            for b in 0..nb {
                fv[ki * nb + b] = f64::from(d.f[k][b][q]);
                cv[ki * nb + b] = f64::from(d.m[k][b][q]);
            }
        }
        rr.push(pearson(&fv, &cv));
        dd.push(1.0 + dist / (r as f64));
    }
    let w0 = if np == 1 { vec![1.0] } else { similar_weights(&rr, &dd) };
    let inv_d: Vec<f64> = dd.iter().map(|v| 1.0 / v).collect();

    // Conversion coefficients per band: (V of the window) or the group's.
    let mut forced = false;
    let mut vwin = vec![1.0f64; nb];
    let mut vgroup: Vec<Vec<f64>> = Vec::new();
    if two {
        if cell_scope {
            for g in &sc.groups {
                let mut vs = Vec::with_capacity(nb);
                for b in 0..nb {
                    let s = &g.sums[b * 8..b * 8 + 8];
                    vs.push(coefficient(s, p).0);
                }
                vgroup.push(vs);
            }
        } else {
            for b in 0..nb {
                let mut s = [0.0f64; 8];
                for &(q, _) in &sc.list {
                    for k in 0..2 {
                        let (xv, yv) = (f64::from(d.m[k][b][q]) - x0s[b], f64::from(d.f[k][b][q]) - y0s[b]);
                        s[0] += 1.0;
                        s[1] += xv;
                        s[2] += yv;
                        s[3] += xv * xv;
                        s[4] += xv * yv;
                        s[5] += yv * yv;
                        s[6 + k] += xv;
                    }
                }
                let (v, built) = coefficient(&s, p);
                vwin[b] = v;
                forced |= !built;
            }
        }
    }
    let vi = |i: usize, b: usize| -> f64 {
        if !two {
            1.0
        } else if cell_scope {
            let q = sc.list[i].0;
            let key = if d.cell[q] < 0 { d.ncell } else { d.cell[q] as usize };
            vgroup[sc.slot[key] as usize][b]
        } else {
            vwin[b]
        }
    };
    if cell_scope {
        // Forced to 1 anywhere among the groups of the similar pixels of the window.
        for &(q, _) in &sc.list {
            let key = if d.cell[q] < 0 { d.ncell } else { d.cell[q] as usize };
            let g = &sc.groups[sc.slot[key] as usize];
            for b in 0..nb {
                let s = &g.sums[b * 8..b * 8 + 8];
                forced |= !coefficient(s, p).1;
            }
        }
    }

    let mut vals = vec![f64::NAN; nb];
    for b in 0..nb {
        let mpc = d.mp[b][c];
        if !mpc.is_finite() {
            continue;
        }
        // Weights of the pixels with a coarse value at the predicted date, renormalised.
        let mut wsum = 0.0;
        for (i, &(q, _)) in sc.list.iter().enumerate() {
            if d.mp[b][q].is_finite() {
                wsum += w0[i];
            }
        }
        let use_base = !(wsum > 0.0);
        let mut norm = 0.0;
        if use_base {
            for (i, &(q, _)) in sc.list.iter().enumerate() {
                if d.mp[b][q].is_finite() {
                    norm += inv_d[i];
                }
            }
        }
        let mut pred = [0.0f64; 2];
        for (ki, &k) in ks.iter().enumerate() {
            let mut acc = 0.0;
            for (i, &(q, _)) in sc.list.iter().enumerate() {
                let mq = d.mp[b][q];
                if mq.is_finite() {
                    let w = if use_base { inv_d[i] / norm } else { w0[i] / wsum };
                    acc += w * vi(i, b) * (f64::from(mq) - f64::from(d.m[k][b][q]));
                }
            }
            pred[ki] = f64::from(d.f[k][b][c]) + acc;
        }
        vals[b] = if two {
            // Window changes since each pair (eq. 18).
            let mut delta = [0.0f64; 2];
            for (ki, &k) in ks.iter().enumerate() {
                let (mut sk, mut sp) = (0.0f64, 0.0f64);
                for yy in win.2..win.3 {
                    for xx in win.0..win.1 {
                        let q = yy * nx + xx;
                        let (a, bq) = (d.m[k][b][q], d.mp[b][q]);
                        if a.is_finite() && bq.is_finite() {
                            sk += f64::from(a);
                            sp += f64::from(bq);
                        }
                    }
                }
                delta[ki] = (sk - sp).abs();
            }
            let (tm, tn) = temporal_weights(delta[0], delta[1]);
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
    Est {
        vals,
        n: np as u32,
        src,
    }
}

/// The size of a coarse cell in fine pixels, from the geometry around the middle of the data extent.
pub fn cell_size_px(coarse: &Coarse, nx: usize, ny: usize) -> f64 {
    let (x, y) = (nx / 2, ny / 2);
    let (x1, y1) = ((x + 1).min(nx - 1), (y + 1).min(ny - 1));
    let (x, y) = if x1 == x { (x - 1, y) } else { (x, y) };
    let (y, y1) = if y1 == y { (y.saturating_sub(1), y1) } else { (y, y1) };
    let (u00, v00) = coarse.uv(x, y, nx);
    let (u10, v10) = coarse.uv(x1, y, nx);
    let (u01, v01) = coarse.uv(x, y1, nx);
    let det = ((u10 - u00) * (v01 - v00) - (u01 - u00) * (v10 - v00)).abs();
    if det > 0.0 {
        1.0 / det.sqrt()
    } else {
        1.0
    }
}

/// ESTARFM over `region`; same layout as [`super::starfm::starfm`]. Every target needs two pairs.
#[allow(clippy::too_many_arguments)]
pub fn estarfm(
    fine: &Fine,
    coarse: &Coarse,
    plan: &Plan,
    sigma: &[f64],
    p: &EstarfmParams,
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
    if plan.targets.iter().any(|t| t.pairs.len() != 2) {
        return Err("ESTARFM needs two pairs for every date".into());
    }
    if p.classes <= 0.0 {
        return Err("classes must be positive".into());
    }
    let cell = cell_map(coarse, nx, ny);
    let ext = cell_size_px(coarse, nx, ny).ceil() as usize + 1;
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
        let data = Data {
            nx,
            ny,
            nb,
            f: t.pairs
                .iter()
                .map(|&(k, _)| (0..nb).map(|b| fine.plane(k, b)).collect())
                .collect(),
            m: t.pairs
                .iter()
                .map(|&(_, cj)| (0..nb).map(|b| &cache[&cj][b * npix..(b + 1) * npix]).collect())
                .collect(),
            mp: (0..nb).map(|b| &mp_all[b * npix..(b + 1) * npix]).collect(),
            sigma: t
                .pairs
                .iter()
                .map(|&(k, _)| (0..nb).map(|b| sigma[k * nb + b]).collect())
                .collect(),
            cell: &cell,
            ncell: coarse.ncell(),
            ext,
        };
        let obs = t.observed;
        let res = rows(region.ny, n_jobs, |ry| {
            let y = region.y0 + ry;
            let mut sc = Scratch::new(coarse.ncell());
            let mut vals = vec![f32::NAN; nb * region.nx];
            let mut q = vec![0u16; 2 * region.nx];
            for rx in 0..region.nx {
                let x = region.x0 + rx;
                let c = y * nx + x;
                if let Some(fi) = obs {
                    if (0..nb).all(|b| fine.plane(fi, b)[c].is_finite()) {
                        for b in 0..nb {
                            vals[b * region.nx + rx] = fine.plane(fi, b)[c];
                        }
                        q[region.nx + rx] = SRC_OBSERVED;
                        continue;
                    }
                }
                let e = pixel(p, &data, &mut sc, x, y);
                for b in 0..nb {
                    vals[b * region.nx + rx] = e.vals[b] as f32;
                }
                q[rx] = e.n.min(u32::from(u16::MAX)) as u16;
                q[region.nx + rx] = e.src;
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
        let keep: Vec<usize> = plan.targets.iter().flat_map(|t| t.pairs.iter().map(|p| p.1)).collect();
        cache.retain(|k, _| keep.contains(k));
    }
    Ok(())
}
