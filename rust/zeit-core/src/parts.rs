//! PARTS (Ives et al. 2021, doi:10.1016/j.rse.2021.112678), remotePARTS's
//! fitGLS_partition: the GLS of a map's pixels (their trends, say) on a
//! design, in random partitions of the pixels, combined into tests that
//! account for the correlation between partitions: a t-test of each
//! coefficient and a chi-square test of the design against a null one.
//!
//! The correlation between two partitions i and j comes from R_ij = L_i^-1
//! V_ij L_j^-T (the cross-covariance under each partition's Cholesky factor).
//! remotePARTS forms the n x n projections H = xx (xx'xx)^-1 xx' and their
//! products with R_ij; here they are of rank p, H = QQ' with Q orthonormal, and
//! the traces reduce to Frobenius norms of Q_i' R_ij Q_j and of R_ij Q_j:
//! R_ij (two triangular solves) is the only n^3 term.
//!
//! remotePARTS recomputes the Cholesky factor of every partition after the
//! first without its nugget when it crosses them; here each partition keeps
//! its nugget. Both agree for a nugget 0.

use faer::linalg::solvers::DenseSolveCore;
use faer::linalg::triangular_solve::solve_lower_triangular_in_place;
use faer::{Mat, MatRef, Par};
use rayon::prelude::*;

use crate::dist::{imhof_upper, t_two_sided};
use crate::gls::{self, GlsFit, MlBy, Nugget, Whitened};
use crate::parallel;
use crate::spatial::{correlations, Correlation};

/// The pixels and the designs.
pub struct Data<'a> {
    /// coordinates of the pixels
    pub x: &'a [f64],
    pub y: &'a [f64],
    /// the response of each pixel
    pub response: &'a [f64],
    /// the design (pixels x p, by column) and the null design (pixels x p0)
    pub design: &'a [f64],
    pub p: usize,
    pub design0: &'a [f64],
    pub p0: usize,
}

/// How the partitions are fitted.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    pub cor: Correlation,
    /// distances divided by their largest in each matrix (remotePARTS's
    /// distm_scaled, its default)
    pub scaled: bool,
    pub nugget: Nugget,
    /// the partitions crossed with each other (remotePARTS's ncross, 6)
    pub ncross: usize,
    pub n_jobs: i32,
}

/// The statistics between two partitions.
#[derive(Clone, Debug, PartialEq)]
pub struct Cross {
    pub i: usize,
    pub j: usize,
    /// the correlation of their coefficients (p x p, by column)
    pub rcoef: Vec<f64>,
    pub rssr: f64,
    pub rsse: f64,
}

/// The combined t-test of each coefficient.
#[derive(Clone, Debug, PartialEq)]
pub struct TTest {
    pub coefficients: Vec<f64>,
    pub se: Vec<f64>,
    pub t: Vec<f64>,
    pub p: Vec<f64>,
    /// the covariance of the coefficients (p x p, by column)
    pub covar: Vec<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct PartsFit {
    /// the fit of each partition
    pub parts: Vec<GlsFit>,
    pub cross: Vec<Cross>,
    /// the means of the partitions' coefficients
    pub coefficients: Vec<f64>,
    /// the mean of the pairs' rcoef
    pub rcoefficients: Vec<f64>,
    pub rssr: f64,
    pub rsse: f64,
    /// the mean of the partitions' F
    pub f: f64,
    /// remotePARTS's calc_dfpart: p - p0, and partsize - (p - 1)
    pub df1: usize,
    pub df2: usize,
    pub t_test: TTest,
    /// the chi-square test of the design against the null one, when p0 < p
    pub p_chisqr: Option<f64>,
    /// the chi-square test's covariance was not positive definite, and was
    /// replaced as remotePARTS does (its rankdef.MSR attribute is not 0)
    pub rank_deficient: bool,
}

/// Fits the partitions (`partitions`: `partsize` pixel indices per partition,
/// partition after partition) and combines them. None when a partition's
/// covariance or design is singular.
pub fn fit(data: &Data, partitions: &[usize], partsize: usize, opts: &Options) -> Option<PartsFit> {
    let npart = partitions.len() / partsize;
    assert_eq!(
        partitions.len(),
        npart * partsize,
        "partitions must be partsize x npart"
    );
    let (p, p0) = (data.p, data.p0);
    let ncross = opts.ncross.min(npart);
    let threads = parallel::threads(opts.n_jobs);
    let part = |k: usize| &partitions[k * partsize..(k + 1) * partsize];
    let coords = |k: usize| -> (Vec<f64>, Vec<f64>) {
        (
            part(k).iter().map(|&i| data.x[i]).collect(),
            part(k).iter().map(|&i| data.y[i]).collect(),
        )
    };
    // each thread's covariance and factor, reused from partition to partition (the n x n
    // matrices are what bounds PARTS on many threads: memory)
    let buffers = || {
        (
            Mat::<f64>::zeros(partsize, partsize),
            Mat::<f64>::zeros(partsize, partsize),
        )
    };
    let fits: Vec<Option<(GlsFit, Option<Whitened>)>> = parallel::pool(threads).install(|| {
        (0..npart)
            .into_par_iter()
            .map_init(buffers, |(v, work), k| {
                let idx = part(k);
                let (cx, cy) = coords(k);
                fill_correlations(v, &cx, &cy, opts.cor, opts.scaled);
                let column = |m: &[f64], cols: usize| -> Vec<f64> {
                    (0..cols)
                        .flat_map(|c| idx.iter().map(move |&i| m[c * data.x.len() + i]))
                        .collect()
                };
                let (x, x0) = (column(data.design, p), column(data.design0, p0));
                let y: Vec<f64> = idx.iter().map(|&i| data.response[i]).collect();
                gls::fit_with(
                    v.as_ref(),
                    &x,
                    p,
                    &x0,
                    p0,
                    &y,
                    opts.nugget,
                    MlBy::Cholesky,
                    k < ncross,
                    work,
                    Par::Seq,
                )
            })
            .collect()
    });
    let mut parts = Vec::with_capacity(npart);
    let mut whitened = Vec::with_capacity(ncross);
    for f in fits {
        let (fit, white) = f?;
        parts.push(fit);
        if let Some(w) = white {
            whitened.push(w);
        }
    }
    let (df1, df2) = (p - p0.min(p), partsize + 1 - p);
    let pairs: Vec<(usize, usize)> = (0..ncross).flat_map(|i| (i + 1..ncross).map(move |j| (i, j))).collect();
    let cross: Vec<Cross> = parallel::pool(threads).install(|| {
        pairs
            .par_iter()
            .map(|&(i, j)| {
                let ((xi, yi), (xj, yj)) = (coords(i), coords(j));
                let vij = correlations(&xi, &yi, &xj, &yj, opts.cor, opts.scaled);
                cross_stats(
                    i,
                    j,
                    &whitened[i],
                    &whitened[j],
                    vij,
                    parts[i].nugget,
                    parts[j].nugget,
                    df1,
                    df2,
                )
            })
            .collect()
    });
    Some(combine(parts, cross, p, p0, df1, df2))
}

/// The correlations among the points (`x`, `y`) into the lower triangle of
/// `out` (as `spatial::correlations`, its distances divided by their largest
/// with `scaled`); the upper triangle is not read.
fn fill_correlations(out: &mut Mat<f64>, x: &[f64], y: &[f64], cor: Correlation, scaled: bool) {
    let n = x.len();
    let squared = |i: usize, j: usize| {
        let (dx, dy) = (x[i] - x[j], y[i] - y[j]);
        dx * dx + dy * dy
    };
    let largest = if scaled {
        (0..n)
            .flat_map(|j| (j..n).map(move |i| (i, j)))
            .map(|(i, j)| squared(i, j))
            .fold(0.0, f64::max)
            .sqrt()
    } else {
        0.0
    };
    let scale = if largest > 0.0 { largest } else { 1.0 };
    for j in 0..n {
        for i in j..n {
            out[(i, j)] = cor.at(squared(i, j).sqrt() / scale);
        }
    }
}

/// remotePARTS's crosspart_GLS between partitions i and j.
#[allow(clippy::too_many_arguments)]
fn cross_stats(
    i: usize,
    j: usize,
    wi: &Whitened,
    wj: &Whitened,
    vij: Vec<f64>,
    nug_i: f64,
    nug_j: f64,
    df1: usize,
    df2: usize,
) -> Cross {
    let n = wi.l.nrows();
    let s = ((1.0 - nug_i) * (1.0 - nug_j)).sqrt();
    // R_ij = L_i^-1 V_ij L_j^-T: L_i^-1 V_ij, then L_j^-1 of its transpose, which is R_ij'
    let mut a = Mat::from_fn(n, n, |r, c| s * vij[c * n + r]);
    solve_lower_triangular_in_place(wi.l.as_ref(), a.as_mut(), Par::Seq);
    let mut rt = a.transpose().to_owned();
    solve_lower_triangular_in_place(wj.l.as_ref(), rt.as_mut(), Par::Seq);
    let r = rt.transpose();
    let (qi, qi0) = (orthonormal(wi.xx.as_ref()), orthonormal(wi.xx0.as_ref()));
    let (qj, qj0) = (orthonormal(wj.xx.as_ref()), orthonormal(wj.xx0.as_ref()));
    let rqj = r * &qj;
    let rqj0 = r * &qj0;
    let fro = |m: MatRef<f64>| m.squared_norm_l2();
    // tr((H_i - H_i0) R (H_j - H_j0) R')
    let ssr = fro((qi.transpose() * &rqj).as_ref())
        - fro((qi.transpose() * &rqj0).as_ref())
        - fro((qi0.transpose() * &rqj).as_ref())
        + fro((qi0.transpose() * &rqj0).as_ref());
    // tr((I - H_i) R (I - H_j) R')
    let qtr = qi.transpose() * r;
    let sse = fro(r) - fro(qtr.as_ref()) - fro(rqj.as_ref()) + fro((qi.transpose() * &rqj).as_ref());
    // Vcoef = W_i xx_i' R xx_j W_j', W = (xx'xx)^-1; rcoef its correlation scaling
    let wmat = |xx: MatRef<f64>| -> Mat<f64> {
        let g = xx.transpose() * xx;
        g.llt(faer::Side::Lower)
            .map(|c| c.inverse())
            .unwrap_or_else(|_| Mat::from_fn(g.nrows(), g.ncols(), |_, _| f64::NAN))
    };
    let (w_i, w_j) = (wmat(wi.xx.as_ref()), wmat(wj.xx.as_ref()));
    let vcoef = &w_i * (wi.xx.transpose() * (r * &wj.xx)) * w_j.transpose();
    let p = vcoef.nrows();
    let rcoef = (0..p * p)
        .map(|e| {
            let (a, b) = (e % p, e / p);
            vcoef[(a, b)] / (w_i[(a, a)] * w_j[(b, b)]).sqrt()
        })
        .collect();
    Cross {
        i,
        j,
        rcoef,
        rssr: ssr / df1 as f64,
        rsse: sse / df2 as f64,
    }
}

/// An orthonormal basis of the columns of `m` (thin QR).
fn orthonormal(m: MatRef<f64>) -> Mat<f64> {
    if m.ncols() == 0 {
        return Mat::zeros(m.nrows(), 0);
    }
    m.qr().compute_thin_Q()
}

/// The mean of the finite values, NaN without one (R's mean(na.rm = TRUE)).
fn mean(v: impl Iterator<Item = f64>) -> f64 {
    let (s, n) = v
        .filter(|x| x.is_finite())
        .fold((0.0, 0.0), |(s, n), x| (s + x, n + 1.0));
    if n > 0.0 {
        s / n
    } else {
        f64::NAN
    }
}

fn combine(parts: Vec<GlsFit>, cross: Vec<Cross>, p: usize, p0: usize, df1: usize, df2: usize) -> PartsFit {
    let npart = parts.len();
    let coefficients: Vec<f64> = (0..p).map(|c| mean(parts.iter().map(|f| f.coefficients[c]))).collect();
    let rcoefficients: Vec<f64> = (0..p * p).map(|e| mean(cross.iter().map(|c| c.rcoef[e]))).collect();
    let rssr = mean(cross.iter().map(|c| c.rssr));
    let rsse = mean(cross.iter().map(|c| c.rsse));
    let f = mean(parts.iter().map(|f| f.null.as_ref().map_or(f64::NAN, |n| n.f)));
    let t_test = t_test(&parts, &coefficients, &rcoefficients, df2, p);
    let (p_chisqr, rank_deficient) = if p0 < p && !cross.is_empty() {
        let (pv, def) = chisqr(f, rssr, df1, npart);
        (Some(pv), def)
    } else {
        (None, false)
    };
    PartsFit {
        parts,
        cross,
        coefficients,
        rcoefficients,
        rssr,
        rsse,
        f,
        df1,
        df2,
        t_test,
        p_chisqr,
        rank_deficient,
    }
}

/// remotePARTS's part_ttest.
fn t_test(parts: &[GlsFit], coefficients: &[f64], rcoef: &[f64], df2: usize, p: usize) -> TTest {
    let npart = parts.len() as f64;
    let se: Vec<f64> = (0..p)
        .map(|c| {
            // sqrt(se' ((1 - r) I + r J) se) / npart
            let r = rcoef[c * p + c];
            let ses = parts.iter().map(|f| f.covar[c * p + c].sqrt());
            let (s1, s2) = ses.fold((0.0, 0.0), |(s1, s2), s| (s1 + s, s2 + s * s));
            ((1.0 - r) * s2 + r * s1 * s1).sqrt() / npart
        })
        .collect();
    let covar = (0..p * p)
        .map(|e| {
            let s: f64 = parts.iter().map(|f| f.covar[e]).sum();
            (s + rcoef[e] * (npart - 1.0) * s.abs()) / (npart * npart)
        })
        .collect();
    let t: Vec<f64> = coefficients.iter().zip(&se).map(|(c, s)| c / s).collect();
    let pv = t.iter().map(|&t| t_two_sided(t, df2 as f64 * npart)).collect();
    TTest {
        coefficients: coefficients.to_vec(),
        se,
        t,
        p: pv,
        covar,
    }
}

/// remotePARTS's part_chisqr: the p-value of npart df1 Fmean under the
/// correlated chi-square, and whether its covariance was rank-deficient
/// (replaced as remotePARTS does). The covariance, I - rZ (I_npart x J_df1) +
/// rZ J, rZ = sqrt(rSSR / df1), has three eigenvalues: 1 + rZ (N - df1) once,
/// 1 - rZ df1 npart - 1 times, 1 for the rest (N = npart df1).
pub fn chisqr(fmean: f64, rssr: f64, df1: usize, npart: usize) -> (f64, bool) {
    if !fmean.is_finite() || !rssr.is_finite() || df1 == 0 {
        return (f64::NAN, false);
    }
    let (d, m) = (df1 as f64, npart as f64);
    let n = d * m;
    let mut rz = (rssr / d).sqrt();
    // R's chol(pivot = TRUE) stops at a pivot below N eps (its diagonal is 1)
    let rank_deficient = !(1.0 - rz * d > n * f64::EPSILON);
    if rank_deficient {
        rz = 0.99 / d;
    }
    let lambdas = [(1.0 + rz * (n - d), 1.0), (1.0 - rz * d, m - 1.0), (1.0, m * (d - 1.0))];
    let pv = imhof_upper(n * fmean, &lambdas);
    (if pv <= 1e-6 { 1e-6 } else { pv }, rank_deficient)
}

/// `npart` partitions of `partsize` pixels drawn without replacement from
/// `npix` (remotePARTS's sample_partitions), by a seeded generator
/// (xoshiro256**): partition after partition.
pub fn sample_partitions(npix: usize, partsize: usize, npart: usize, seed: u64) -> Vec<usize> {
    assert!(
        partsize * npart <= npix,
        "npart * partsize may not be greater than npix"
    );
    let mut sm = seed;
    let mut splitmix = || {
        sm = sm.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = sm;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    };
    let mut s = [splitmix(), splitmix(), splitmix(), splitmix()];
    let mut next = || {
        let r = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        r
    };
    let mut idx: Vec<usize> = (0..npix).collect();
    let take = partsize * npart;
    for k in 0..take {
        // a uniform index in k..npix (Lemire's multiply-shift, its bias below 2^-40 here)
        let span = (npix - k) as u64;
        let pick = k + ((u128::from(next()) * u128::from(span)) >> 64) as usize;
        idx.swap(k, pick);
    }
    idx.truncate(take);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn partitions_are_disjoint_and_reproducible() {
        let a = sample_partitions(1000, 30, 20, 7);
        assert_eq!(a, sample_partitions(1000, 30, 20, 7));
        assert_ne!(a, sample_partitions(1000, 30, 20, 8));
        let mut s = a.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), 600);
        assert!(s.iter().all(|&i| i < 1000));
    }

    #[test]
    fn chi_square_eigenvalues() {
        // rZ = 0: independent partitions, a chi-square of N = npart df1 degrees of freedom
        let (pv, def) = chisqr(1.3, 0.0, 2, 5);
        // P(X > 13) for X of 10 degrees of freedom
        let want = (-6.5f64).exp()
            * (0..5)
                .map(|k| 6.5f64.powi(k) / (1..=k).product::<i32>().max(1) as f64)
                .sum::<f64>();
        assert!((pv - want).abs() < 1e-10 && !def, "{pv} {want}");
        // a covariance that is not positive definite is replaced
        assert!(chisqr(1.3, 0.9, 2, 5).1);
    }
}
