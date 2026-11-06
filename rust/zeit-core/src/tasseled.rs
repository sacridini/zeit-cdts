//! Tasseled Cap: the weighted sum of the six reflective bands (blue, green, red, NIR, SWIR1, SWIR2) of every
//! date, and the disturbance `B - G - W` made of three of its rows.
//!
//! The arithmetic is FORCE's `index_tasseled` (src/modules/higher-level/index-compute-hl.c, FORCE 3.11.00, David
//! Frantz, GPL-3.0-or-later): float32 throughout, the six products added in the band order, no fused multiply-add,
//! so that the values truncated to integers are FORCE's. The matrices (Crist 1985, Huang et al. 2002, Baig et al.
//! 2014) are the caller's.
//!
//! A value equal to one of the `missing` numbers (compared as float32) or NaN makes the pixel NaN in every
//! component; infinities are not special (they go through the sums as float32 arithmetic has them).

use crate::parallel::{pool, threads, Shared};
use rayon::prelude::*;

/// The six bands of the matrices.
pub const BANDS: usize = 6;

/// Pixels per job: the six converted bands and the rows being summed stay in the L1/L2 cache.
const TILE: usize = 4096;

/// A sample type the kernel converts to float32 as numpy's `astype(float32)` does (round to nearest even).
pub trait Sample32: Copy + Send + Sync + 'static {
    fn to_f32(self) -> f32;
}

macro_rules! sample32 {
    ($($t:ty),*) => {$(
        impl Sample32 for $t {
            #[inline(always)]
            fn to_f32(self) -> f32 {
                self as f32
            }
        }
    )*};
}
sample32!(i16, u16, i32, u32, f32, f64);

/// What a job needs, shared by the threads.
struct Plan<'a, T> {
    x: &'a [T],
    src: [usize; BANDS],
    matrices: &'a [f32],
    rows: &'a [usize],
    disturbance: Option<[usize; 3]>,
    missing: Vec<f32>,
    nb: usize,
    nm: usize,
    npix: usize,
    ncomp: usize,
    /// The matrix rows to sum, sorted, each once, and where each is in `acc`.
    needed: Vec<usize>,
}

/// The scratch of a thread: the six bands of a tile and the sums of the needed rows.
struct Scratch {
    x: Vec<f32>,
    acc: Vec<f32>,
}

impl<T: Sample32> Plan<'_, T> {
    fn scratch(&self) -> Scratch {
        Scratch {
            x: vec![0.0; BANDS * TILE],
            acc: vec![0.0; self.needed.len() * TILE],
        }
    }

    /// Date `k`, pixels `p0..p1` of the image: into `out` (every component plane of the whole output).
    fn tile(&self, k: usize, p0: usize, p1: usize, out: &Shared<f32>, s: &mut Scratch) {
        let n = p1 - p0;
        for (b, &band) in self.src.iter().enumerate() {
            let from = &self.x[(k * self.nb + band) * self.npix + p0..][..n];
            let to = &mut s.x[b * TILE..][..n];
            for (t, &v) in to.iter_mut().zip(from) {
                *t = v.to_f32();
            }
            for &m in &self.missing {
                for t in to.iter_mut() {
                    if *t == m {
                        *t = f32::NAN;
                    }
                }
            }
        }
        let (x0, rest) = s.x.split_at(TILE);
        let (x1, rest) = rest.split_at(TILE);
        let (x2, rest) = rest.split_at(TILE);
        let (x3, rest) = rest.split_at(TILE);
        let (x4, x5) = rest.split_at(TILE);
        let (x0, x1, x2, x3, x4, x5) = (&x0[..n], &x1[..n], &x2[..n], &x3[..n], &x4[..n], &x5[..n]);
        for (slot, &r) in self.needed.iter().enumerate() {
            let m = &self.matrices[(k * self.nm + r) * BANDS..][..BANDS];
            let acc = &mut s.acc[slot * TILE..][..n];
            for i in 0..n {
                // FORCE's sum, in its order, in float32
                let mut a = m[0] * x0[i];
                a += m[1] * x1[i];
                a += m[2] * x2[i];
                a += m[3] * x3[i];
                a += m[4] * x4[i];
                a += m[5] * x5[i];
                acc[i] = a;
            }
        }
        let slot = |r: usize| self.needed.binary_search(&r).expect("needed row");
        let plane = |c: usize| {
            // SAFETY: each job writes its own date and pixels of the planes, which no other job touches.
            unsafe { out.slice_mut(((k * self.ncomp + c) * self.npix + p0)..((k * self.ncomp + c) * self.npix + p1)) }
        };
        for (c, &r) in self.rows.iter().enumerate() {
            plane(c).copy_from_slice(&s.acc[slot(r) * TILE..][..n]);
        }
        if let Some([b, g, w]) = self.disturbance {
            let (b, g, w) = (
                &s.acc[slot(b) * TILE..][..n],
                &s.acc[slot(g) * TILE..][..n],
                &s.acc[slot(w) * TILE..][..n],
            );
            let o = plane(self.rows.len());
            for i in 0..n {
                o[i] = (b[i] - g[i]) - w[i];
            }
        }
    }
}

/// The Tasseled Cap components of `x` (time, band, pixels, C order, `npix` pixels of each plane), into `out`
/// (time, component, pixels): the components are `rows` (rows of `matrices`, which is (time, `nm` rows, 6)
/// float32 for the bands `src` of `x`), then `B - G - W` of the rows `disturbance` if given. Every value of
/// `out` is written. `n_jobs` as everywhere.
#[allow(clippy::too_many_arguments)]
pub fn tasseled_batch<T: Sample32>(
    x: &[T],
    nb: usize,
    npix: usize,
    src: [usize; BANDS],
    matrices: &[f32],
    nm: usize,
    rows: &[usize],
    disturbance: Option<[usize; 3]>,
    missing: &[f64],
    out: &mut [f32],
    n_jobs: i32,
) {
    let nt = if nb * npix == 0 { 0 } else { x.len() / (nb * npix) };
    let ncomp = rows.len() + usize::from(disturbance.is_some());
    assert!(src.iter().all(|&b| b < nb), "src out of the bands");
    assert!(
        rows.iter().chain(disturbance.iter().flatten()).all(|&r| r < nm),
        "row out of the matrix"
    );
    assert_eq!(x.len(), nt * nb * npix);
    assert_eq!(matrices.len(), nt * nm * BANDS, "matrices must be (time, rows, 6)");
    assert_eq!(out.len(), nt * ncomp * npix);
    if nt == 0 || npix == 0 || ncomp == 0 {
        return;
    }
    let mut needed: Vec<usize> = rows
        .iter()
        .copied()
        .chain(disturbance.iter().flatten().copied())
        .collect();
    needed.sort_unstable();
    needed.dedup();
    let plan = Plan {
        x,
        src,
        matrices,
        rows,
        disturbance,
        missing: missing.iter().map(|&m| m as f32).collect(),
        nb,
        nm,
        npix,
        ncomp,
        needed,
    };
    let per_date = npix.div_ceil(TILE);
    let shared = Shared::new(out);
    let job = |s: &mut Scratch, j: usize| {
        let (k, t) = (j / per_date, j % per_date);
        plan.tile(k, t * TILE, ((t + 1) * TILE).min(npix), &shared, s);
    };
    let jobs = nt * per_date;
    let t = threads(n_jobs).min(jobs);
    if t <= 1 {
        let mut s = plan.scratch();
        (0..jobs).for_each(|j| job(&mut s, j));
    } else {
        pool(t).install(|| {
            (0..jobs)
                .into_par_iter()
                .for_each_init(|| plan.scratch(), |s, j| job(s, j));
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sum the plain way, one pixel.
    fn sum(m: &[f32], x: [f32; 6]) -> f32 {
        let mut a = m[0] * x[0];
        for j in 1..6 {
            a += m[j] * x[j];
        }
        a
    }

    const M: [[f32; 6]; 3] = [
        [0.2043, 0.4158, 0.5524, 0.5741, 0.3124, 0.2303],
        [-0.1603, -0.2819, -0.4934, 0.794, -0.0002, -0.1446],
        [0.0315, 0.2021, 0.3102, 0.1594, -0.6806, -0.6109],
    ];

    #[test]
    fn a_known_product_and_the_disturbance() {
        // 2 dates, 5 bands of the cube (the kernel reads 6 of them, in the order src), 3 pixels
        let (nt, nb, npix) = (2, 7, 3);
        let x: Vec<i16> = (0..nt * nb * npix).map(|i| ((i * 37) % 1000) as i16 + 100).collect();
        let src = [6, 5, 4, 3, 2, 1];
        let m: Vec<f32> = (0..nt).flat_map(|_| M.iter().flatten().copied()).collect();
        for n_jobs in [1, 3] {
            let mut out = vec![0f32; nt * 3 * npix];
            tasseled_batch(
                &x,
                nb,
                npix,
                src,
                &m,
                3,
                &[2, 0],
                Some([0, 1, 2]),
                &[],
                &mut out,
                n_jobs,
            );
            for k in 0..nt {
                for p in 0..npix {
                    let v = src.map(|b| f32::from(x[(k * nb + b) * npix + p]));
                    let (b, g, w) = (sum(&M[0], v), sum(&M[1], v), sum(&M[2], v));
                    let at = |c: usize| out[(k * 3 + c) * npix + p];
                    assert_eq!(at(0).to_bits(), w.to_bits());
                    assert_eq!(at(1).to_bits(), b.to_bits());
                    assert_eq!(at(2).to_bits(), ((b - g) - w).to_bits());
                }
            }
        }
    }

    #[test]
    fn the_first_pixel_by_hand() {
        // all bands 1000: brightness 1000 * (sum of the row), as float32 sums do
        let x = vec![1000i16; 6];
        let mut out = vec![0f32; 1];
        let m: Vec<f32> = M[0].to_vec();
        tasseled_batch(&x, 6, 1, [0, 1, 2, 3, 4, 5], &m, 1, &[0], None, &[], &mut out, 1);
        assert!((out[0] - 2289.3).abs() < 0.01, "{}", out[0]);
    }

    #[test]
    fn missing_and_nan_make_nan() {
        // 1 date, 6 bands, 4 pixels: pixel 1 has -9999 in one band, pixel 2 a NaN, pixel 3 is fine
        let mut x = vec![500f32; 24];
        x[2 * 4 + 1] = -9999.0;
        x[4 * 4 + 2] = f32::NAN;
        let m: Vec<f32> = M[0].to_vec();
        let mut out = vec![0f32; 4];
        tasseled_batch(&x, 6, 4, [0, 1, 2, 3, 4, 5], &m, 1, &[0], None, &[-9999.0], &mut out, 1);
        assert!(out[0].is_finite() && out[3].is_finite());
        assert!(out[1].is_nan() && out[2].is_nan());
        // without the missing value it is a number
        tasseled_batch(&x, 6, 4, [0, 1, 2, 3, 4, 5], &m, 1, &[0], None, &[], &mut out, 1);
        assert!(out[1].is_finite() && out[2].is_nan());
    }

    #[test]
    fn many_tiles_and_types_agree() {
        let (nt, nb, npix) = (3, 6, TILE * 2 + 17);
        let x: Vec<u16> = (0..nt * nb * npix).map(|i| ((i * 7919) % 10000) as u16).collect();
        let xf: Vec<f64> = x.iter().map(|&v| f64::from(v)).collect();
        let m: Vec<f32> = (0..nt).flat_map(|_| M.iter().flatten().copied()).collect();
        let src = [0, 1, 2, 3, 4, 5];
        let mut a = vec![0f32; nt * 4 * npix];
        let mut b = vec![0f32; nt * 4 * npix];
        let mut c = vec![0f32; nt * 4 * npix];
        tasseled_batch(&x, nb, npix, src, &m, 3, &[0, 1, 2], Some([0, 1, 2]), &[0.0], &mut a, 1);
        tasseled_batch(
            &xf,
            nb,
            npix,
            src,
            &m,
            3,
            &[0, 1, 2],
            Some([0, 1, 2]),
            &[0.0],
            &mut b,
            1,
        );
        tasseled_batch(&x, nb, npix, src, &m, 3, &[0, 1, 2], Some([0, 1, 2]), &[0.0], &mut c, 3);
        let same = |p: &[f32], q: &[f32]| p.iter().zip(q).all(|(u, v)| u.to_bits() == v.to_bits());
        assert!(same(&a, &b) && same(&a, &c));
    }
}
