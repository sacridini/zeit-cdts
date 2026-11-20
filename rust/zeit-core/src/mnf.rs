//! The Minimum Noise Fraction (MNF) transform: components of a multiband image ordered by signal-to-noise ratio
//! instead of variance, with the noise taken from the differences between neighbouring pixels, and its inverse
//! (the denoising of the bands by the first components).
//!
//! References: Green, Berman, Switzer and Craig (1988, IEEE TGRS 26:65-74), the transform; Lee, Woodyatt and
//! Berman (1990, IEEE TGRS 28:295-304), the same components as a noise whitening followed by a principal
//! components rotation (the form solved here); Switzer and Green (1984) and Nielsen, Conradsen and Simpson (1998,
//! RSE 64:1-19), the min/max autocorrelation factors that the shift-difference noise makes MNF equal to; Chan,
//! Golub and LeVeque (1983, The American Statistician 37:242-247), the merging of partial moments.
//!
//! The model is `Z = S + N` with the signal smooth and the noise white: `Sigma = Sigma_S + Sigma_N`, and
//! `Sigma_N = Sigma_D / 2` with `Sigma_D` the covariance of `z(p) - z(q)` over the pairs of a shift `q = p + d`.
//! The components solve `Sigma a = lambda Sigma_N a` with `a' Sigma_N a = 1`, so that each has noise of unit
//! variance and a variance `lambda = 1 + SNR`.
//!
//! The statistics (valid-pixel mean, total covariance, covariance of the differences) are gathered in blocks of
//! whole rows with one halo row below (so that every pair is counted once), each block in f64 and in its own
//! [`MnfStats`], and merged in block order: the result does not depend on the threads. Every statistic is
//! computed from the sample converted straight to f64. A pixel counts only if all the bands it uses are finite,
//! different from every `missing` value and, with a `mask`, in it; a pair counts only if both pixels do.
//!
//! The solution is the two-rotation form with a diagonal pre-scaling of the bands (MNF does not depend on the
//! units of a band), a relative floor under the noise eigenvalues, and a sign rule that does not change under a
//! positive scaling or a shift of a band (the component correlates positively with the band it is most
//! correlated with).

use crate::parallel::{pool, threads, Shared};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par, Side};
use rayon::prelude::*;
use std::fmt;

/// The relative floor of the noise eigenvalues (after the pre-scaling).
pub const EPS_NOISE: f64 = 1e-9;
/// A component whose eigenvalue is below this has no variance (a band that combines others exactly).
pub const DEGENERATE: f64 = 1e-6;

/// Bytes of a block of rows converted to f64.
const BLOCK_BYTES: usize = 8 << 20;
/// Blocks held at once between two merges: fixed, so that the sums do not depend on the threads.
const WAVE: usize = 64;
/// Pixels per job of the transforms.
const TILE: usize = 512;

/// A sample type the statistics convert to f64 exactly (integers up to 2^53, f32 and f64 as they are).
pub trait Sample64: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
}

macro_rules! sample64 {
    ($($t:ty),*) => {$(
        impl Sample64 for $t {
            #[inline(always)]
            fn to_f64(self) -> f64 {
                self as f64
            }
        }
    )*};
}
sample64!(i16, u16, i32, u32, f32, f64);

/// What the noise covariance is made of.
#[derive(Clone, Debug, PartialEq)]
pub enum Noise {
    /// Differences with the pixel to the right.
    DiffH,
    /// Differences with the pixel below.
    DiffV,
    /// Right and below, pooled (each direction centred on its own mean).
    DiffHv,
    /// Lower right.
    DiffDiag,
    /// Lower left.
    DiffAntidiag,
    /// A noise covariance (`nb * nb`, row-major) from elsewhere.
    Given(Vec<f64>),
}

impl Noise {
    /// The shifts `(dy, dx)` of the pairs.
    pub fn directions(&self) -> Vec<(i32, i32)> {
        match self {
            Noise::DiffH => vec![(0, 1)],
            Noise::DiffV => vec![(1, 0)],
            Noise::DiffHv => vec![(0, 1), (1, 0)],
            Noise::DiffDiag => vec![(1, 1)],
            Noise::DiffAntidiag => vec![(1, -1)],
            Noise::Given(_) => vec![],
        }
    }
}

fn direction_name(dy: i32, dx: i32) -> &'static str {
    match (dy, dx) {
        (0, 1) => "h",
        (1, 0) => "v",
        (1, 1) => "diag",
        (1, -1) => "antidiag",
        _ => "?",
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum MnfError {
    /// Fewer valid pixels than `need` (the bands plus one).
    TooFewPixels {
        n: u64,
        need: u64,
    },
    /// Fewer pairs of a direction than `need` (the bands plus one).
    TooFewPairs {
        direction: &'static str,
        m: u64,
        need: u64,
    },
    NoValidPixel,
    /// The noise has no variance in any direction.
    NoNoise,
    /// The given noise covariance is not a finite symmetric `nb x nb` matrix with a non-negative diagonal.
    BadNoiseCov(String),
    AllBandsConstant,
    /// A parameter that cannot be.
    BadInput(String),
    /// The eigensolver did not converge.
    Eigen(String),
}

impl fmt::Display for MnfError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MnfError::TooFewPixels { n, need } => {
                write!(
                    f,
                    "only {n} valid pixels: the MNF needs at least {need} (one more than the bands)"
                )
            }
            MnfError::TooFewPairs { direction, m, need } => write!(
                f,
                "only {m} pairs of valid neighbouring pixels in the direction {direction}: the noise estimate needs at \
                 least {need} (one more than the bands); take a larger noise region"
            ),
            MnfError::NoValidPixel => write!(f, "no pixel has every band valid"),
            MnfError::NoNoise => write!(f, "the noise estimate is zero in every direction (no noise at all)"),
            MnfError::BadNoiseCov(s) => write!(f, "bad noise covariance: {s}"),
            MnfError::AllBandsConstant => write!(f, "every band is constant"),
            MnfError::BadInput(s) => write!(f, "{s}"),
            MnfError::Eigen(s) => write!(f, "the eigendecomposition did not converge: {s}"),
        }
    }
}

impl std::error::Error for MnfError {}

/// How the statistics are gathered.
pub struct StatsOptions<'a> {
    pub noise: Noise,
    /// `ny * nx`, nonzero = the pixel is used.
    pub mask: Option<&'a [u8]>,
    /// `ny * nx`, nonzero = inside: the differences are taken only between two pixels inside.
    pub noise_region: Option<&'a [u8]>,
    /// The values that mean missing besides NaN (compared as f64).
    pub missing: &'a [f64],
    /// The fraction of the pixels (and of the pair anchors) used, in (0, 1].
    pub sample_fraction: f64,
    pub seed: u64,
    /// The index of the first date of the data, the `t` of the sampling key (for a cube cut by date).
    pub date0: usize,
    /// `nb` values, nonzero = the band counts in the validity of a pixel (none: all do). A band with no valid
    /// value at all is left out by [`mnf_stats_batch`].
    pub active: Option<&'a [u8]>,
}

impl<'a> StatsOptions<'a> {
    pub fn new(noise: Noise) -> Self {
        StatsOptions {
            noise,
            mask: None,
            noise_region: None,
            missing: &[],
            sample_fraction: 1.0,
            seed: 0,
            date0: 0,
            active: None,
        }
    }
}

/// The moments of the differences of one direction.
#[derive(Clone, Debug, PartialEq)]
pub struct DiffStats {
    pub dy: i32,
    pub dx: i32,
    /// The pairs.
    pub m: u64,
    /// The mean of the differences, `nb`.
    pub mean: Vec<f64>,
    /// The centred co-moment of the differences, `nb x nb`.
    pub m2: Vec<f64>,
}

/// Partial statistics of a part of the image; [`MnfStats::merge`] is Chan's update.
#[derive(Clone, Debug, PartialEq)]
pub struct MnfStats {
    pub nb: usize,
    /// The valid (and sampled) pixels.
    pub n: u64,
    pub mean: Vec<f64>,
    /// `sum (z - mean)(z - mean)^T`, `nb x nb`.
    pub m2: Vec<f64>,
    /// One per direction of the noise estimator.
    pub pairs: Vec<DiffStats>,
    /// The values of each band that are finite and not missing (in the mask), sampled or not.
    pub band_valid: Vec<u64>,
    /// The given noise covariance, if any.
    pub given: Option<Vec<f64>>,
}

/// Chan, Golub and LeVeque's update of `(n_a, mean_a, m2_a)` by `(n_b, mean_b, m2_b)`.
fn chan(n_a: u64, mean_a: &mut [f64], m2_a: &mut [f64], n_b: u64, mean_b: &[f64], m2_b: &[f64]) {
    if n_b == 0 {
        return;
    }
    if n_a == 0 {
        mean_a.copy_from_slice(mean_b);
        m2_a.copy_from_slice(m2_b);
        return;
    }
    let nb = mean_a.len();
    let n = (n_a + n_b) as f64;
    let w = n_a as f64 * n_b as f64 / n;
    let delta: Vec<f64> = (0..nb).map(|i| mean_b[i] - mean_a[i]).collect();
    for i in 0..nb {
        for j in 0..nb {
            m2_a[i * nb + j] += m2_b[i * nb + j] + delta[i] * delta[j] * w;
        }
    }
    for i in 0..nb {
        mean_a[i] += delta[i] * (n_b as f64 / n);
    }
}

fn check_given(given: &[f64], nb: usize) -> Result<(), MnfError> {
    if given.len() != nb * nb {
        return Err(MnfError::BadNoiseCov(format!(
            "it has {} values, not {nb} x {nb}",
            given.len()
        )));
    }
    if given.iter().any(|v| !v.is_finite()) {
        return Err(MnfError::BadNoiseCov("it has values that are not finite".into()));
    }
    let scale = given.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    for i in 0..nb {
        if given[i * nb + i] < 0.0 {
            return Err(MnfError::BadNoiseCov(format!("the variance of band {i} is negative")));
        }
        for j in 0..i {
            if (given[i * nb + j] - given[j * nb + i]).abs() > 1e-9 * scale {
                return Err(MnfError::BadNoiseCov(format!("it is not symmetric at ({i}, {j})")));
            }
        }
    }
    Ok(())
}

impl MnfStats {
    pub fn new(nb: usize, noise: &Noise) -> Self {
        MnfStats {
            nb,
            n: 0,
            mean: vec![0.0; nb],
            m2: vec![0.0; nb * nb],
            pairs: noise
                .directions()
                .into_iter()
                .map(|(dy, dx)| DiffStats {
                    dy,
                    dx,
                    m: 0,
                    mean: vec![0.0; nb],
                    m2: vec![0.0; nb * nb],
                })
                .collect(),
            band_valid: vec![0; nb],
            given: match noise {
                Noise::Given(g) => Some(g.clone()),
                _ => None,
            },
        }
    }

    /// Rows `row0 .. row0 + rows` of date `t` (planar `nb x rows x nx`), plus the halo row below (planar
    /// `nb x nx`, `None` for the last row of the image); the pairs are anchored on the rows of the block only, so
    /// that blocks that follow one another count every pair once. Blocks may be added in any order, but the result
    /// is bit-identical only for the same order.
    #[allow(clippy::too_many_arguments)]
    pub fn add_block<T: Sample64>(
        &mut self,
        x: &[T],
        halo: Option<&[T]>,
        t: usize,
        row0: usize,
        rows: usize,
        ny: usize,
        nx: usize,
        opts: &StatsOptions,
    ) {
        assert_eq!(x.len(), self.nb * rows * nx, "x must be nb x rows x nx");
        if let Some(h) = halo {
            assert_eq!(h.len(), self.nb * nx, "the halo must be nb x nx");
        }
        let row = |b: usize, r: usize| -> &[T] {
            if r < rows {
                &x[(b * rows + r) * nx..][..nx]
            } else {
                &halo.expect("halo row")[b * nx..][..nx]
            }
        };
        let part = block_stats(&row, halo.is_some(), t, row0, rows, ny, nx, &self.empty_like(), opts);
        self.merge(&part);
    }

    /// Statistics with nothing in them, of the same bands and directions.
    pub fn empty_like(&self) -> MnfStats {
        let mut out = self.clone();
        out.n = 0;
        out.mean.iter_mut().for_each(|v| *v = 0.0);
        out.m2.iter_mut().for_each(|v| *v = 0.0);
        out.band_valid.iter_mut().for_each(|v| *v = 0);
        for d in out.pairs.iter_mut() {
            d.m = 0;
            d.mean.iter_mut().for_each(|v| *v = 0.0);
            d.m2.iter_mut().for_each(|v| *v = 0.0);
        }
        out
    }

    /// Adds the statistics of another part (after this one: Chan's update in that order).
    pub fn merge(&mut self, other: &MnfStats) {
        assert_eq!(self.nb, other.nb);
        assert_eq!(self.pairs.len(), other.pairs.len());
        chan(self.n, &mut self.mean, &mut self.m2, other.n, &other.mean, &other.m2);
        self.n += other.n;
        for (a, b) in self.pairs.iter_mut().zip(&other.pairs) {
            chan(a.m, &mut a.mean, &mut a.m2, b.m, &b.mean, &b.m2);
            a.m += b.m;
        }
        for (a, b) in self.band_valid.iter_mut().zip(&other.band_valid) {
            *a += b;
        }
    }
}

fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E3779B97F4A7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
    z ^ (z >> 31)
}

/// Whether the pixel with this key is in the sample: a function of the pixel and the seed only.
fn sampled(seed_hash: u64, fraction: f64, key: u64) -> bool {
    if fraction >= 1.0 {
        return true;
    }
    let h = splitmix64(key ^ seed_hash);
    ((h >> 11) as f64) * (1.0 / 9007199254740992.0) < fraction
}

/// Mean and centred co-moment of `nb` columns of `n` values (`data[b * n + i]`, centred in place).
fn moments(data: &mut [f64], nb: usize, n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut mean = vec![0.0; nb];
    let mut m2 = vec![0.0; nb * nb];
    if n == 0 {
        return (mean, m2);
    }
    for b in 0..nb {
        let col = &mut data[b * n..][..n];
        let m = col.iter().sum::<f64>() / n as f64;
        mean[b] = m;
        col.iter_mut().for_each(|v| *v -= m);
    }
    // the cross-products by the product `X^T X` of the centred columns (faer's kernels, one thread: the same bits
    // whatever the pool), the upper triangle mirrored so that the matrix is exactly symmetric
    let x = MatRef::from_column_major_slice(data, n, nb);
    let mut g = vec![0.0; nb * nb];
    matmul(
        MatMut::from_column_major_slice_mut(&mut g, nb, nb),
        Accum::Replace,
        x.transpose(),
        x,
        1.0,
        Par::Seq,
    );
    for i in 0..nb {
        for j in i..nb {
            let v = g[j * nb + i];
            m2[i * nb + j] = v;
            m2[j * nb + i] = v;
        }
    }
    (mean, m2)
}

/// The statistics of one block: `row(band, r)` is row `r` of the block (`r == rows` is the halo).
#[allow(clippy::too_many_arguments)]
fn block_stats<'a, T: Sample64>(
    row: &dyn Fn(usize, usize) -> &'a [T],
    has_halo: bool,
    t: usize,
    row0: usize,
    rows: usize,
    ny: usize,
    nx: usize,
    template: &MnfStats,
    opts: &StatsOptions,
) -> MnfStats {
    let nb = template.nb;
    let mut out = template.empty_like();
    let rr = rows + usize::from(has_halo);
    let np = rr * nx;
    let anchors = rows * nx;
    let active = |b: usize| opts.active.is_none_or(|a| a[b] != 0);

    let mut inmask = vec![1u8; np];
    if let Some(m) = opts.mask {
        for r in 0..rr {
            for c in 0..nx {
                inmask[r * nx + c] = u8::from(m[(row0 + r) * nx + c] != 0);
            }
        }
    }
    let mut valid = inmask.clone();
    let mut vals = vec![0.0f64; nb * np];
    for b in 0..nb {
        let act = active(b);
        for r in 0..rr {
            let src = row(b, r);
            let dst = &mut vals[b * np + r * nx..][..nx];
            for c in 0..nx {
                let v = src[c].to_f64();
                let ok = v.is_finite() && !opts.missing.contains(&v);
                let p = r * nx + c;
                if r < rows && inmask[p] != 0 && ok {
                    out.band_valid[b] += 1;
                }
                if act {
                    dst[c] = v;
                    if !ok {
                        valid[p] = 0;
                    }
                } else {
                    dst[c] = 0.0;
                }
            }
        }
    }
    let seed_hash = splitmix64(opts.seed);
    let samp: Vec<bool> = (0..anchors)
        .map(|p| {
            let (y, x) = (row0 + p / nx, p % nx);
            let key = (((opts.date0 + t) * ny + y) * nx + x) as u64;
            sampled(seed_hash, opts.sample_fraction, key)
        })
        .collect();

    // the signal statistics: valid and sampled pixels of the block's rows
    let idx: Vec<usize> = (0..anchors).filter(|&p| valid[p] != 0 && samp[p]).collect();
    let n = idx.len();
    let mut data = vec![0.0f64; nb * n];
    for b in 0..nb {
        let col = &vals[b * np..][..np];
        for (k, &p) in idx.iter().enumerate() {
            data[b * n + k] = col[p];
        }
    }
    let (mean, m2) = moments(&mut data, nb, n);
    out.n = n as u64;
    out.mean = mean;
    out.m2 = m2;

    // the pairs of each direction anchored on the block's rows
    for ds in out.pairs.iter_mut() {
        let (dy, dx) = (ds.dy as isize, ds.dx as isize);
        let mut list: Vec<(usize, usize)> = Vec::new();
        for r in 0..rows {
            let rq = r as isize + dy;
            if rq < 0 || rq as usize >= rr {
                continue;
            }
            for c in 0..nx {
                let cq = c as isize + dx;
                if cq < 0 || cq as usize >= nx {
                    continue;
                }
                let (p, q) = (r * nx + c, rq as usize * nx + cq as usize);
                if valid[p] == 0 || !samp[p] || valid[q] == 0 {
                    continue;
                }
                if let Some(reg) = opts.noise_region {
                    if reg[(row0 + r) * nx + c] == 0 || reg[(row0 + rq as usize) * nx + cq as usize] == 0 {
                        continue;
                    }
                }
                list.push((p, q));
            }
        }
        let m = list.len();
        let mut d = vec![0.0f64; nb * m];
        for b in 0..nb {
            let col = &vals[b * np..][..np];
            for (k, &(p, q)) in list.iter().enumerate() {
                d[b * m + k] = col[p] - col[q];
            }
        }
        let (mean, m2) = moments(&mut d, nb, m);
        ds.m = m as u64;
        ds.mean = mean;
        ds.m2 = m2;
    }
    out
}

/// Rows of a block: a function of the image only (never of the threads).
pub fn rows_per_block(nx: usize, nb: usize) -> usize {
    (BLOCK_BYTES / (nx.max(1) * nb.max(1) * 8)).max(2)
}

/// The statistics of a whole in-memory cube `(nt, nb, ny, nx)`: blocks of rows, the threads of `n_jobs`, merged in
/// block order. Bands with no valid value at all are left out of the validity of the pixels (when some other band
/// has valid values): their `band_valid` is 0 in the result.
pub fn mnf_stats_batch<T: Sample64>(
    x: &[T],
    nt: usize,
    nb: usize,
    ny: usize,
    nx: usize,
    opts: &StatsOptions,
    n_jobs: i32,
) -> Result<MnfStats, MnfError> {
    mnf_stats_blocks(x, nt, nb, ny, nx, opts, rows_per_block(nx, nb), n_jobs)
}

/// [`mnf_stats_batch`] with the rows of a block given (for the tests of the blocks).
#[allow(clippy::too_many_arguments)]
pub fn mnf_stats_blocks<T: Sample64>(
    x: &[T],
    nt: usize,
    nb: usize,
    ny: usize,
    nx: usize,
    opts: &StatsOptions,
    rows_per_block: usize,
    n_jobs: i32,
) -> Result<MnfStats, MnfError> {
    if x.len() != nt * nb * ny * nx {
        return Err(MnfError::BadInput(format!(
            "the cube has {} values, not {nt} x {nb} x {ny} x {nx}",
            x.len()
        )));
    }
    if !(opts.sample_fraction > 0.0 && opts.sample_fraction <= 1.0) {
        return Err(MnfError::BadInput("sample_fraction must be in (0, 1]".into()));
    }
    if let Noise::Given(g) = &opts.noise {
        check_given(g, nb)?;
    }
    for (name, m) in [("mask", opts.mask), ("noise_region", opts.noise_region)] {
        if m.is_some_and(|m| m.len() != ny * nx) {
            return Err(MnfError::BadInput(format!("{name} must be ny x nx")));
        }
    }
    let first = gather(x, nt, nb, ny, nx, opts, rows_per_block, n_jobs);
    let empty: Vec<u8> = first.band_valid.iter().map(|&v| u8::from(v > 0)).collect();
    let some_empty = empty.contains(&0);
    if first.n == 0 && some_empty && empty.contains(&1) {
        let mut again = StatsOptions {
            noise: opts.noise.clone(),
            mask: opts.mask,
            noise_region: opts.noise_region,
            missing: opts.missing,
            sample_fraction: opts.sample_fraction,
            seed: opts.seed,
            date0: opts.date0,
            active: None,
        };
        let both: Vec<u8> = match opts.active {
            Some(a) => a.iter().zip(&empty).map(|(&p, &q)| p & q).collect(),
            None => empty,
        };
        again.active = Some(&both);
        let mut s = gather(x, nt, nb, ny, nx, &again, rows_per_block, n_jobs);
        s.band_valid = first.band_valid;
        return Ok(s);
    }
    Ok(first)
}

#[allow(clippy::too_many_arguments)]
fn gather<T: Sample64>(
    x: &[T],
    nt: usize,
    nb: usize,
    ny: usize,
    nx: usize,
    opts: &StatsOptions,
    rows_per_block: usize,
    n_jobs: i32,
) -> MnfStats {
    let rpb = rows_per_block.clamp(1, ny.max(1));
    let blocks: Vec<(usize, usize, usize)> = (0..nt)
        .flat_map(|t| (0..ny).step_by(rpb).map(move |r0| (t, r0, rpb.min(ny - r0))))
        .collect();
    let template = MnfStats::new(nb, &opts.noise);
    let one = |&(t, r0, rows): &(usize, usize, usize)| -> MnfStats {
        let row = |b: usize, r: usize| -> &[T] { &x[((t * nb + b) * ny + r0 + r) * nx..][..nx] };
        block_stats(&row, r0 + rows < ny, t, r0, rows, ny, nx, &template, opts)
    };
    let mut total = template.clone();
    let th = threads(n_jobs).min(blocks.len().max(1));
    for wave in blocks.chunks(WAVE) {
        let parts: Vec<MnfStats> = if th <= 1 {
            wave.iter().map(one).collect()
        } else {
            pool(th).install(|| wave.par_iter().map(one).collect())
        };
        for p in &parts {
            total.merge(p);
        }
    }
    total
}

// ---------------------------------------------------------------------------
// The generalized eigenproblem

/// The eigenvalues (descending, ties in the solver's order) and eigenvectors (column `i` of the row-major `n x n`
/// result) of the symmetric `a` (row-major, symmetrized by the caller).
pub fn sym_eig_desc(a: &[f64], n: usize) -> Result<(Vec<f64>, Vec<f64>), MnfError> {
    pool(1).install(|| {
        let e = MatRef::from_column_major_slice(a, n, n)
            .self_adjoint_eigen(Side::Lower)
            .map_err(|e| MnfError::Eigen(format!("{e:?}")))?;
        let s = e.S().column_vector();
        let u = e.U();
        let mut vals = vec![0.0; n];
        let mut vecs = vec![0.0; n * n];
        for c in 0..n {
            let src = n - 1 - c; // faer's are ascending
            vals[c] = s[src];
            for r in 0..n {
                vecs[r * n + c] = u[(r, src)];
            }
        }
        Ok((vals, vecs))
    })
}

/// `a (ar x ac) * b (ac x bc)`, row-major.
fn mat_mul(a: &[f64], ar: usize, ac: usize, b: &[f64], bc: usize) -> Vec<f64> {
    let mut out = vec![0.0; ar * bc];
    for i in 0..ar {
        for k in 0..ac {
            let v = a[i * ac + k];
            if v != 0.0 {
                let (orow, brow) = (&mut out[i * bc..][..bc], &b[k * bc..][..bc]);
                for j in 0..bc {
                    orow[j] += v * brow[j];
                }
            }
        }
    }
    out
}

fn transpose(a: &[f64], r: usize, c: usize) -> Vec<f64> {
    let mut out = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            out[j * r + i] = a[i * c + j];
        }
    }
    out
}

/// Why a band is not in the eigenproblem.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dropped {
    /// Its variance is zero (or not a number).
    Constant,
    /// It has no valid value.
    Empty,
}

/// A fitted transform; the matrices are row-major and in the units of the bands (the `"noise"` scale).
#[derive(Clone, Debug)]
pub struct MnfTransform {
    pub nb: usize,
    /// The components, `kept.len()`.
    pub m: usize,
    pub kept: Vec<usize>,
    pub dropped: Vec<(usize, Dropped)>,
    pub mean: Vec<f64>,
    pub cov: Vec<f64>,
    pub noise_cov: Vec<f64>,
    /// Descending: `1 + SNR`.
    pub eigenvalues: Vec<f64>,
    /// `m x nb`: row `i` is `a_i`, zero at the dropped bands.
    pub forward: Vec<f64>,
    /// `nb x m`: column `i` is `p_i`, the pattern of component `i`.
    pub inverse: Vec<f64>,
    /// `nb x m`: the correlation of band `j` with component `i`.
    pub correlation: Vec<f64>,
    pub n_pixels: u64,
    pub n_pairs: Vec<u64>,
    /// The noise eigenvalues that the floor lifted.
    pub noise_floored: usize,
    /// Per component: it lies mostly in the directions the floor lifted.
    pub floored: Vec<bool>,
}

impl MnfTransform {
    /// `eigenvalue - 1` of each component.
    pub fn snr(&self) -> Vec<f64> {
        self.eigenvalues.iter().map(|l| l - 1.0).collect()
    }

    /// The components whose SNR is at least `snr`.
    pub fn n_with_snr(&self, snr: f64) -> usize {
        self.eigenvalues.iter().filter(|&&l| l - 1.0 >= snr).count()
    }

    pub fn is_degenerate(&self, i: usize) -> bool {
        self.eigenvalues[i] < DEGENERATE
    }
}

/// The transform of the statistics (see the module). Needs more valid pixels, and more pairs of each direction,
/// than bands.
pub fn mnf_fit(stats: &MnfStats) -> Result<MnfTransform, MnfError> {
    let nb = stats.nb;
    if stats.n == 0 {
        return Err(MnfError::NoValidPixel);
    }
    let empty: Vec<bool> = stats.band_valid.iter().map(|&v| v == 0).collect();
    let used = empty.iter().filter(|&&e| !e).count() as u64;
    if stats.n <= used {
        return Err(MnfError::TooFewPixels {
            n: stats.n,
            need: used + 1,
        });
    }
    for d in &stats.pairs {
        if d.m <= used {
            return Err(MnfError::TooFewPairs {
                direction: direction_name(d.dy, d.dx),
                m: d.m,
                need: used + 1,
            });
        }
    }
    let n1 = (stats.n - 1) as f64;
    let cov: Vec<f64> = stats.m2.iter().map(|v| v / n1).collect();
    let noise_cov: Vec<f64> = match &stats.given {
        Some(g) => {
            check_given(g, nb)?;
            g.clone()
        }
        None => {
            let total: u64 = stats.pairs.iter().map(|d| d.m).sum();
            let denom = 2.0 * (total - stats.pairs.len() as u64) as f64;
            let mut sum = vec![0.0; nb * nb];
            for d in &stats.pairs {
                sum.iter_mut().zip(&d.m2).for_each(|(a, b)| *a += b);
            }
            sum.iter().map(|v| v / denom).collect()
        }
    };
    let mut tr = fit_core(&stats.mean, &cov, &noise_cov, nb, &empty)?;
    tr.n_pixels = stats.n;
    tr.n_pairs = stats.pairs.iter().map(|d| d.m).collect();
    Ok(tr)
}

/// The transform of given covariances (`cov` the total covariance, `noise_cov` the noise's, `nb x nb`).
pub fn mnf_fit_cov(mean: &[f64], cov: &[f64], noise_cov: &[f64], nb: usize) -> Result<MnfTransform, MnfError> {
    if mean.len() != nb || cov.len() != nb * nb {
        return Err(MnfError::BadInput(format!(
            "mean must have {nb} values and cov {nb} x {nb}"
        )));
    }
    check_given(noise_cov, nb)?;
    fit_core(mean, cov, noise_cov, nb, &vec![false; nb])
}

fn fit_core(mean: &[f64], cov: &[f64], noise_cov: &[f64], nb: usize, empty: &[bool]) -> Result<MnfTransform, MnfError> {
    let mut kept = Vec::new();
    let mut dropped = Vec::new();
    for j in 0..nb {
        if empty[j] {
            dropped.push((j, Dropped::Empty));
        } else if !(cov[j * nb + j] > 0.0 && cov[j * nb + j].is_finite()) {
            dropped.push((j, Dropped::Constant));
        } else {
            kept.push(j);
        }
    }
    let b = kept.len();
    if b == 0 {
        return Err(MnfError::AllBandsConstant);
    }
    // pre-scaling: unit diagonal of the total covariance
    let s: Vec<f64> = kept.iter().map(|&j| 1.0 / cov[j * nb + j].sqrt()).collect();
    let mut sig = vec![0.0; b * b];
    let mut sn = vec![0.0; b * b];
    for a in 0..b {
        for c in 0..b {
            let (ja, jc) = (kept[a], kept[c]);
            sig[a * b + c] = 0.5 * s[a] * s[c] * (cov[ja * nb + jc] + cov[jc * nb + ja]);
            sn[a * b + c] = 0.5 * s[a] * s[c] * (noise_cov[ja * nb + jc] + noise_cov[jc * nb + ja]);
        }
    }
    // noise whitening, with a floor under the noise eigenvalues
    let (mut d, e) = sym_eig_desc(&sn, b)?;
    let dmax = d[0];
    if !(dmax > 0.0 && dmax.is_finite()) {
        return Err(MnfError::NoNoise);
    }
    let floor = EPS_NOISE * dmax;
    let floored_dir: Vec<bool> = d.iter().map(|&v| v < floor).collect();
    let noise_floored = floored_dir.iter().filter(|&&f| f).count();
    d.iter_mut().for_each(|v| *v = v.max(floor));
    let mut w = e.clone(); // E D^{-1/2}
    let mut v = e; // E D^{+1/2}
    for a in 0..b {
        for k in 0..b {
            w[a * b + k] /= d[k].sqrt();
            v[a * b + k] *= d[k].sqrt();
        }
    }
    // second rotation: PCA of the whitened total covariance
    let mut c = mat_mul(&transpose(&w, b, b), b, b, &mat_mul(&sig, b, b, &w, b), b);
    for a in 0..b {
        for k in 0..a {
            let m = 0.5 * (c[a * b + k] + c[k * b + a]);
            c[a * b + k] = m;
            c[k * b + a] = m;
        }
    }
    let (lam, u) = sym_eig_desc(&c, b)?;
    let wu = mat_mul(&w, b, b, &u, b);
    let vu = mat_mul(&v, b, b, &u, b);
    let mut a_mat = vec![0.0; b * b]; // columns a_i, band units
    let mut p_mat = vec![0.0; b * b];
    let mut r_mat = vec![0.0; b * b];
    for j in 0..b {
        for i in 0..b {
            a_mat[j * b + i] = s[j] * wu[j * b + i];
            p_mat[j * b + i] = vu[j * b + i] / s[j];
        }
    }
    let mut floored = vec![false; b];
    for i in 0..b {
        let sl = lam[i].max(0.0).sqrt();
        for j in 0..b {
            r_mat[j * b + i] = sl * s[j] * p_mat[j * b + i];
        }
        // the sign: positive correlation with the band it is most correlated with
        let mx = (0..b).fold(0.0f64, |m, j| m.max(r_mat[j * b + i].abs()));
        if mx > 0.0 {
            let jstar = (0..b)
                .find(|&j| r_mat[j * b + i].abs() >= mx * (1.0 - 1e-12))
                .unwrap_or(0);
            if r_mat[jstar * b + i] < 0.0 {
                for j in 0..b {
                    a_mat[j * b + i] = -a_mat[j * b + i];
                    p_mat[j * b + i] = -p_mat[j * b + i];
                    r_mat[j * b + i] = -r_mat[j * b + i];
                }
            }
        }
        let share: f64 = (0..b)
            .filter(|&k| floored_dir[k])
            .map(|k| u[k * b + i] * u[k * b + i])
            .sum();
        floored[i] = share > 0.5;
    }
    let mut forward = vec![0.0; b * nb];
    let mut inverse = vec![0.0; nb * b];
    let mut correlation = vec![0.0; nb * b];
    for (jj, &j) in kept.iter().enumerate() {
        for i in 0..b {
            forward[i * nb + j] = a_mat[jj * b + i];
            inverse[j * b + i] = p_mat[jj * b + i];
            correlation[j * b + i] = r_mat[jj * b + i];
        }
    }
    Ok(MnfTransform {
        nb,
        m: b,
        kept,
        dropped,
        mean: mean.to_vec(),
        cov: cov.to_vec(),
        noise_cov: noise_cov.to_vec(),
        eigenvalues: lam,
        forward,
        inverse,
        correlation,
        n_pixels: 0,
        n_pairs: Vec::new(),
        noise_floored,
        floored,
    })
}

// ---------------------------------------------------------------------------
// The transforms of the pixels

/// The scale of the components.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    /// Noise of unit variance, a variance of `lambda`.
    Noise,
    /// Unit variance (the degenerate components 0).
    Unit,
}

struct Ctx<'a, T> {
    x: &'a [T],
    nb: usize,
    npix: usize,
    tr: &'a MnfTransform,
    missing: &'a [f64],
    mask: Option<&'a [u8]>,
    /// The bands whose values decide the validity of a pixel: all but the empty ones.
    checked: Vec<usize>,
    /// The position among the kept bands of each band.
    pos: Vec<Option<usize>>,
}

struct Scratch {
    c: Vec<f64>,
    valid: Vec<u8>,
    acc: Vec<f64>,
    y: Vec<f64>,
}

impl<'a, T: Sample64> Ctx<'a, T> {
    fn new(
        x: &'a [T],
        nb: usize,
        npix: usize,
        tr: &'a MnfTransform,
        missing: &'a [f64],
        mask: Option<&'a [u8]>,
    ) -> Self {
        assert_eq!(nb, tr.nb, "the bands of the data are not those of the transform");
        let mut pos = vec![None; nb];
        for (i, &j) in tr.kept.iter().enumerate() {
            pos[j] = Some(i);
        }
        let checked = (0..nb)
            .filter(|&j| !tr.dropped.iter().any(|&(d, r)| d == j && r == Dropped::Empty))
            .collect();
        Ctx {
            x,
            nb,
            npix,
            tr,
            missing,
            mask,
            checked,
            pos,
        }
    }

    fn scratch(&self, k: usize) -> Scratch {
        Scratch {
            c: vec![0.0; self.tr.kept.len() * TILE],
            valid: vec![0; TILE],
            acc: vec![0.0; TILE],
            y: vec![0.0; k.max(1) * TILE],
        }
    }

    /// The centred kept bands of date `t`, pixels `p0..p0 + n`, and which pixels are valid.
    fn centre(&self, t: usize, p0: usize, n: usize, s: &mut Scratch) {
        for i in 0..n {
            s.valid[i] = self.mask.map_or(1, |m| u8::from(m[p0 + i] != 0));
        }
        for &band in &self.checked {
            let from = &self.x[(t * self.nb + band) * self.npix + p0..][..n];
            let mu = self.tr.mean[band];
            match self.pos[band] {
                Some(jj) => {
                    let to = &mut s.c[jj * TILE..][..n];
                    for i in 0..n {
                        let v = from[i].to_f64();
                        if !v.is_finite() || self.missing.contains(&v) {
                            s.valid[i] = 0;
                        }
                        to[i] = v - mu;
                    }
                }
                None => {
                    for i in 0..n {
                        let v = from[i].to_f64();
                        if !v.is_finite() || self.missing.contains(&v) {
                            s.valid[i] = 0;
                        }
                    }
                }
            }
        }
        for jj in 0..self.tr.kept.len() {
            let c = &mut s.c[jj * TILE..][..n];
            for i in 0..n {
                if s.valid[i] == 0 {
                    c[i] = 0.0;
                }
            }
        }
    }

    /// The first `k` components (noise scale) of the centred tile into `s.y`.
    fn components(&self, n: usize, k: usize, s: &mut Scratch) {
        let nb = self.nb;
        for i in 0..k {
            let y = &mut s.y[i * TILE..][..n];
            y.iter_mut().for_each(|v| *v = 0.0);
            for (jj, &j) in self.tr.kept.iter().enumerate() {
                let a = self.tr.forward[i * nb + j];
                let c = &s.c[jj * TILE..][..n];
                for p in 0..n {
                    y[p] += a * c[p];
                }
            }
        }
    }
}

/// The factor of each of the first `k` components that gives the scale.
fn scale_factors(tr: &MnfTransform, k: usize, scale: Scale) -> Vec<f64> {
    (0..k)
        .map(|i| match scale {
            Scale::Noise => 1.0,
            Scale::Unit if tr.eigenvalues[i] >= DEGENERATE => 1.0 / tr.eigenvalues[i].sqrt(),
            Scale::Unit => 0.0,
        })
        .collect()
}

/// Runs `job(scratch, date, p0, p1)` over the tiles of every date.
fn run_tiles<S: Send>(
    nt: usize,
    npix: usize,
    n_jobs: i32,
    init: impl Fn() -> S + Sync + Send,
    job: impl Fn(&mut S, usize, usize, usize) + Sync + Send,
) {
    if nt == 0 || npix == 0 {
        return;
    }
    let per_date = npix.div_ceil(TILE);
    let jobs = nt * per_date;
    let run = |s: &mut S, j: usize| {
        let (k, t) = (j / per_date, j % per_date);
        job(s, k, t * TILE, ((t + 1) * TILE).min(npix));
    };
    let th = threads(n_jobs).min(jobs);
    if th <= 1 {
        let mut s = init();
        (0..jobs).for_each(|j| run(&mut s, j));
    } else {
        pool(th).install(|| {
            (0..jobs).into_par_iter().for_each_init(&init, |s, j| run(s, j));
        });
    }
}

/// The components of `x` `(nt, nb, npix)`: `(nt, k_out, npix)` f32 into `out`, NaN at the invalid pixels (a band
/// missing, or outside the `mask`). Every value of `out` is written.
#[allow(clippy::too_many_arguments)]
pub fn mnf_forward_batch<T: Sample64>(
    x: &[T],
    nb: usize,
    npix: usize,
    tr: &MnfTransform,
    k_out: usize,
    scale: Scale,
    missing: &[f64],
    mask: Option<&[u8]>,
    out: &mut [f32],
    n_jobs: i32,
) {
    assert!(k_out <= tr.m, "k_out is more than the components");
    let nt = x.len().checked_div(nb * npix).unwrap_or(0);
    assert_eq!(x.len(), nt * nb * npix);
    assert_eq!(out.len(), nt * k_out * npix);
    let ctx = Ctx::new(x, nb, npix, tr, missing, mask);
    let f = scale_factors(tr, k_out, scale);
    let shared = Shared::new(out);
    run_tiles(
        nt,
        npix,
        n_jobs,
        || ctx.scratch(0),
        |s, t, p0, p1| {
            let n = p1 - p0;
            ctx.centre(t, p0, n, s);
            for i in 0..k_out {
                let acc = &mut s.acc[..n];
                acc.iter_mut().for_each(|v| *v = 0.0);
                for (jj, &j) in tr.kept.iter().enumerate() {
                    let a = tr.forward[i * nb + j];
                    let c = &s.c[jj * TILE..][..n];
                    for p in 0..n {
                        acc[p] += a * c[p];
                    }
                }
                // SAFETY: each job writes its own date and pixels of the planes.
                let o = unsafe { shared.slice_mut(((t * k_out + i) * npix + p0)..((t * k_out + i) * npix + p1)) };
                for p in 0..n {
                    o[p] = if s.valid[p] != 0 {
                        (acc[p] * f[i]) as f32
                    } else {
                        f32::NAN
                    };
                }
            }
        },
    );
}

/// The bands from components `y` `(nt, k_in, npix)` f32 (in `scale`): the first `k <= k_in` of them,
/// `mean + sum p_i y_i`, into `out` `(nt, nb, npix)`; NaN where a component is NaN, and in the empty bands.
#[allow(clippy::too_many_arguments)]
pub fn mnf_inverse_batch(
    y: &[f32],
    k_in: usize,
    npix: usize,
    tr: &MnfTransform,
    k: usize,
    scale: Scale,
    out: &mut [f32],
    n_jobs: i32,
) {
    assert!(k <= k_in && k <= tr.m, "k is more than the components");
    let nb = tr.nb;
    let nt = y.len().checked_div(k_in * npix).unwrap_or(0);
    assert_eq!(y.len(), nt * k_in * npix);
    assert_eq!(out.len(), nt * nb * npix);
    // y in the noise scale = y * sqrt(lambda) for a unit input
    let f: Vec<f64> = (0..k)
        .map(|i| match scale {
            Scale::Noise => 1.0,
            Scale::Unit => tr.eigenvalues[i].max(0.0).sqrt(),
        })
        .collect();
    let empty: Vec<bool> = {
        let mut e = vec![false; nb];
        for &(j, r) in &tr.dropped {
            e[j] = r == Dropped::Empty;
        }
        e
    };
    let shared = Shared::new(out);
    run_tiles(
        nt,
        npix,
        n_jobs,
        || (vec![0.0f64; TILE], vec![0u8; TILE]),
        |(acc, valid), t, p0, p1| {
            let n = p1 - p0;
            for p in 0..n {
                valid[p] = 1;
            }
            for i in 0..k_in.min(k.max(1)) {
                let from = &y[(t * k_in + i) * npix + p0..][..n];
                for p in 0..n {
                    if from[p].is_nan() {
                        valid[p] = 0;
                    }
                }
            }
            for j in 0..nb {
                // SAFETY: each job writes its own date and pixels of the planes.
                let o = unsafe { shared.slice_mut(((t * nb + j) * npix + p0)..((t * nb + j) * npix + p1)) };
                if empty[j] {
                    o.iter_mut().for_each(|v| *v = f32::NAN);
                    continue;
                }
                acc[..n].iter_mut().for_each(|v| *v = 0.0);
                for i in 0..k {
                    let pij = tr.inverse[j * tr.m + i] * f[i];
                    let from = &y[(t * k_in + i) * npix + p0..][..n];
                    for p in 0..n {
                        let v = if valid[p] != 0 { f64::from(from[p]) } else { 0.0 };
                        acc[p] += pij * v;
                    }
                }
                for p in 0..n {
                    o[p] = if valid[p] != 0 {
                        (tr.mean[j] + acc[p]) as f32
                    } else {
                        f32::NAN
                    };
                }
            }
        },
    );
}

/// Forward then inverse with the first `k` components, without keeping them: `(nt, nb, npix)` into `out`.
#[allow(clippy::too_many_arguments)]
pub fn mnf_denoise_batch<T: Sample64>(
    x: &[T],
    nb: usize,
    npix: usize,
    tr: &MnfTransform,
    k: usize,
    missing: &[f64],
    mask: Option<&[u8]>,
    out: &mut [f32],
    n_jobs: i32,
) {
    assert!(k <= tr.m, "k is more than the components");
    let nt = x.len().checked_div(nb * npix).unwrap_or(0);
    assert_eq!(x.len(), nt * nb * npix);
    assert_eq!(out.len(), nt * nb * npix);
    let ctx = Ctx::new(x, nb, npix, tr, missing, mask);
    let empty: Vec<bool> = {
        let mut e = vec![false; nb];
        for &(j, r) in &tr.dropped {
            e[j] = r == Dropped::Empty;
        }
        e
    };
    let shared = Shared::new(out);
    run_tiles(
        nt,
        npix,
        n_jobs,
        || ctx.scratch(k),
        |s, t, p0, p1| {
            let n = p1 - p0;
            ctx.centre(t, p0, n, s);
            ctx.components(n, k, s);
            for j in 0..nb {
                // SAFETY: each job writes its own date and pixels of the planes.
                let o = unsafe { shared.slice_mut(((t * nb + j) * npix + p0)..((t * nb + j) * npix + p1)) };
                if empty[j] {
                    o.iter_mut().for_each(|v| *v = f32::NAN);
                    continue;
                }
                let acc = &mut s.acc[..n];
                acc.iter_mut().for_each(|v| *v = 0.0);
                for i in 0..k {
                    let pij = tr.inverse[j * tr.m + i];
                    let yi = &s.y[i * TILE..][..n];
                    for p in 0..n {
                        acc[p] += pij * yi[p];
                    }
                }
                for p in 0..n {
                    o[p] = if s.valid[p] != 0 {
                        (tr.mean[j] + acc[p]) as f32
                    } else {
                        f32::NAN
                    };
                }
            }
        },
    );
}

#[cfg(test)]
mod tests;
