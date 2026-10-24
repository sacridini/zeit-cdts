//! The reversible-jump sampler of one series (`beast2_main_corev4` of
//! `beastv2_COREV4.c`, for one pixel): the chains, each from the same
//! initial model, the proposals accepted or not on the marginal likelihood
//! and the model prior, the coefficients' precisions and the noise variance
//! resampled every 20 iterations, and the samples after the burn-in, every
//! `thin`-th, averaged into the posterior of the season and the trend.

use super::basis::{Basis, Move, Proposal, Seg};
use super::config::{Config, PrecType};
use super::cpt;
use super::kern::{
    chol_add_cols, cross, dot, fastexp, fastlog, fastsqrt, gemv, solve_lu_invdiag, solve_u_invdiag, sum_log_diag,
};
use super::rng::{self, Pcg};

const MAX_RAND_NUM: usize = 5000;
const MIN_PREC_VALUE: f64 = 0.001;
const MIN_SIG2_VALUE: f64 = 0.001;
const MIN_ALPHA2_VALUE: f64 = 0.0001;

/// The random numbers the sampler draws its decisions from, in blocks
/// (`RND32`, `RND16`, `RND08`, `RNDGAMMA`): the 16- and 8-bit ones are the
/// halves and bytes of 32-bit draws, as the C reads them through its
/// pointers.
struct Rnd {
    r32: Vec<u32>,
    i32: usize,
    r16: Vec<u16>,
    i16: usize,
    r08: Vec<u8>,
    i08: usize,
    gamma: Vec<f32>,
    ig: usize,
    gamma_a: f32,
    gamma_end: usize,
}

impl Rnd {
    fn new() -> Rnd {
        Rnd {
            r32: vec![0; MAX_RAND_NUM],
            i32: 0,
            r16: vec![0; MAX_RAND_NUM * 2],
            i16: 0,
            r08: vec![0; MAX_RAND_NUM * 4],
            i08: 0,
            gamma: vec![0.0; MAX_RAND_NUM],
            ig: 0,
            gamma_a: 0.0,
            gamma_end: 0,
        }
    }

    fn fill16(&mut self, rng: &mut Pcg, n32: usize) {
        for j in 0..n32 {
            let v = rng.next_u32();
            self.r16[2 * j] = v as u16;
            self.r16[2 * j + 1] = (v >> 16) as u16;
        }
    }

    fn fill08(&mut self, rng: &mut Pcg, n32: usize) {
        for j in 0..n32 {
            let v = rng.next_u32().to_le_bytes();
            self.r08[4 * j..4 * j + 4].copy_from_slice(&v);
        }
    }

    fn start(&mut self, rng: &mut Pcg, gamma_a: f32, n_prec_grp: usize) {
        rng.fill(&mut self.r32);
        self.fill16(rng, MAX_RAND_NUM);
        self.fill08(rng, MAX_RAND_NUM);
        rng::gamma(rng, &mut self.gamma, gamma_a);
        self.i32 = 0;
        self.i16 = 0;
        self.i08 = 0;
        self.ig = 0;
        self.gamma_a = gamma_a;
        self.gamma_end = MAX_RAND_NUM - n_prec_grp - 1;
    }

    /// The refills at the top of each iteration: the values used are drawn
    /// again, the rest kept.
    fn refill(&mut self, rng: &mut Pcg) {
        if self.i32 >= MAX_RAND_NUM - 7 {
            let n = self.i32;
            rng.fill(&mut self.r32[..n]);
            self.i32 = 0;
        }
        if self.i16 >= MAX_RAND_NUM * 2 - 7 {
            let n = (self.i16 * 2 + 3) / 4;
            self.fill16(rng, n);
            self.i16 = 0;
        }
        if self.i08 >= MAX_RAND_NUM * 4 - 7 - 3 {
            let n = (self.i08 + 3) / 4;
            self.fill08(rng, n);
            self.i08 = 0;
        }
        if self.ig >= self.gamma_end {
            rng::gamma(rng, &mut self.gamma, self.gamma_a);
            self.ig = 0;
        }
    }

    #[inline]
    fn u32(&mut self) -> u32 {
        let v = self.r32[self.i32];
        self.i32 += 1;
        v
    }

    #[inline]
    fn u16(&mut self) -> u16 {
        let v = self.r16[self.i16];
        self.i16 += 1;
        v
    }

    #[inline]
    fn u8(&mut self) -> u8 {
        let v = self.r08[self.i08];
        self.i08 += 1;
        v
    }

    #[inline]
    fn gamma(&mut self) -> f32 {
        let v = self.gamma[self.ig];
        self.ig += 1;
        v
    }
}

/// `RANDINT(x0, x1, seed)`: the seed is drawn only when the range has more
/// than one value.
#[inline]
fn randint(x0: i32, x1: i32, seed: impl FnOnce() -> u32) -> i32 {
    if x0 == x1 {
        x1
    } else {
        let range = (x1 - x0 + 1) as u32;
        if range == 0 {
            return x0;
        }
        x0 + (seed() % range) as i32
    }
}

/// The 1-based position of the `nth` one of `v` (`i08_find_nth_onebyte_binvec`).
fn find_nth_one(v: &[u8], nth: i32) -> i32 {
    let mut count = 0;
    for (i, &b) in v.iter().enumerate() {
        count += b as i32;
        if count == nth && b == 1 {
            return i as i32 + 1;
        }
    }
    -1
}

/// One model: its cross products, their Cholesky factor (with the
/// reciprocals on the diagonal), the posterior mean of the coefficients and
/// the marginal likelihood (`BEAST2_MODELDATA`).
struct ModelData {
    xtx: Vec<f32>,
    xty: Vec<f32>,
    chol: Vec<f32>,
    beta_mean: Vec<f32>,
    prec_diag: Vec<f32>,
    n_terms: [i32; 2],
    alpha2q: f32,
    marg_lik: f32,
    k: usize,
}

impl ModelData {
    fn new(kcap: usize) -> Self {
        ModelData {
            xtx: vec![0.0; kcap * kcap],
            xty: vec![0.0; kcap],
            chol: vec![0.0; kcap * kcap],
            beta_mean: vec![0.0; kcap],
            prec_diag: vec![0.0; kcap],
            n_terms: [0; 2],
            alpha2q: 0.0,
            marg_lik: 0.0,
            k: 0,
        }
    }
}

/// The k largest and k smallest values seen at each date, for the credible
/// intervals: the k-th largest is the upper end, the k-th smallest the
/// lower, as Rbeast's strips keep them (before k values the ends stay 0).
struct CiTails {
    n: usize,
    k: usize,
    inserted: usize,
    top: Vec<f32>,
    bottom: Vec<f32>,
    /// `upper[0..n]` then `lower[0..n]`.
    result: Vec<f32>,
}

fn sift_down(h: &mut [f32], mut i: usize) {
    let n = h.len();
    loop {
        let l = 2 * i + 1;
        if l >= n {
            break;
        }
        let r = l + 1;
        let m = if r < n && h[r] < h[l] { r } else { l };
        if h[m] < h[i] {
            h.swap(m, i);
            i = m;
        } else {
            break;
        }
    }
}

fn heapify(h: &mut [f32]) {
    for i in (0..h.len() / 2).rev() {
        sift_down(h, i);
    }
}

/// `i32_transpose_inplace_prev_two_ends`: the in-place transposition of a
/// column-major `nrow` by `ncol` matrix that Rbeast's credible intervals
/// use, cycle by cycle, with its early stop on a count of the values moved.
fn transpose_inplace_two_ends(m: &mut [f32], nrow: u64, ncol: u64) {
    const NWORK: u64 = 1000;
    if nrow == 1 || ncol == 1 {
        return;
    }
    let gcd = |mut a: i64, mut b: i64| loop {
        let r = a % b;
        a = b;
        b = r;
        if r == 0 {
            return a;
        }
    };
    let nfixed = (gcd(nrow as i64 - 1, ncol as i64 - 1) + 1) as u64;
    let mut nprocessed = nfixed;
    let mut worked = [false; NWORK as usize + 1];
    let k = nrow * ncol - 1;
    let prev_of = |s: u64| (s * nrow) % k;
    let next_of = |s: u64| (s * ncol) % k;
    let mut start = 1u64;
    while start < k {
        if start <= NWORK && worked[start as usize] {
            start += 1;
            continue;
        }
        if nprocessed > k {
            break;
        }
        let mut curr = start;
        let mut prev = prev_of(start);
        if prev <= start {
            start += 1;
            continue;
        }
        if start <= NWORK {
            let start_comp = k - start;
            let mut prev_comp = k - prev;
            let mut curr_comp = start_comp;
            worked[if curr_comp <= NWORK { curr_comp as usize } else { 0 }] = true;
            let tmp = m[start as usize];
            let tmp_comp = m[start_comp as usize];
            while prev != start {
                if prev == start_comp {
                    break;
                }
                worked[if prev <= NWORK { prev as usize } else { 0 }] = true;
                m[curr as usize] = m[prev as usize];
                nprocessed += 1;
                worked[if prev_comp <= NWORK { prev_comp as usize } else { 0 }] = true;
                m[curr_comp as usize] = m[prev_comp as usize];
                nprocessed += 1;
                curr = prev;
                prev = prev_of(curr);
                curr_comp = prev_comp;
                prev_comp = prev_of(curr_comp);
            }
            nprocessed += 1;
            m[curr as usize] = tmp;
            nprocessed += 1;
            m[curr_comp as usize] = tmp_comp;
        } else {
            let mut next = start;
            loop {
                next = next_of(next);
                if next <= start {
                    break;
                }
            }
            if next < start {
                start += 1;
                continue;
            }
            let tmp = m[start as usize];
            while prev != start {
                m[curr as usize] = m[prev as usize];
                curr = prev;
                prev = prev_of(curr);
                nprocessed += 1;
            }
            nprocessed += 1;
            m[curr as usize] = tmp;
        }
        start += 1;
    }
}

impl CiTails {
    fn new(n: usize, k: usize) -> Self {
        CiTails {
            n,
            k,
            inserted: 0,
            top: vec![0.0; n * k],
            bottom: vec![0.0; n * k],
            result: vec![0.0; 2 * n],
        }
    }

    fn reset(&mut self) {
        self.inserted = 0;
        self.result.fill(0.0);
    }

    fn insert(&mut self, row: &[f32]) {
        let (n, k) = (self.n, self.k);
        if self.inserted < k {
            // The first k rows side by side, then transposed in place as
            // Rbeast does, with its transposition's slips (a few dates start
            // from a few values of other dates).
            let s = self.inserted;
            self.top[s * n..(s + 1) * n].copy_from_slice(&row[..n]);
            self.inserted += 1;
            if self.inserted == k {
                transpose_inplace_two_ends(&mut self.top, n as u64, k as u64);
                for i in 0..n * k {
                    // The bottom tail is a min-heap of the negated values.
                    self.bottom[i] = -self.top[i];
                }
                for i in 0..n {
                    let t = &mut self.top[i * k..(i + 1) * k];
                    heapify(t);
                    self.result[i] = t[0];
                    let b = &mut self.bottom[i * k..(i + 1) * k];
                    heapify(b);
                    self.result[n + i] = -b[0];
                }
            }
            return;
        }
        let (upper, lower) = self.result.split_at_mut(n);
        for (i, ((&y, u), l)) in row[..n].iter().zip(upper.iter_mut()).zip(lower.iter_mut()).enumerate() {
            if y > *u {
                let t = &mut self.top[i * k..(i + 1) * k];
                t[0] = y;
                sift_down(t, 0);
                *u = t[0];
            }
            if y < *l {
                let b = &mut self.bottom[i * k..(i + 1) * k];
                b[0] = -y;
                sift_down(b, 0);
                *l = -b[0];
            }
        }
    }
}

/// One component's sums over a chain (`BEAST2_RESULT` of a chain).
#[derive(Default)]
struct Acc {
    ncp_pr: Vec<i32>,
    cp_occ: Vec<i32>,
    order: Vec<i32>,
    y: Vec<f32>,
    ysd: Vec<f32>,
}

impl Acc {
    fn new(n: usize, max_knot: usize) -> Self {
        Acc {
            ncp_pr: vec![0; max_knot + 1],
            cp_occ: vec![0; n],
            order: vec![0; n],
            y: vec![0.0; n],
            ysd: vec![0.0; n],
        }
    }

    fn reset(&mut self) {
        self.ncp_pr.fill(0);
        self.cp_occ.fill(0);
        self.order.fill(0);
        self.y.fill(0.0);
        self.ysd.fill(0.0);
    }
}

/// The posterior of one component.
#[derive(Clone, Debug, Default)]
pub struct Component {
    /// The mean number of changepoints, its most probable value, median
    /// and 10th and 90th percentiles.
    pub ncp: f32,
    pub ncp_mode: f32,
    pub ncp_median: f32,
    pub ncp_pct90: f32,
    pub ncp_pct10: f32,
    /// The probability of 0, 1, ... changepoints.
    pub ncp_pr: Vec<f32>,
    /// The probability of a changepoint at each date.
    pub cp_occ: Vec<f32>,
    /// The mean harmonic order (season) or polynomial degree (trend).
    pub order: Vec<f32>,
    /// The fitted component, its standard deviation and 95% interval
    /// (`ci[..n]` the upper end, `ci[n..]` the lower).
    pub y: Vec<f32>,
    pub sd: Vec<f32>,
    pub ci: Vec<f32>,
    /// The changepoints found, most probable first (NaN after the last):
    /// their times, probabilities, the jump of the component there and
    /// their 95% intervals (`cp_ci[..max]` lower, `cp_ci[max..]` upper).
    pub cp: Vec<f32>,
    pub cp_pr: Vec<f32>,
    pub cp_abrupt: Vec<f32>,
    pub cp_ci: Vec<f32>,
    /// The season's amplitude or the trend's slope, its standard
    /// deviation; for the trend, the probabilities of a positive and a zero
    /// slope and the slope's 95% interval.
    pub extra: Vec<f32>,
    pub extra_sd: Vec<f32>,
    pub slope_pos_pr: Vec<f32>,
    pub slope_zero_pr: Vec<f32>,
    pub slope_ci: Vec<f32>,
}

impl Component {
    fn zeros(n: usize, max_knot: usize, ci: bool, slope: bool, amp: bool) -> Self {
        Component {
            ncp_pr: vec![0.0; max_knot + 1],
            cp_occ: vec![0.0; n],
            order: vec![0.0; n],
            y: vec![0.0; n],
            sd: vec![0.0; n],
            ci: if ci { vec![0.0; 2 * n] } else { vec![] },
            extra: if slope || amp { vec![0.0; n] } else { vec![] },
            extra_sd: if slope || amp { vec![0.0; n] } else { vec![] },
            slope_pos_pr: if slope { vec![0.0; n] } else { vec![] },
            slope_zero_pr: if slope { vec![0.0; n] } else { vec![] },
            slope_ci: if slope && ci { vec![0.0; 2 * n] } else { vec![] },
            ..Default::default()
        }
    }
}

/// BEAST's decomposition of one series.
#[derive(Clone, Debug, Default)]
pub struct Fit {
    pub marg_lik: f32,
    pub sig2: f32,
    pub r2: f32,
    pub rmse: f32,
    pub season: Option<Component>,
    pub trend: Component,
    /// The series as fitted (aggregated), NaN where missing.
    pub data: Vec<f32>,
}

/// The scratch memory of one thread: the bases (their terms depend only on
/// the options and the length of the series), the design matrix and the
/// two models, the random numbers and the sums of a chain.
pub struct Workspace {
    cfg: Config,
    n: usize,
    npad16: usize,
    bases: Vec<Basis>,
    xt: Vec<f32>,
    xnew: Vec<f32>,
    backup: Vec<f32>,
    membuf: Vec<f32>,
    curr: ModelData,
    prop: ModelData,
    beta: Vec<f32>,
    prec: [f32; 2],
    log_prec: [f32; 2],
    sig2: f32,
    rnd: Rnd,
    y: Vec<f32>,
    missing: Vec<usize>,
    n_good: i32,
    yty: f32,
    alpha1_star: f32,
    deviation: Vec<f32>,
    avg_dev: f32,
    extreme: Vec<u8>,
    tmp_good: Vec<u8>,
    dev_next: u32,
    dev_shall_update: bool,
    bands: Vec<Seg>,
    acc: Vec<Acc>,
    amp: Vec<f32>,
    amp_sd: Vec<f32>,
    slp: Vec<f32>,
    slp_sd: Vec<f32>,
    slp_pos: Vec<i32>,
    slp_zero: Vec<i32>,
    ci: Vec<CiTails>,
}

struct Chain {
    marg_lik: f32,
    sig2: f32,
    samples: u32,
}

impl Workspace {
    pub fn new(cfg: &Config) -> Self {
        let n = cfg.n;
        let npad16 = n.div_ceil(16) * 16;
        let mut bases = Vec::new();
        if cfg.season {
            bases.push(Basis::new_season(
                cfg.sp,
                n,
                cfg.period,
                cfg.k_max,
                cfg.move_step,
                cfg.s_resample,
                cfg.season_btype,
            ));
        }
        bases.push(Basis::new_trend(
            cfg.tp,
            n,
            cfg.k_max,
            cfg.move_step,
            cfg.t_resample,
            cfg.trend_btype,
        ));
        let kcap = cfg.k_max as usize + cfg.max_new_terms() + 2;
        let nb = bases.len();
        let acc = bases.iter().map(|b| Acc::new(n, b.prior.max_knot as usize)).collect();
        let nci = nb + 1;
        let ci = if cfg.ci {
            (0..nci).map(|_| CiTails::new(n, cfg.ci_k)).collect()
        } else {
            vec![]
        };
        Workspace {
            n,
            npad16,
            xt: vec![0.0; n * kcap],
            xnew: vec![0.0; n * cfg.max_new_terms()],
            backup: vec![0.0; n * cfg.max_new_terms()],
            membuf: vec![0.0; 4 * n],
            curr: ModelData::new(kcap),
            prop: ModelData::new(kcap),
            beta: vec![0.0; kcap],
            prec: [cfg.prec_value; 2],
            log_prec: [rng::logf(cfg.prec_value); 2],
            sig2: cfg.sig2,
            rnd: Rnd::new(),
            y: vec![0.0; n],
            missing: Vec::with_capacity(n),
            n_good: 0,
            yty: 0.0,
            alpha1_star: 0.0,
            deviation: vec![0.0; n],
            avg_dev: 1.0,
            extreme: vec![0; npad16],
            tmp_good: vec![0; npad16],
            dev_next: 1,
            dev_shall_update: false,
            bands: Vec::new(),
            acc,
            amp: vec![0.0; n],
            amp_sd: vec![0.0; n],
            slp: vec![0.0; n],
            slp_sd: vec![0.0; n],
            slp_pos: vec![0; n],
            slp_zero: vec![0; n],
            ci,
            bases,
            cfg: cfg.clone(),
        }
    }

    fn n_prec_grp(&self) -> usize {
        match self.cfg.prec_type {
            PrecType::ComponentWise => self.bases.len(),
            _ => 1,
        }
    }

    fn update_kbase(&mut self) {
        self.bases[0].kbase = 0;
        if self.bases.len() == 2 {
            self.bases[1].kbase = self.bases[0].k;
        }
    }

    /// `SetPrecXtXDiag` of the current model.
    fn set_prec_diag_curr(&mut self) {
        if self.cfg.prec_type == PrecType::ComponentWise {
            let mut i = 0;
            for (id, b) in self.bases.iter().enumerate() {
                for _ in 0..b.k {
                    self.curr.prec_diag[i] = self.prec[id];
                    i += 1;
                }
            }
        }
    }

    fn chol_full(&mut self, prop: bool, k0: usize) {
        let m = if prop { &mut self.prop } else { &mut self.curr };
        let k = m.k;
        if k0 > k {
            return;
        }
        if self.cfg.prec_type == PrecType::ComponentWise {
            chol_add_cols(&m.xtx, &mut m.chol, &m.prec_diag, k, k0, k);
        } else {
            chol_add_cols(&m.xtx, &mut m.chol, &self.prec[..1], k, k0, k);
        }
    }

    /// `ComputeMargLik_prec01` / `_prec23`.
    fn marg_lik(&mut self, prop: bool) {
        let m = if prop { &mut self.prop } else { &mut self.curr };
        let k = m.k;
        solve_lu_invdiag(&m.chol, &m.xty, &mut m.beta_mean, k);
        let alpha2_star = ((self.yty - dot(&m.xty[..k], &m.beta_mean[..k])) as f64 * 0.5) as f32;
        let post = sum_log_diag(&m.chol, k);
        let prior = if self.cfg.prec_type == PrecType::ComponentWise {
            let mut p = 0f32;
            for i in 0..self.bases.len() {
                p += self.log_prec[i] * m.n_terms[i] as f32;
            }
            (p as f64 * -0.5) as f32
        } else {
            -0.5f32 * self.log_prec[0] * k as f32
        };
        m.marg_lik = post - prior - self.alpha1_star * fastlog(alpha2_star);
        m.alpha2q = if alpha2_star as f64 > MIN_ALPHA2_VALUE {
            alpha2_star
        } else {
            MIN_ALPHA2_VALUE as f32
        };
    }

    fn increase_prec(&mut self) {
        match self.cfg.prec_type {
            PrecType::Const => {}
            PrecType::Uniform => {
                self.prec[0] *= 2.0;
                self.log_prec[0] = rng::logf(self.prec[0]);
            }
            PrecType::ComponentWise => {
                for id in 0..self.bases.len() {
                    self.prec[id] = self.prec[id] + self.prec[id];
                    self.log_prec[id] = rng::logf(self.prec[id]);
                }
            }
        }
    }

    fn resample_prec(&mut self, rng: &mut Pcg) {
        let cfg = &self.cfg;
        let mut g = [0f32; 1];
        match cfg.prec_type {
            PrecType::Const => {}
            PrecType::Uniform => {
                let k = self.curr.k;
                let sumq = dot(&self.beta[..k], &self.beta[..k]);
                rng::gamma(rng, &mut g, cfg.delta1 + k as f32 * 0.5f32);
                let p = g[0] / (cfg.delta2 + 0.5f32 * sumq / self.sig2);
                self.prec[0] = if p as f64 > MIN_PREC_VALUE {
                    p
                } else {
                    MIN_PREC_VALUE as f32
                };
                self.log_prec[0] = rng::logf(self.prec[0]);
            }
            PrecType::ComponentWise => {
                for id in 0..self.bases.len() {
                    let (kb, k) = (self.bases[id].kbase as usize, self.bases[id].k as usize);
                    if k == 0 {
                        continue;
                    }
                    let b = &self.beta[kb..kb + k];
                    let sumq = dot(b, b);
                    rng::gamma(rng, &mut g, cfg.delta1 + k as f32 * 0.5f32);
                    let p = g[0] / (cfg.delta2 + 0.5f32 * sumq / self.sig2);
                    self.prec[id] = if p as f64 > MIN_PREC_VALUE {
                        p
                    } else {
                        MIN_PREC_VALUE as f32
                    };
                    self.log_prec[id] = rng::logf(self.prec[id]);
                }
            }
        }
    }

    /// `BEAST2_EvaluateModel` of the current model: its terms into the
    /// design matrix, the cross products (the missing rows zeroed), the
    /// factor and the marginal likelihood.
    fn evaluate_curr(&mut self) {
        let n = self.n;
        let mut k = 0usize;
        for b in &self.bases {
            let order1 = if b.trend { 0 } else { 1 };
            for i in 1..=b.n_knot + 1 {
                let seg = Seg {
                    r1: b.knot(i - 2),
                    r2: b.knot(i - 1) - 1,
                    order1,
                    order2: b.order[(i - 1) as usize],
                    k: 0,
                };
                k += b.gen_terms(&mut self.xt[n * k..], n, &seg) as usize;
            }
        }
        let nm = self.missing.len();
        // Zero the missing rows (the C saves them, then restores them).
        let mut saved = vec![0f32; nm * k];
        for c in 0..k {
            for (j, &r) in self.missing.iter().enumerate() {
                saved[c * nm + j] = self.xt[c * n + r];
                self.xt[c * n + r] = 0.0;
            }
        }
        let m = &mut self.curr;
        m.k = k;
        for c in 0..k {
            let xc = &self.xt[c * n..(c + 1) * n];
            for r in 0..k {
                m.xtx[c * k + r] = dot(&self.xt[r * n..(r + 1) * n], xc);
            }
            m.xty[c] = dot(xc, &self.y);
        }
        for c in 0..k {
            for (j, &r) in self.missing.iter().enumerate() {
                self.xt[c * n + r] = saved[c * nm + j];
            }
        }
        self.set_prec_diag_curr();
        self.chol_full(false, 1);
        for (i, b) in self.bases.iter().enumerate() {
            self.curr.n_terms[i] = b.k;
        }
        self.marg_lik(false);
    }

    /// `DSVT_Propose`.
    fn propose(&mut self, id: usize, sample: u32) -> Proposal {
        let ktotal = self.curr.k as i32;
        let b = &self.bases[id];
        let good_num = b.good_num;
        let nk = b.n_knot;
        let rnd = &mut self.rnd;
        let (minkn, maxkn) = (b.prior.min_knot, b.prior.max_knot);
        let kstop = b.kstopping;
        let r = rnd.u8();
        let pp = b.prop;
        let flag = if minkn != maxkn {
            if r < pp.birth {
                let mut f = Move::Birth;
                if nk >= maxkn || good_num == 0 {
                    f = Move::Move;
                }
                if ktotal >= kstop {
                    f = if nk == 0 { Move::Birth } else { Move::Move };
                }
                f
            } else if r < pp.mov {
                if nk == 0 {
                    Move::Birth
                } else {
                    Move::Move
                }
            } else if r < pp.death {
                if nk == minkn {
                    Move::Birth
                } else {
                    Move::Death
                }
            } else if r <= pp.merge {
                if nk == minkn {
                    Move::Birth
                } else if nk >= 2 {
                    Move::Merge
                } else if nk == 0 {
                    Move::Birth
                } else {
                    Move::Death
                }
            } else if ktotal < kstop {
                Move::ChOrder
            } else if nk == minkn {
                Move::Birth
            } else {
                Move::Move
            }
        } else if minkn == 0 {
            if pp.merge == 255 {
                Move::NoChange
            } else {
                Move::ChOrder
            }
        } else if pp.merge == 255 {
            Move::Move
        } else if r > 128 {
            Move::Move
        } else {
            Move::ChOrder
        };

        let mut p = Proposal {
            jump: flag,
            ..Default::default()
        };
        let new_idx: i32;
        let end_idx: i32;
        match flag {
            Move::Birth => {
                let npad16 = self.npad16;
                let use_extreme = (rnd.u8() as f32) < 255.0 * 0.5;
                let mut use_tmp = false;
                let mut tmp_num = good_num;
                if use_extreme {
                    if sample >= self.dev_next {
                        self.calc_deviation(sample);
                        let extra = (10 + sample / 8).min(200);
                        self.dev_next = sample + extra;
                        self.dev_shall_update = true;
                    }
                    if self.dev_shall_update {
                        let thr = self.avg_dev * self.cfg.sig_factor;
                        for i in 0..self.n {
                            self.extreme[i] = (self.deviation[i] > thr) as u8;
                        }
                        self.dev_shall_update = false;
                    }
                    let b = &self.bases[id];
                    let good = b.good(npad16);
                    let mut s = 0;
                    for i in 0..npad16 {
                        let v = self.extreme[i] & good[i];
                        self.tmp_good[i] = v;
                        s += v as i32;
                    }
                    if s != 0 {
                        use_tmp = true;
                        tmp_num = s;
                    }
                }
                let rnd = &mut self.rnd;
                let b = &self.bases[id];
                let rand_loc = randint(1, tmp_num, || rnd.u32());
                let v: &[u8] = if use_tmp { &self.tmp_good } else { b.good(npad16) };
                p.new_knot = find_nth_one(&v[..npad16], rand_loc);
                let mut idx = 1;
                let mut j = 0;
                while b.knot(j) < p.new_knot {
                    j += 1;
                    idx += 1;
                }
                new_idx = idx;
                p.num_seg = 2;
                p.seg[0].r1 = b.knot(new_idx - 2);
                p.seg[0].r2 = p.new_knot - 1;
                p.seg[1].r1 = p.new_knot;
                p.seg[1].r2 = b.knot(new_idx - 1) - 1;
                let o = b.order[(new_idx - 1) as usize];
                p.seg[0].order2 = o;
                p.seg[1].order2 = o;
                end_idx = new_idx;
                p.n_knot_new = nk + 1;
            }
            Move::Death => {
                new_idx = randint(1, nk, || rnd.u16() as u32);
                p.num_seg = 1;
                p.seg[0].r1 = b.knot(new_idx - 2);
                p.seg[0].r2 = b.knot(new_idx) - 1;
                p.seg[0].order2 = b.order[new_idx as usize];
                end_idx = new_idx + 1;
                p.n_knot_new = nk - 1;
            }
            Move::Merge => {
                new_idx = randint(1, nk - 1, || rnd.u16() as u32);
                let r1 = b.knot(new_idx - 1);
                let r2 = b.knot(new_idx);
                let count = (r2 - r1) + 1 - 2;
                p.new_knot = if count == 0 {
                    if (rnd.u8() as i8) > 0 {
                        r1
                    } else {
                        r2
                    }
                } else {
                    randint(r1 + 1, r2 - 1, || rnd.u32())
                };
                p.num_seg = 2;
                p.seg[0].r1 = b.knot(new_idx - 2);
                p.seg[0].r2 = p.new_knot - 1;
                p.seg[1].r1 = p.new_knot;
                p.seg[1].r2 = b.knot(new_idx + 1) - 1;
                p.seg[0].order2 = b.order[(new_idx - 1) as usize];
                p.seg[1].order2 = b.order[(new_idx + 1) as usize];
                end_idx = new_idx + 2;
                p.n_knot_new = nk - 1;
            }
            Move::Move => {
                new_idx = randint(1, nk, || rnd.u16() as u32);
                let old = b.knot(new_idx - 1);
                let r1 = if new_idx == 1 { b.knot(-3) } else { b.knot(new_idx - 2) };
                let r2 = if new_idx == nk { b.knot(-2) } else { b.knot(new_idx) };
                let sep = b.prior.min_sep;
                let step = b.move_step;
                let r1 = (r1 + sep + 1).max(old - step);
                let r2 = (r2 - sep - 1).min(old + step);
                p.new_knot = if r2 <= r1 {
                    old
                } else {
                    // RANDINT_SKIPONE(out, r1, old, r2): r1..r2 without old.
                    let v = randint(r1, r2 - 1, || rnd.u32());
                    if v < old {
                        v
                    } else {
                        v + 1
                    }
                };
                p.num_seg = 2;
                p.seg[0].r1 = b.knot(new_idx - 2);
                p.seg[0].r2 = p.new_knot - 1;
                p.seg[1].r1 = p.new_knot;
                p.seg[1].r2 = b.knot(new_idx) - 1;
                p.seg[0].order2 = b.order[(new_idx - 1) as usize];
                p.seg[1].order2 = b.order[new_idx as usize];
                end_idx = new_idx + 1;
                p.n_knot_new = nk;
            }
            Move::ChOrder => {
                new_idx = randint(1, nk + 1, || rnd.u16() as u32);
                let old = b.order[(new_idx - 1) as usize];
                let (mino, maxo) = (b.prior.min_order, b.prior.max_order);
                let new = if old == mino {
                    old + 1
                } else if old == maxo {
                    old - 1
                } else if (rnd.u8() as i8) > 0 {
                    old - 1
                } else {
                    old + 1
                };
                p.new_order = new;
                p.old_order = old;
                p.seg[0].r1 = b.knot(new_idx - 2);
                p.seg[0].r2 = b.knot(new_idx - 1) - 1;
                p.seg[0].order1 = new;
                p.seg[0].order2 = new;
                p.num_seg = if new > old { 1 } else { 0 };
                end_idx = new_idx;
                p.n_knot_new = nk;
            }
            Move::NoChange => {
                p.num_seg = 0;
                p.new_knot = -9999;
                new_idx = 1;
                end_idx = 1;
                p.n_knot_new = nk;
            }
        }
        p.new_idx = new_idx;
        let b = &self.bases[id];
        if flag != Move::ChOrder {
            let start = if b.trend { 0 } else { 1 };
            p.seg[0].order1 = start;
            p.seg[1].order1 = start;
            if flag == Move::NoChange {
                p.k1 = b.k + 1;
                p.k2_old = b.k;
                p.new_idx = 0x0fff;
            } else {
                p.k1 = b.ks[(new_idx - 1) as usize];
                p.k2_old = b.ke[(end_idx - 1) as usize];
            }
        } else {
            p.k2_old = b.ke[(new_idx - 1) as usize];
            p.k1 = if p.new_order <= p.old_order {
                if b.trend {
                    p.k2_old
                } else {
                    p.k2_old - 1
                }
            } else {
                p.k2_old + 1
            };
        }
        p
    }

    /// `__CalcAbsDeviation`: the absolute residuals of the chain's mean fit
    /// so far.
    fn calc_deviation(&mut self, sample: u32) {
        let inv = 1.0f32 / sample as f32;
        let mut im = 0;
        let mut sum = 0f32;
        let nb = self.bases.len();
        for i in 0..self.n {
            if im < self.missing.len() && i == self.missing[im] {
                self.deviation[i] = f32::NAN;
                im += 1;
            } else {
                let pred = if nb == 1 {
                    self.acc[0].y[i]
                } else {
                    self.acc[0].y[i] + self.acc[1].y[i]
                };
                let e = (self.y[i] - pred * inv).abs();
                self.deviation[i] = e;
                sum += e;
            }
        }
        self.avg_dev = sum / self.n_good as f32;
    }

    /// `GetInfoBandList`: the bands of the current terms up to column `klast`.
    fn band_list(&mut self, klast: i32) {
        self.bands.clear();
        'outer: for b in &self.bases {
            for j in 0..=b.n_knot {
                let ks = b.kbase + b.ks[j as usize];
                if klast >= ks {
                    let ke = b.kbase + b.ke[j as usize];
                    self.bands.push(Seg {
                        r1: b.knot(j - 1),
                        r2: b.knot(j) - 1,
                        k: ke.min(klast) - ks + 1,
                        ..Default::default()
                    });
                } else {
                    break 'outer;
                }
            }
        }
    }

    /// `GetInfoBandList_post`: the bands of the current terms from column `kstart`.
    fn band_list_post(&mut self, kstart: i32) {
        self.bands.clear();
        for b in &self.bases {
            for j in 0..=b.n_knot {
                let ke = b.kbase + b.ke[j as usize];
                if kstart <= ke {
                    let ks = b.kbase + b.ks[j as usize];
                    self.bands.push(Seg {
                        r1: b.knot(j - 1),
                        r2: b.knot(j) - 1,
                        k: ke - ks.max(kstart) + 1,
                        ..Default::default()
                    });
                }
            }
        }
    }

    /// `Update_XtX_from_Xnewterm_ByGroup` and `Update_XtY_from_Xnewterm_ByGroup`.
    fn update_cross(&mut self, p: &Proposal, knewterm: usize, kold: usize, knew: usize) {
        let n = self.n;
        let (k1, k2o, k2n) = (p.k1 as usize, p.k2_old as usize, p.k2_new as usize);
        for i in 1..k1 {
            let (src, dst) = ((i - 1) * kold, (i - 1) * knew);
            self.prop.xtx[dst..dst + i].copy_from_slice(&self.curr.xtx[src..src + i]);
        }
        let segs = &p.seg[..p.num_seg];
        if knewterm != 0 {
            let off = (k1 - 1) * knew;
            self.prop.xtx[off..off + (knew - k1 + 1) * knew].fill(0.0);
            if k1 > 1 {
                self.band_list(p.k1 - 1);
                mat_x_mat(
                    &self.bands,
                    &self.xt,
                    segs,
                    &self.xnew,
                    &mut self.prop.xtx,
                    off,
                    n,
                    knew,
                );
            }
            // XtX_ByGroup, with its r2 = max of the two segments' ends.
            let out = off + k1 - 1;
            let mut colcsum = 0usize;
            for i in 0..segs.len() {
                let kcol = segs[i].k as usize;
                let mut rowcsum = 0usize;
                for j in 0..=i {
                    let krow = segs[j].k as usize;
                    let r1 = segs[i].r1.max(segs[j].r1);
                    let r2 = segs[i].r2.max(segs[j].r2);
                    if r2 >= r1 {
                        let (a, z) = ((r1 - 1) as usize, r2 as usize);
                        let (xr, xc) = (&self.xnew[rowcsum * n + a..], &self.xnew[colcsum * n + a..]);
                        let o = out + colcsum * knew + rowcsum;
                        cross(xr, n, krow, xc, n, kcol, z - a, &mut self.prop.xtx[o..], knew);
                    }
                    rowcsum += krow;
                }
                colcsum += kcol;
            }
        }
        if k2o != kold {
            let mut kn = k2n + 1;
            for ko in k2o + 1..=kold {
                let (so, sn) = ((ko - 1) * kold, (kn - 1) * knew);
                self.prop.xtx[sn..sn + k1 - 1].copy_from_slice(&self.curr.xtx[so..so + k1 - 1]);
                let len = ko - k2o;
                self.prop.xtx[sn + k2n..sn + k2n + len].copy_from_slice(&self.curr.xtx[so + k2o..so + k2o + len]);
                kn += 1;
            }
            if knewterm != 0 {
                self.band_list_post(p.k2_old + 1);
                let off = k2n * knew + k1 - 1;
                mat_x_mat(
                    segs,
                    &self.xnew,
                    &self.bands,
                    &self.xt[k2o * n..],
                    &mut self.prop.xtx,
                    off,
                    n,
                    knew,
                );
            }
        }
        // XtY.
        if k1 > 1 {
            let (src, dst) = (&self.curr.xty[..k1 - 1], k1 - 1);
            self.prop.xty[..dst].copy_from_slice(src);
        }
        if knewterm > 0 {
            let mut col = 0usize;
            for s in segs {
                let (a, z) = ((s.r1 - 1) as usize, s.r2 as usize);
                for c in 0..s.k as usize {
                    self.prop.xty[k1 - 1 + col + c] =
                        dot(&self.xnew[(col + c) * n + a..(col + c) * n + z], &self.y[a..z]);
                }
                col += s.k as usize;
            }
        }
        if k2o != kold {
            let len = knew - k2n;
            self.prop.xty[k2n..k2n + len].copy_from_slice(&self.curr.xty[k2o..k2o + len]);
        }
    }

    /// Fits one series of `n` values (NaN where missing); `None` when it
    /// cannot be fitted (too many missing values, or the sampler failed).
    pub fn fit(&mut self, values: &[f32]) -> Option<Fit> {
        let cfg = self.cfg.clone();
        let n = self.n;
        assert_eq!(values.len(), n);
        let nb = self.bases.len();
        let tid = nb - 1;

        // f32_normalize_multicols_zeroout_nans.
        self.missing.clear();
        let (mut sum_y, mut sum_yy) = (0f64, 0f64);
        for (i, &v) in values.iter().enumerate() {
            let bad = !v.is_finite();
            if bad {
                self.missing.push(i);
            }
            let y = if bad { 0f32 } else { v };
            sum_y += y as f64;
            sum_yy += (y * y) as f64;
        }
        let nmiss = self.missing.len();
        let ngood = n - nmiss;
        if nmiss as f32 > n as f32 * cfg.max_missing_rate || ngood == 0 {
            return None;
        }
        let mean = (sum_y / ngood as f64) as f32;
        let mut sd = sum_yy / ngood as f64 - (mean * mean) as f64;
        sd = if sd > 0.0 { sd.sqrt() } else { 1.0 };
        let mut im = 0;
        for i in 0..n {
            if im < nmiss && i == self.missing[im] {
                self.y[i] = 0.0;
                im += 1;
            } else {
                self.y[i] = ((values[i] - mean) as f64 * (1.0 / sd)) as f32;
            }
        }
        let sd = sd as f32;
        self.n_good = ngood as i32;
        self.yty = dot(&self.y, &self.y) + 2.0 * cfg.alpha2;
        self.alpha1_star = (ngood as f64 * 0.5 + cfg.alpha1 as f64) as f32;

        // A fresh stream and model per series, as Rbeast's beast() for one.
        let mut rng = Pcg::new(cfg.seed.wrapping_add(0x4f352a3dc));
        self.prec = [cfg.prec_value; 2];
        self.log_prec = [rng::logf(cfg.prec_value); 2];
        self.sig2 = cfg.sig2;
        let gamma_a = cfg.alpha1 + ngood as f32 * 0.5f32;
        let npg = self.n_prec_grp();
        self.rnd.start(&mut rng, gamma_a, npg);

        let with_amp = cfg.season;
        let mut res_season = if cfg.season {
            Some(Component::zeros(n, cfg.sp.max_knot as usize, cfg.ci, false, with_amp))
        } else {
            None
        };
        let mut res_trend = Component::zeros(n, cfg.tp.max_knot as usize, cfg.ci, true, false);
        let mut res_ml = 0f32;
        let mut res_sig2 = 0f32;

        for _chain in 0..cfg.chains {
            for b in self.bases.iter_mut() {
                b.init_knots(n as i32);
            }
            self.evaluate_curr();
            for b in self.bases.iter_mut() {
                b.knots_to_goodvec(n);
            }
            self.extreme[..n].fill(1);
            self.deviation.fill(1e30);
            for &r in &self.missing {
                self.extreme[r] = 0;
                self.deviation[r] = f32::NAN;
            }
            self.avg_dev = 1.0;
            for a in self.acc.iter_mut() {
                a.reset();
            }
            self.amp.fill(0.0);
            self.amp_sd.fill(0.0);
            self.slp.fill(0.0);
            self.slp_sd.fill(0.0);
            self.slp_pos.fill(0);
            self.slp_zero.fill(0);
            for c in self.ci.iter_mut() {
                c.reset();
            }
            self.update_kbase();
            self.dev_next = 1;
            self.dev_shall_update = false;
            let mut chain = Chain {
                marg_lik: 0.0,
                sig2: 0.0,
                samples: 0,
            };
            let ok = self.run_chain(&mut rng, &mut chain);
            if !ok {
                return None;
            }
            self.finish_chain(
                &chain,
                sd,
                mean,
                res_season.as_mut(),
                &mut res_trend,
                &mut res_ml,
                &mut res_sig2,
            );
        }
        // Averaged over the chains.
        let chains = cfg.chains;
        if chains >= 2 {
            let inv = 1.0f32 / chains as f32;
            res_ml *= inv;
            res_sig2 *= inv;
            for c in res_season.iter_mut().chain(std::iter::once(&mut res_trend)) {
                c.ncp *= inv;
                for v in c
                    .ncp_pr
                    .iter_mut()
                    .chain(c.cp_occ.iter_mut())
                    .chain(c.y.iter_mut())
                    .chain(c.sd.iter_mut())
                    .chain(c.order.iter_mut())
                    .chain(c.extra.iter_mut())
                    .chain(c.extra_sd.iter_mut())
                    .chain(c.slope_pos_pr.iter_mut())
                    .chain(c.slope_zero_pr.iter_mut())
                    .chain(c.ci.iter_mut())
                    .chain(c.slope_ci.iter_mut())
                {
                    *v *= inv;
                }
            }
        }
        // The mode and percentiles of the number of changepoints (Rbeast
        // leaves them 0 with one chain; they are computed here for any).
        for c in res_season.iter_mut().chain(std::iter::once(&mut res_trend)) {
            c.ncp_mode = maxidx(&c.ncp_pr) as f32;
            c.ncp_median = percentile_ncp(&c.ncp_pr, 0.5);
            c.ncp_pct90 = percentile_ncp(&c.ncp_pr, 0.9);
            c.ncp_pct10 = percentile_ncp(&c.ncp_pr, 0.1);
            remove_nans(&mut c.sd);
        }
        remove_nans(&mut res_trend.extra_sd);

        // The changepoints.
        let (t0, dt) = (cfg.start, cfg.dt);
        if let Some(s) = res_season.as_mut() {
            changepoints(s, &cfg.sp, t0, dt);
        }
        changepoints(&mut res_trend, &cfg.tp, t0, dt);

        // The data as fitted, and the fit's correlation and error.
        let mut data = vec![0f32; n];
        for i in 0..n {
            data[i] = self.y[i] * sd + mean;
        }
        for &r in &self.missing {
            data[r] = f32::NAN;
        }
        let mut fit = res_trend.y.clone();
        if let Some(s) = &res_season {
            for i in 0..n {
                fit[i] += s.y[i];
            }
        }
        let (r, rmse) = corr_rmse_nan(&fit, &data);
        let _ = tid;
        Some(Fit {
            marg_lik: res_ml,
            sig2: res_sig2,
            r2: r * r,
            rmse,
            season: res_season,
            trend: res_trend,
            data,
        })
    }

    /// One chain; false when the sampler failed (Rbeast's `skipCurrentPixel`).
    fn run_chain(&mut self, rng: &mut Pcg, chain: &mut Chain) -> bool {
        let cfg = self.cfg.clone();
        let n = self.n;
        let nb = self.bases.len();
        let tid = nb - 1;
        let npad16 = self.npad16;
        let mut ite = 0u32;
        let mut sample = 0u32;
        let mut num_bad = 0;
        let nmiss = self.missing.len();
        while sample < cfg.samples {
            ite += 1;
            self.rnd.refill(rng);
            let id = self.pick_basis();
            let mut p = self.propose(id, sample);
            let mut knewterm = 0usize;
            for i in 0..p.num_seg {
                let seg = p.seg[i];
                let k = self.bases[id].gen_terms(&mut self.xnew[n * knewterm..], n, &seg);
                p.seg[i].k = k;
                knewterm += k as usize;
            }
            let kbase = self.bases[id].kbase;
            p.k2_new = p.k1 + knewterm as i32 - 1;
            p.k1 += kbase;
            p.k2_old += kbase;
            p.k2_new += kbase;
            let kold = self.curr.k;
            let knew = (kold as i32 + p.k2_new - p.k2_old) as usize;
            if nmiss > 0 && knewterm > 0 {
                for c in 0..knewterm {
                    for (j, &r) in self.missing.iter().enumerate() {
                        self.backup[c * nmiss + j] = self.xnew[c * n + r];
                        self.xnew[c * n + r] = 0.0;
                    }
                }
            }
            self.update_cross(&p, knewterm, kold, knew);
            let k1 = p.k1 as usize;
            for i in 1..k1 {
                let (src, dst) = ((i - 1) * kold, (i - 1) * knew);
                self.prop.chol[dst..dst + i].copy_from_slice(&self.curr.chol[src..src + i]);
            }
            if cfg.prec_type == PrecType::ComponentWise {
                let (k2o, k2n) = (p.k2_old as usize, p.k2_new as usize);
                self.prop.prec_diag[..k1 - 1].copy_from_slice(&self.curr.prec_diag[..k1 - 1]);
                let len = kold - k2o;
                self.prop.prec_diag[k2n..k2n + len].copy_from_slice(&self.curr.prec_diag[k2o..k2o + len]);
                for k in k1..=k2n {
                    self.prop.prec_diag[k - 1] = self.prec[id];
                }
                self.prop.n_terms = self.curr.n_terms;
                self.prop.n_terms[id] += p.k2_new - p.k2_old;
            }
            self.prop.k = knew;
            self.chol_full(true, k1);
            self.marg_lik(true);
            if !self.prop.marg_lik.is_finite() {
                num_bad += 1;
                if num_bad < 15 {
                    self.increase_prec();
                    self.set_prec_diag_curr();
                    self.chol_full(false, 1);
                    self.marg_lik(false);
                    continue;
                } else {
                    return false;
                }
            } else {
                num_bad = 0;
            }
            let mut delta = self.prop.marg_lik - self.curr.marg_lik;
            if p.jump != Move::Move {
                delta += self.bases[id].model_prior(&p);
            }
            let accept = if delta > 0.0 {
                true
            } else if delta < -23.0 {
                false
            } else {
                let e = fastexp(delta);
                if delta > -0.5 {
                    (self.rnd.u8() as f32) < e * 255.0f32
                } else if delta > -5.0 {
                    (self.rnd.u16() as f32) < e * 65535.0f32
                } else {
                    (self.rnd.u32() as f64) < e as f64 * 4.294967296e+09
                }
            };
            if accept {
                if nmiss > 0 && knewterm > 0 {
                    for c in 0..knewterm {
                        for (j, &r) in self.missing.iter().enumerate() {
                            self.xnew[c * n + r] = self.backup[c * nmiss + j];
                        }
                    }
                }
                let (k2o, k2n) = (p.k2_old as usize, p.k2_new as usize);
                if k2o != kold && k2n != k2o {
                    self.xt.copy_within(k2o * n..kold * n, k2n * n);
                }
                if knewterm != 0 {
                    self.xt[(k1 - 1) * n..(k1 - 1 + knewterm) * n].copy_from_slice(&self.xnew[..knewterm * n]);
                }
                let b = &mut self.bases[id];
                let nk = b.n_knot;
                b.set_knot(-1, b.knot(-3));
                b.set_knot(nk, b.knot(-2));
                b.update_knots(&p, npad16);
                b.set_knot(-1, 1);
                let nk = b.n_knot;
                b.set_knot(nk, n as i32 + 1);
                b.calc_ks_ke();
                if id == 0 {
                    self.update_kbase();
                }
                std::mem::swap(&mut self.curr, &mut self.prop);
            }

            let resample = ite >= 100 && ite.is_multiple_of(20);
            let store = ite > cfg.burnin && ite.is_multiple_of(cfg.thin);
            if resample || store {
                let g = self.rnd.gamma();
                let sig2_inv = g * 1.0f32 / self.curr.alpha2q;
                let sig2 = 1.0f32 / sig2_inv;
                if sig2 as f64 > MIN_SIG2_VALUE {
                    self.sig2 = sig2;
                }
            }
            if resample {
                let k = self.curr.k;
                rng::gauss(rng, &mut self.beta[..k]);
                solve_u_invdiag(&self.curr.chol, &mut self.beta, k, k);
                let s = fastsqrt(self.sig2);
                for v in self.beta[..k].iter_mut() {
                    *v *= s;
                }
                for i in 0..k {
                    self.beta[i] += self.curr.beta_mean[i];
                }
                let mut ntries = 0;
                loop {
                    if ntries == 0 {
                        self.resample_prec(rng);
                    } else {
                        self.increase_prec();
                    }
                    ntries += 1;
                    self.set_prec_diag_curr();
                    self.chol_full(false, 1);
                    self.marg_lik(false);
                    if !(self.curr.marg_lik.is_nan() && ntries < 20) {
                        break;
                    }
                }
                if self.curr.marg_lik.is_nan() {
                    return false;
                }
            }
            if !store {
                continue;
            }
            sample += 1;
            chain.marg_lik += self.curr.marg_lik;
            chain.sig2 += self.sig2;
            for i in 0..nb {
                let b = &self.bases[i];
                let a = &mut self.acc[i];
                a.ncp_pr[b.n_knot as usize] += 1;
                for j in 0..b.n_knot {
                    a.cp_occ[(b.knot(j) - 1) as usize] += 1;
                }
                for j in 0..=b.n_knot {
                    let (r1, r2) = (b.knot(j - 1), b.knot(j) - 1);
                    let o = b.order[j as usize];
                    for v in &mut a.order[(r1 - 1) as usize..r2 as usize] {
                        *v += o;
                    }
                }
                let (kb, k) = (b.kbase as usize, b.k as usize);
                let row = &mut self.membuf[i * n..(i + 1) * n];
                gemv(&self.xt[kb * n..], n, n, &self.curr.beta_mean[kb..kb + k], row);
                for ((&y, s), q) in row.iter().zip(&mut a.y[..n]).zip(&mut a.ysd[..n]) {
                    *s += y;
                    *q += y * y;
                }
            }
            if cfg.season {
                let b = &self.bases[0];
                let beta = &self.curr.beta_mean;
                let mut boff = 0usize;
                let amp_row = 3 * n;
                for i in 0..=b.n_knot {
                    let (r1, r2) = (b.knot(i - 1), b.knot(i) - 1);
                    let order2 = (b.order[i as usize] * 2) as usize;
                    let mut amp = 0f32;
                    if b.btype == 0 {
                        for j in 0..order2 {
                            let cs = &b.sqr_csum[j * (n + 1) + 1..];
                            let lo = b.sqr_csum[j * (n + 1) + (r1 - 1) as usize];
                            let scaling = n as f32 / (cs[(r2 - 1) as usize] - lo);
                            let b0 = beta[boff + j] * b.scale_factor[j];
                            amp += (b0 * b0) * scaling;
                        }
                    } else {
                        for j in 0..order2 {
                            let b0 = beta[boff + j] * b.scale_factor[j];
                            amp += b0 * b0;
                        }
                    }
                    self.membuf[amp_row + (r1 - 1) as usize..amp_row + r2 as usize].fill(amp);
                    boff += order2;
                }
                let amps = &self.membuf[amp_row..amp_row + n];
                for ((&a, s), q) in amps.iter().zip(&mut self.amp[..n]).zip(&mut self.amp_sd[..n]) {
                    *s += a;
                    *q += a * a;
                }
            }
            {
                // f32_diff_back of the trend into the slope row.
                let (t, s) = (tid * n, nb * n);
                for i in 1..n {
                    self.membuf[s + i] = self.membuf[t + i] - self.membuf[t + i - 1];
                }
                self.membuf[s] = self.membuf[s + 1];
                let slopes = &self.membuf[s..s + n];
                let sums = self.slp[..n].iter_mut().zip(&mut self.slp_sd[..n]);
                let counts = self.slp_pos[..n].iter_mut().zip(&mut self.slp_zero[..n]);
                for ((&v, (sm, sq)), (pos, zero)) in slopes.iter().zip(sums).zip(counts) {
                    *sm += v;
                    *sq += v * v;
                    // v > 1e-10 and |v| < 1e-10 in double, as the C compares.
                    *pos += (v as f64 > 1e-10) as i32;
                    *zero += ((v as f64) < 1e-10 && v as f64 > -1e-10) as i32;
                }
            }
            if cfg.ci && (!cfg.ci_fast || self.rnd.u16() < cfg.ci_sub) {
                for (v, c) in self.ci.iter_mut().enumerate() {
                    c.insert(&self.membuf[v * n..(v + 1) * n]);
                }
            }
        }
        chain.samples = sample;
        true
    }

    fn pick_basis(&mut self) -> usize {
        if self.bases.len() == 1 {
            return 0;
        }
        let fixed =
            |b: &Basis| b.prior.min_order == b.prior.max_order && b.prior.min_knot == 0 && b.prior.max_knot == 0;
        let (f0, f1) = (fixed(&self.bases[0]), fixed(&self.bases[1]));
        if f0 && f1 {
            0
        } else if f0 {
            1
        } else if f1 {
            0
        } else {
            (self.rnd.u8() > 128) as usize
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_chain(
        &mut self,
        chain: &Chain,
        sd: f32,
        mean: f32,
        season: Option<&mut Component>,
        trend: &mut Component,
        res_ml: &mut f32,
        res_sig2: &mut f32,
    ) {
        let cfg = &self.cfg;
        let n = self.n;
        let nb = self.bases.len();
        let samples = chain.samples;
        let inv = 1.0f32 / samples as f32;
        *res_ml += chain.marg_lik * inv;
        *res_sig2 += chain.sig2 * inv * sd * sd;
        let to_f32 = |x: &[i32], out: &mut [f32]| {
            for (o, &v) in out.iter_mut().zip(x) {
                *o += v as f32 * inv;
            }
        };
        let avgstd = |sx: &[f32], sxx: &[f32], scale: f32, offset: f32, y: &mut [f32], s: &mut [f32]| {
            let inv1 = (1.0 / samples as f64 * scale as f64) as f32;
            let inv2 = (1.0 / samples as f64 * scale as f64 * scale as f64) as f32;
            for i in 0..sx.len() {
                let avg = sx[i] * inv1;
                let sxx_n = sxx[i] * inv2;
                s[i] += (sxx_n - avg * avg).sqrt();
                y[i] += avg + offset;
            }
        };
        let mut comps: Vec<&mut Component> = Vec::new();
        if let Some(s) = season {
            comps.push(s);
        }
        comps.push(trend);
        for (i, c) in comps.into_iter().enumerate() {
            let a = &self.acc[i];
            let is_trend = i == nb - 1;
            let total: i32 = a.cp_occ.iter().sum();
            c.ncp += total as f32 * inv;
            to_f32(&a.ncp_pr, &mut c.ncp_pr);
            to_f32(&a.cp_occ, &mut c.cp_occ);
            avgstd(&a.y, &a.ysd, sd, if is_trend { mean } else { 0.0 }, &mut c.y, &mut c.sd);
            to_f32(&a.order, &mut c.order);
            if !is_trend {
                avgstd(&self.amp, &self.amp_sd, sd * sd, 0.0, &mut c.extra, &mut c.extra_sd);
                if cfg.ci {
                    let r = &self.ci[0].result;
                    for j in 0..2 * n {
                        c.ci[j] += r[j] * sd;
                    }
                }
            } else {
                avgstd(&self.slp, &self.slp_sd, sd / cfg.dt, 0.0, &mut c.extra, &mut c.extra_sd);
                to_f32(&self.slp_pos, &mut c.slope_pos_pr);
                to_f32(&self.slp_zero, &mut c.slope_zero_pr);
                if cfg.ci {
                    let r = &self.ci[i].result;
                    for j in 0..2 * n {
                        c.ci[j] += r[j] * sd + mean;
                    }
                    // Rbeast sums the slope's interval over the chains and
                    // leaves it per time step; here it is their mean, per
                    // unit of time as the slope.
                    let slope_scale = sd / cfg.dt;
                    let r = &self.ci[i + 1].result;
                    for j in 0..2 * n {
                        c.slope_ci[j] += r[j] * slope_scale;
                    }
                }
            }
        }
        let _ = n;
    }
}

/// `MatxMat`: the cross products of the `x` bands' columns with the `y`
/// bands', over the rows both span, into `out` from `off` (leading
/// dimension `ld`).
#[allow(clippy::too_many_arguments)]
fn mat_x_mat(xb: &[Seg], x: &[f32], yb: &[Seg], y: &[f32], out: &mut [f32], off: usize, n: usize, ld: usize) {
    let mut ycol = 0usize;
    for yi in yb {
        let kseg = yi.k as usize;
        let mut xcol = 0usize;
        for xj in xb {
            let kband = xj.k as usize;
            let r1 = yi.r1.max(xj.r1);
            let r2 = yi.r2.min(xj.r2);
            if r2 >= r1 {
                let (a, z) = ((r1 - 1) as usize, r2 as usize);
                let o = off + ycol * ld + xcol;
                cross(
                    &x[xcol * n + a..],
                    n,
                    kband,
                    &y[ycol * n + a..],
                    n,
                    kseg,
                    z - a,
                    &mut out[o..],
                    ld,
                );
            }
            xcol += kband;
        }
        ycol += kseg;
    }
}

/// `gen_f32_maxidx`: the first index of the largest value, compared in pairs.
fn maxidx(x: &[f32]) -> usize {
    let n = x.len();
    if n == 0 {
        return 0;
    }
    let regular = n & !1;
    let mut max_val = x[0];
    let mut max_idx = 0;
    let mut i = 0;
    while i < regular {
        let (val, idx) = if x[i + 1] > x[i] { (x[i + 1], i + 1) } else { (x[i], i) };
        if max_val < val {
            max_val = val;
            max_idx = idx;
        }
        i += 2;
    }
    while i < n {
        if max_val < x[i] {
            max_val = x[i];
            max_idx = i;
        }
        i += 1;
    }
    max_idx
}

/// `GetPercentileNcp`.
fn percentile_ncp(prob: &[f32], pct: f32) -> f32 {
    let mut cum = 0f32;
    for (i, &p) in prob.iter().enumerate() {
        cum += p;
        if cum > pct {
            return i as f32;
        }
    }
    prob.len() as f32 - 1.0
}

/// `tsRemoveNaNs`: NaNs filled by linear interpolation (the nearest value
/// at the ends).
fn remove_nans(x: &mut [f32]) {
    let n = x.len() as i64;
    let mut pre: i64 = -1;
    let mut post: i64 = -1;
    let mut i: i64 = 0;
    while i < n {
        let iu = i as usize;
        if !x[iu].is_nan() {
            pre = i;
            i += 1;
            continue;
        }
        if post <= i {
            for j in i + 1..n {
                if !x[j as usize].is_nan() {
                    post = j;
                    break;
                }
            }
        }
        if pre < 0 {
            if post <= i {
                break;
            } else {
                x[iu] = x[post as usize];
            }
        } else if post <= i {
            x[iu] = x[pre as usize];
        } else {
            x[iu] = (x[pre as usize] * (post - i) as f32 + x[post as usize] * (i - pre) as f32) / (post - pre) as f32;
        }
        pre = i;
        i += 1;
    }
}

/// `f32_corr_rmse_nan`: the correlation and the root mean square
/// difference of `x` and `y` where both have values.
fn corr_rmse_nan(x: &[f32], y: &[f32]) -> (f32, f32) {
    let (mut sx, mut sxx, mut sy, mut syy, mut sxy, mut d2) = (0f32, 0f32, 0f32, 0f32, 0f32, 0f32);
    let mut n = 0i32;
    for i in 0..x.len() {
        let good = !x[i].is_nan() && !y[i].is_nan();
        n += good as i32;
        let a = if good { x[i] } else { 0.0 };
        let b = if good { y[i] } else { 0.0 };
        sx += a;
        sy += b;
        sxx += a * a;
        syy += b * b;
        sxy += a * b;
        d2 += (a - b) * (a - b);
    }
    let nf = n as f32;
    let r = (nf * sxy - sx * sy) / ((nf * sxx - sx * sx) * (nf * syy - sy * sy)).sqrt();
    (r, (d2 / nf).sqrt())
}

/// The changepoints of a component, as the core lists them after the chains.
fn changepoints(c: &mut Component, prior: &super::config::BasisPrior, t0: f32, dt: f32) {
    let maxk = prior.max_knot.max(0) as usize;
    let found = cpt::find(
        &c.cp_occ,
        0.01f32,
        prior.min_sep,
        prior.max_knot,
        prior.left,
        prior.right,
    );
    let num = found.pos.len();
    c.cp = vec![f32::NAN; maxk];
    c.cp_pr = vec![f32::NAN; maxk];
    c.cp_abrupt = vec![f32::NAN; maxk];
    c.cp_ci = vec![f32::NAN; 2 * maxk];
    for i in 0..num.min(maxk) {
        let pos = found.pos[i];
        c.cp[i] = pos as f32 * dt + t0;
        c.cp_pr[i] = found.prob[i];
        let loc = if pos == 0 { 1 } else { pos } as usize;
        c.cp_abrupt[i] = c.y[loc] - c.y[loc - 1];
        c.cp_ci[i] = found.ci[i] * dt + t0;
        c.cp_ci[maxk + i] = found.ci[num + i] * dt + t0;
    }
}
