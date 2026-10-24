//! The two components of BEAST's model, each a piecewise basis over the
//! segments its changepoints ("knots") cut: a piecewise harmonic season and a
//! piecewise polynomial trend. Their precomputed terms (`preCalc_terms_*` of
//! `abc_ts_func.c`), the terms of a segment (`beastv2_basis_gensegment.c`),
//! the proposals of the reversible-jump sampler (`beastv2_basis_proposeNew_q.c`),
//! the bookkeeping after a move is accepted (`beastv2_basis_updategoodvec.c`,
//! `beastv2_basis_updateKsKe_prec0123.c`) and the prior on the number of
//! models (`beastv2_prior_model.c`).

use super::config::BasisPrior;
use super::kern::{dot, fastsqrt, normalize, normalize_factor, seq};

/// The moves of the sampler (`MOVETYPE`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Move {
    Birth,
    Death,
    Merge,
    Move,
    ChOrder,
    NoChange,
}

/// A segment of new terms (`BEAST2_BASESEG`).
#[derive(Clone, Copy, Debug, Default)]
pub struct Seg {
    pub r1: i32,
    pub r2: i32,
    pub order1: i32,
    pub order2: i32,
    pub k: i32,
}

/// A proposal (`NEWTERM` and its `NEWCOLINFO`).
#[derive(Clone, Copy, Debug)]
pub struct Proposal {
    pub seg: [Seg; 2],
    pub num_seg: usize,
    pub new_knot: i32,
    pub new_idx: i32,
    pub n_knot_new: i32,
    pub jump: Move,
    pub new_order: i32,
    pub old_order: i32,
    pub k1: i32,
    pub k2_old: i32,
    pub k2_new: i32,
}

impl Default for Proposal {
    fn default() -> Self {
        Proposal {
            seg: [Seg::default(); 2],
            num_seg: 0,
            new_knot: 0,
            new_idx: 0,
            n_knot_new: 0,
            jump: Move::NoChange,
            new_order: 0,
            old_order: 0,
            k1: 0,
            k2_old: 0,
            k2_new: 0,
        }
    }
}

/// The probabilities of the moves, out of 255 (`PROP_PROB_STRUCT`).
#[derive(Clone, Copy, Debug)]
pub struct PropProb {
    pub birth: u8,
    pub mov: u8,
    pub death: u8,
    pub merge: u8,
}

impl PropProb {
    pub fn new(resample_order_prob: f32) -> Self {
        let cum = 1.0f32 - resample_order_prob;
        let c = cum as f64;
        PropProb {
            birth: (c * 0.33 * 255.0) as u8,
            mov: (c * (0.33 + 0.33) * 255.0) as u8,
            death: (c * (0.33 + 0.33 + 0.165) * 255.0) as u8,
            merge: (c * 1.0 * 255.0) as u8,
        }
    }
}

/// One component (`BEAST2_BASIS`).
#[derive(Clone)]
pub struct Basis {
    pub trend: bool,
    pub prior: BasisPrior,
    /// `KNOT[-3..]`: the fake start and end, the knot before the first
    /// (1), the knots, and N + 1 after the last; `knot(i)` is `KNOT[i]`.
    knots: Vec<i32>,
    pub order: Vec<i32>,
    pub ks: Vec<i32>,
    pub ke: Vec<i32>,
    pub k: i32,
    pub kbase: i32,
    pub n_knot: i32,
    /// The positions where a knot may be born (1) or not (0), with padding
    /// on both sides; `goodvec[good_origin]` is the first value's.
    pub goodvec: Vec<u8>,
    pub good_origin: usize,
    pub good_num: i32,
    pub kstopping: i32,
    pub move_step: i32,
    pub prop: PropProb,
    /// The log numbers of models (`priorNmodelsPerNseg`, `priorMat`).
    pub prior_vec: Vec<f64>,
    pub prior_mat: Vec<f64>,
    /// The terms of the whole series: trend powers or season cosines and
    /// sines, `n` values each.
    pub terms: Vec<f32>,
    pub inv_sqr: Vec<f32>,
    pub coeff_a: Vec<f32>,
    pub coeff_b: Vec<f32>,
    pub sqr_csum: Vec<f32>,
    pub scale_factor: Vec<f32>,
    pub btype: u8,
}

fn k_out_of_n(n: i32, k: i32) -> f64 {
    if k < 0 || k > n {
        return 0.0;
    }
    let k = k.min(n - k);
    let mut y = 1f64;
    let mut nd = n as f64;
    let mut kd = k as f64;
    for _ in 0..k {
        y *= nd / kd;
        nd -= 1.0;
        kd -= 1.0;
    }
    y
}

/// `PreCaclModelNumber` (prior type 1): the log numbers of models per number
/// of segments, and per number of segments and of terms.
fn model_numbers(min_order: i32, max_order: i32, max_numseg: i32, n: i32, min_sep: i32) -> (Vec<f64>, Vec<f64>) {
    let mut vec = vec![0f64; max_numseg as usize];
    for ng in 1..=max_numseg {
        let v = k_out_of_n(n - min_sep * ng + ng - 1, ng - 1);
        vec[(ng - 1) as usize] = if v > 0.0 { v.ln() } else { v };
    }
    let kmax = max_numseg * max_order;
    let mut mat = vec![0f64; (kmax * max_numseg).max(0) as usize];
    for ng in 1..=max_numseg {
        for m in min_order * ng..=max_order * ng {
            let v = k_out_of_n(m - ng * min_order + ng - 1, ng - 1);
            let idx = ((ng - 1) * kmax + m - 1) as usize;
            if idx < mat.len() {
                mat[idx] = if v > 0.0 { v.ln() } else { v };
            }
        }
    }
    (vec, mat)
}

/// `preCalc_terms_season`: the cosine and sine of each order, normalized to
/// a mean square of 1, their scale factors and the cumulative sums of their
/// squares (with a leading 0).
fn season_terms(n: usize, period: f32, max_order: i32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mo = max_order.max(0) as usize;
    let mut terms = vec![0f32; n * mo * 2];
    let mut csum = vec![0f32; (n + 1) * mo * 2];
    let mut scale = vec![0f32; mo * 2];
    let freq = 2.0f32 * std::f32::consts::PI / period;
    for order in 1..=mo {
        let base = (order - 1) * 2 * n;
        let (c, s) = terms[base..base + 2 * n].split_at_mut(n);
        seq(c, 1.0, 1.0);
        let f = freq * order as f32;
        for v in c.iter_mut() {
            *v *= f;
        }
        s.copy_from_slice(c);
        for i in 0..n {
            let x = s[i] as f64;
            c[i] = x.cos() as f32;
            s[i] = x.sin() as f32;
        }
        let d = dot(c, c);
        let scale1 = 1.0f32 / (d / n as f32).sqrt();
        for v in c.iter_mut() {
            *v *= scale1;
        }
        let d = dot(s, s);
        let scale2 = 1.0f32 / (d / n as f32).sqrt();
        for v in s.iter_mut() {
            *v *= scale2;
        }
        scale[(order - 1) * 2] = scale1;
        scale[(order - 1) * 2 + 1] = scale2;
        for (col, src) in [(0usize, &*c), (1, &*s)] {
            let off = ((order - 1) * 2 + col) * (n + 1);
            csum[off] = 0.0;
            let mut acc = 0f32;
            for i in 0..n {
                acc += src[i] * src[i];
                csum[off + 1 + i] = acc;
            }
        }
    }
    (terms, csum, scale)
}

/// `preCalc_terms_trend` and `preCalc_XmarsTerms_extra`: the powers of
/// 1..n normalized, `sqrt(n) / sqrt(i)`, and the coefficients of the linear
/// term of a segment of length i.
fn trend_terms(n: usize, max_order: i32) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mo = max_order.max(0) as usize;
    let mut terms = vec![0f32; n * (mo + 1)];
    for i in 1..=mo + 1 {
        let col = &mut terms[(i - 1) * n..i * n];
        if i == 1 {
            col.fill(1.0);
        } else {
            seq(col, 1.0, 1.0);
            let pw = (i - 1) as f32 as f64;
            for v in col.iter_mut() {
                *v = (*v as f64).powf(pw) as f32;
            }
            normalize(col);
        }
    }
    let sqrt_n = fastsqrt(n as f32);
    let inv_sqr: Vec<f32> = (0..n).map(|i| sqrt_n / fastsqrt((i + 1) as f32)).collect();
    let mut a = vec![0f32; n];
    let mut b = vec![0f32; n];
    b[0] = 0.0;
    a[0] = fastsqrt(n as f32);
    for m in 2..=n {
        let mi = m as i64;
        let sum = (1 + mi) as f32 / 2.0f32;
        let bb = 1.0f32 / (((mi + 1) * (2 * mi + 1)) as f32 / 6.0f32 - sum * sum);
        let bsqrt = fastsqrt(bb * n as f32 / m as f32);
        b[m - 1] = bsqrt;
        a[m - 1] = -bsqrt * sum;
    }
    (terms, inv_sqr, a, b)
}

impl Basis {
    pub fn new_trend(prior: BasisPrior, n: usize, k_max: i32, move_step: i32, resample: f32, btype: u8) -> Self {
        let (terms, inv_sqr, coeff_a, coeff_b) = trend_terms(n, prior.max_order);
        let (prior_vec, prior_mat) = model_numbers(
            prior.min_order + 1,
            prior.max_order + 1,
            prior.max_knot + 1,
            n as i32,
            prior.min_sep,
        );
        let mut b = Self::empty(true, prior, n);
        b.kstopping = k_max - (prior.max_order + 1);
        b.move_step = move_step;
        b.prop = PropProb::new(resample);
        b.terms = terms;
        b.inv_sqr = inv_sqr;
        b.coeff_a = coeff_a;
        b.coeff_b = coeff_b;
        b.prior_vec = prior_vec;
        b.prior_mat = prior_mat;
        b.btype = btype;
        b
    }

    pub fn new_season(
        prior: BasisPrior,
        n: usize,
        period: f32,
        k_max: i32,
        move_step: i32,
        resample: f32,
        btype: u8,
    ) -> Self {
        let (terms, sqr_csum, scale_factor) = season_terms(n, period, prior.max_order);
        let (prior_vec, prior_mat) = model_numbers(
            prior.min_order,
            prior.max_order,
            prior.max_knot + 1,
            n as i32,
            prior.min_sep,
        );
        let mut b = Self::empty(false, prior, n);
        b.kstopping = k_max - 2 * prior.max_order;
        b.move_step = move_step;
        b.prop = PropProb::new(resample);
        b.terms = terms;
        b.sqr_csum = sqr_csum;
        b.scale_factor = scale_factor;
        b.prior_vec = prior_vec;
        b.prior_mat = prior_mat;
        b.btype = btype;
        b
    }

    fn empty(trend: bool, prior: BasisPrior, n: usize) -> Self {
        let max_knot = prior.max_knot.max(0) as usize;
        let npad16 = n.div_ceil(16) * 16;
        let extra_left = (prior.min_sep - prior.left).max(0) as usize;
        let extra_right = (prior.min_sep - prior.right).max(0) as usize;
        let left_pad8 = extra_left.div_ceil(8) * 8;
        let right_pad = (npad16 - n).max(extra_right);
        Basis {
            trend,
            prior,
            knots: vec![0; 3 + max_knot + 2],
            order: vec![0; max_knot + 2],
            ks: vec![0; max_knot + 2],
            ke: vec![0; max_knot + 2],
            k: 0,
            kbase: 0,
            n_knot: 0,
            goodvec: vec![0; left_pad8 + n + right_pad],
            good_origin: left_pad8,
            good_num: 0,
            kstopping: 0,
            move_step: 0,
            prop: PropProb {
                birth: 0,
                mov: 0,
                death: 0,
                merge: 0,
            },
            prior_vec: vec![],
            prior_mat: vec![],
            terms: vec![],
            inv_sqr: vec![],
            coeff_a: vec![],
            coeff_b: vec![],
            sqr_csum: vec![],
            scale_factor: vec![],
            btype: 0,
        }
    }

    /// `KNOT[i]`, `i` from -3.
    #[inline]
    pub fn knot(&self, i: i32) -> i32 {
        self.knots[(i + 3) as usize]
    }

    #[inline]
    pub fn set_knot(&mut self, i: i32, v: i32) {
        self.knots[(i + 3) as usize] = v;
    }

    /// `CalcBasisKsKeK_prec012`.
    pub fn calc_ks_ke(&mut self) {
        let mut kc = 1;
        for i in 0..=self.n_knot as usize {
            self.ks[i] = kc;
            kc += if self.trend {
                self.order[i] + 1
            } else {
                2 * self.order[i]
            };
            self.ke[i] = kc - 1;
        }
        self.k = kc - 1;
    }

    /// `GenarateRandomBasis` (DSVT): the minimum number of knots, evenly
    /// spaced, every segment of the middle order.
    pub fn init_knots(&mut self, n: i32) {
        let p = self.prior;
        let rnd_order = ((p.max_order + p.min_order) as f64 / 2.0).ceil() as i32;
        self.n_knot = p.min_knot;
        let nvalid = (n - p.right) - (1 + p.left + 1) + 1;
        let sep = nvalid / self.n_knot.max(1);
        let mut init = 1 + p.left + 1;
        for i in 1..=self.n_knot {
            self.order[(i - 1) as usize] = rnd_order;
            self.set_knot(i - 1, init);
            init += sep;
        }
        self.order[self.n_knot as usize] = rnd_order;
        self.set_knot(-3, 1 + (p.left - p.min_sep));
        self.set_knot(-2, (n + 1) - p.right + p.min_sep);
        self.set_knot(-1, 1);
        self.set_knot(self.n_knot, n + 1);
        self.calc_ks_ke();
    }

    /// Sets `len` values of the good vector from position `pos` (1-based,
    /// possibly in the padding) to `v`.
    fn set_good(&mut self, pos: i64, len: i64, v: u8) {
        let start = self.good_origin as i64 + pos - 1;
        let end = (start + len).min(self.goodvec.len() as i64);
        let start = start.max(0);
        if end > start {
            self.goodvec[start as usize..end as usize].fill(v);
        }
    }

    fn sum_good(&self, npad16: usize) -> i32 {
        self.goodvec[self.good_origin..self.good_origin + npad16]
            .iter()
            .map(|&b| b as i32)
            .sum()
    }

    /// `CvtKnotsToBinVec` (DSVT).
    pub fn knots_to_goodvec(&mut self, n: usize) {
        let o = self.good_origin;
        self.goodvec[o..o + n].fill(1);
        let sep = self.prior.min_sep as i64;
        for i in 1..=self.n_knot {
            let kn = self.knot(i - 1) as i64;
            self.set_good(kn - sep, 2 * sep + 1, 0);
        }
        let (l, r) = (self.prior.left as i64, self.prior.right as i64);
        self.set_good(1, l + 1, 0);
        self.set_good(n as i64 - r + 1, r, 0);
        self.good_num = self.sum_good(n.div_ceil(16) * 16);
    }

    /// The good vector from its origin, `npad16` values (the padding after
    /// the series is zero).
    pub fn good(&self, npad16: usize) -> &[u8] {
        &self.goodvec[self.good_origin..self.good_origin + npad16]
    }

    /// `GenTerms`: the terms of a segment into `x` (columns of `n` values),
    /// zero outside it; returns their number.
    pub fn gen_terms(&self, x: &mut [f32], n: usize, seg: &Seg) -> i32 {
        let nseg = (seg.r2 - seg.r1 + 1) as usize;
        let r1 = (seg.r1 - 1) as usize;
        if self.trend {
            let kterms = (seg.order2 - seg.order1 + 1) as usize;
            x[..kterms * n].fill(0.0);
            let scale = self.inv_sqr[nseg - 1];
            let mut k = 0;
            for order in seg.order1..=seg.order2 {
                let col = &mut x[k * n + r1..k * n + r1 + nseg];
                let t = &self.terms[order as usize * n + r1..order as usize * n + r1 + nseg];
                if self.btype == 1 {
                    col.copy_from_slice(t);
                    if order != 0 {
                        let first = col[0];
                        for v in col.iter_mut() {
                            *v += -first;
                        }
                    }
                } else if order == 0 {
                    col.fill(scale);
                } else if order == 1 {
                    seq(col, self.coeff_a[nseg - 1], self.coeff_b[nseg - 1]);
                } else {
                    col.copy_from_slice(t);
                    normalize_factor(col, scale);
                }
                k += 1;
            }
            k as i32
        } else {
            let kterms = ((seg.order2 - seg.order1 + 1) * 2) as usize;
            x[..kterms * n].fill(0.0);
            let mut k = 0usize;
            for order in seg.order1..=seg.order2 {
                let o = (order - 1) as usize;
                for c in 0..2 {
                    let tcol = (o * 2 + c) * n;
                    let col = &mut x[(k + c) * n + r1..(k + c) * n + r1 + nseg];
                    col.copy_from_slice(&self.terms[tcol + r1..tcol + r1 + nseg]);
                    if self.btype == 0 {
                        let cs = &self.sqr_csum[(o * 2 + c) * (n + 1) + 1..];
                        // season_csum[R2-1] - season_csum[(R1-1)-1], the csum
                        // shifted by one so that index -1 is the leading 0.
                        let hi = cs[(seg.r2 - 1) as usize];
                        let lo = self.sqr_csum[(o * 2 + c) * (n + 1) + (seg.r1 - 1) as usize];
                        let f = (n as f32 / (hi - lo)).sqrt();
                        for v in col.iter_mut() {
                            *v *= f;
                        }
                    }
                }
                k += 2;
            }
            k as i32
        }
    }

    /// `ST_ModelPriorFactor1`: the log ratio of the prior probabilities of the
    /// proposed and current model, with the ratio of the proposal
    /// probabilities of a birth or a death.
    pub fn model_prior(&self, p: &Proposal) -> f32 {
        let div = if self.trend { 1 } else { 2 };
        let kold = self.k / div;
        let knew = (p.k2_new - p.k2_old) / div + kold;
        let nseg_old = self.n_knot + 1;
        let nseg_new = p.n_knot_new + 1;
        let o1 = self.prior.min_order + self.trend as i32;
        let kmax = (self.prior.max_knot + 1) * (self.prior.max_order + self.trend as i32);
        let mat = |i: i32| self.prior_mat.get(i as usize).copied().unwrap_or(0.0);
        let vecv = |i: i32| self.prior_vec.get(i as usize).copied().unwrap_or(0.0);
        let mut f0 = vecv(nseg_old - 1) - vecv(nseg_new - 1) + mat((nseg_old - 1) * kmax + kold - 1)
            - mat((nseg_new - 1) * kmax + knew - 1);
        let cplx = ((nseg_new - nseg_old) + (knew - nseg_new * o1) - (kold - nseg_old * o1)) as f32;
        f0 += (self.prior.complexity * cplx) as f64;
        let factor = match p.jump {
            Move::ChOrder => f0 as f32,
            Move::Birth => {
                let ratio = self.good_num as f32 / (self.n_knot + 1) as f32;
                (f0 + (ratio as f64).ln()) as f32
            }
            _ => {
                let ratio = (self.good_num + self.prior.min_sep * 2) as f32 / self.n_knot as f32;
                (f0 - (ratio as f64).ln()) as f32
            }
        };
        factor
    }

    /// `DSVT_UpdateGoodVecForNewTerm`: the good vector, the knots and the
    /// orders after an accepted move (with `KNOT[-1]` and `KNOT[nKnot]` the
    /// fake start and end, as the caller sets them).
    pub fn update_knots(&mut self, p: &Proposal, npad16: usize) {
        let sep = self.prior.min_sep as i64;
        let new_knot = p.new_knot as i64;
        let idx = p.new_idx;
        match p.jump {
            Move::Birth => self.set_good(new_knot - sep, 2 * sep + 1, 0),
            Move::Death => {
                let old = self.knot(idx - 1) as i64;
                let r1 = self.knot(idx - 2) as i64;
                let r2 = self.knot(idx) as i64 - 1;
                self.set_good(old - sep, 2 * sep + 1, 1);
                self.set_good(r1, sep + 1, 0);
                self.set_good(r2 - sep + 1, sep, 0);
            }
            Move::Move => {
                let old = self.knot(idx - 1) as i64;
                let r1 = self.knot(idx - 2) as i64;
                let r2 = self.knot(idx) as i64 - 1;
                self.set_good(old - sep, 2 * sep + 1, 1);
                self.set_good(new_knot - sep, 2 * sep + 1, 0);
                self.set_good(r1, sep + 1, 0);
                self.set_good(r2 - sep + 1, sep, 0);
            }
            Move::Merge => {
                let r1 = self.knot(idx - 2) as i64;
                let r2 = self.knot(idx + 1) as i64 - 1;
                let old = self.knot(idx - 1) as i64;
                self.set_good(old - sep, 2 * sep + 1, 1);
                let old = self.knot(idx) as i64;
                self.set_good(old - sep, 2 * sep + 1, 1);
                self.set_good(new_knot - sep, 2 * sep + 1, 0);
                self.set_good(r1, sep + 1, 0);
                self.set_good(r2 - sep + 1, sep, 0);
            }
            _ => {}
        }
        if p.jump != Move::ChOrder {
            self.good_num = self.sum_good(npad16);
        }
        let nk = self.n_knot;
        let base = 3usize; // knots[base + i] is KNOT[i]
        match p.jump {
            Move::Birth => {
                // InsertNewElem(knotList, nKnot + 1, newIdx, newKnot); RepeatElem(orderList, ...).
                let n = (nk + 1) as usize;
                let ni = idx as usize;
                let mut i = n;
                while i >= ni {
                    self.knots[base + i] = self.knots[base + i - 1];
                    i -= 1;
                }
                self.knots[base + ni - 1] = p.new_knot;
                let mut i = n;
                while i >= ni {
                    self.order[i] = self.order[i - 1];
                    i -= 1;
                }
            }
            Move::Death => {
                let n = (nk + 1) as usize;
                let ni = idx as usize;
                self.knots.copy_within(base + ni..base + n, base + ni - 1);
                self.order.copy_within(ni..n, ni - 1);
            }
            Move::Merge => {
                let n = (nk + 1) as usize;
                let ni = idx as usize;
                self.knots[base + ni - 1] = p.new_knot;
                // memmove(dst + newIdx, dst + newIdx + 1, n - (newIdx + 2) + 1)
                let cnt = n + 1 - (ni + 2);
                self.knots.copy_within(base + ni + 1..base + ni + 1 + cnt, base + ni);
                // DeleteElem(orderList, nKnot + 1, newIdx + 1)
                self.order.copy_within(ni + 1..n, ni);
            }
            Move::Move => self.knots[base + idx as usize - 1] = p.new_knot,
            Move::ChOrder => self.order[idx as usize - 1] = p.new_order,
            Move::NoChange => {}
        }
        self.n_knot = p.n_knot_new;
    }
}
