//! Self-Organizing Maps: a port of src/som.cpp, itself an operation-by-operation
//! port of MiniSom 2.3.x (G. Vettigli, MIT): `train_online` is MiniSom.train(),
//! `train_batch` MiniSom.train_batch_offline(). NumPy's pairwise summation for
//! the distances, argmin over the square-rooted distance, the same neighborhood,
//! decay and update expressions, so the same bits as the C++ (and MiniSom).

// The kernels index neurons and features as the C++ does.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

use crate::parallel;
use std::sync::atomic::{AtomicUsize, Ordering};

/// exp from the C runtime, one value at a time (as NumPy's).
#[inline(never)]
fn scalar_exp(v: f64) -> f64 {
    v.exp()
}

/// pow(v, 2.0): NumPy squares (v * v), the correctly rounded value pow gives.
#[inline]
fn square(v: f64) -> f64 {
    v * v
}

/// NumPy's DOUBLE_pairwise_sum.
fn pairwise_sum(a: &[f64]) -> f64 {
    let n = a.len();
    if n < 8 {
        let mut res = 0.0;
        for &v in a {
            res += v;
        }
        return res;
    }
    if n <= 128 {
        let mut r: [f64; 8] = a[..8].try_into().expect("8 values");
        let mut i = 8;
        while i < n - n % 8 {
            for j in 0..8 {
                r[j] += a[i + j];
            }
            i += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while i < n {
            res += a[i];
            i += 1;
        }
        return res;
    }
    let mut n2 = n / 2;
    n2 -= n2 % 8;
    pairwise_sum(&a[..n2]) + pairwise_sum(&a[n2..])
}

/// The squared distance as NumPy rounds it: add.reduce((x - w) * (x - w)).
#[inline]
fn squared_distance(x: &[f64], w: &[f64], buf: &mut [f64]) -> f64 {
    let d_n = x.len();
    if d_n < 8 {
        let mut res = 0.0;
        for d in 0..d_n {
            let diff = x[d] - w[d];
            res += diff * diff;
        }
        return res;
    }
    if d_n <= 128 {
        let mut r = [0.0; 8];
        for j in 0..8 {
            let diff = x[j] - w[j];
            r[j] = diff * diff;
        }
        let mut d = 8;
        while d < d_n - d_n % 8 {
            for j in 0..8 {
                let diff = x[d + j] - w[d + j];
                r[j] += diff * diff;
            }
            d += 8;
        }
        let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        while d < d_n {
            let diff = x[d] - w[d];
            res += diff * diff;
            d += 1;
        }
        return res;
    }
    for d in 0..d_n {
        let diff = x[d] - w[d];
        buf[d] = diff * diff;
    }
    pairwise_sum(&buf[..d_n])
}

/// argmin over sqrt(squared distance), the first index on ties (np.argmin);
/// the square root only when the squared distance improves.
fn find_bmu(x: &[f64], w: &[f64], k_n: usize, buf: &mut [f64]) -> usize {
    let d_n = x.len();
    let (mut bmu, mut best_sq, mut best) = (0, f64::INFINITY, f64::INFINITY);
    for k in 0..k_n {
        let sq = squared_distance(x, &w[k * d_n..(k + 1) * d_n], buf);
        if sq < best_sq {
            let dist = sq.sqrt();
            if dist < best {
                best = dist;
                best_sq = sq;
                bmu = k;
            }
        }
    }
    bmu
}

fn decay_learning_rate(code: i32, lr: f64, t: f64, max_iter: f64) -> f64 {
    match code {
        1 => {
            let c = max_iter / 100.0;
            lr * c / (c + t)
        }
        2 => lr * (1.0 - t / max_iter),
        _ => lr / (1.0 + t / (max_iter / 2.0)),
    }
}

fn decay_sigma(code: i32, sigma: f64, t: f64, max_iter: f64) -> f64 {
    match code {
        1 => {
            let c = (sigma - 1.0) / max_iter;
            sigma / (1.0 + (t * c))
        }
        2 => sigma + (t * (1.0 - sigma) / max_iter),
        _ => sigma / (1.0 + t / (max_iter / 2.0)),
    }
}

/// The map: MiniSom's euclidean neuron coordinates (flattened in (i, j)
/// order), and their distinct values for the separable gaussian.
pub struct Grid {
    x: usize,
    y: usize,
    k: usize,
    xx: Vec<f64>,
    yy: Vec<f64>,
    ux: Vec<f64>,
    uy: Vec<f64>,
    ix: Vec<usize>,
    iy: Vec<usize>,
}

impl Grid {
    pub fn new(x: usize, y: usize, xx: &[f64], yy: &[f64]) -> Result<Grid, &'static str> {
        if xx.len() != x * y || yy.len() != x * y {
            return Err("xx/yy must contain one coordinate per neuron");
        }
        let unique = |v: &[f64]| {
            let mut u = v.to_vec();
            u.sort_by(f64::total_cmp);
            u.dedup();
            let idx = v.iter().map(|&a| u.partition_point(|&b| b < a)).collect::<Vec<_>>();
            (u, idx)
        };
        let (ux, ix) = unique(xx);
        let (uy, iy) = unique(yy);
        Ok(Grid {
            x,
            y,
            k: x * y,
            xx: xx.to_vec(),
            yy: yy.to_vec(),
            ux,
            uy,
            ix,
            iy,
        })
    }

    /// The neighborhood of winner `c` with spread `sigma`, into h (K values).
    fn neighborhood(&self, code: i32, c: usize, sigma: f64, h: &mut [f64], ax: &mut [f64], ay: &mut [f64]) {
        let (ci, cj) = ((c / self.y) as i64, (c % self.y) as i64);
        match code {
            1 => {
                // mexican_hat
                let d = 2.0 * sigma * sigma;
                for k in 0..self.k {
                    let p = square(self.xx[k] - self.xx[c]) + square(self.yy[k] - self.yy[c]);
                    h[k] = scalar_exp(-p / d) * (1.0 - 2.0 / d * p);
                }
            }
            2 => {
                // bubble
                for i in 0..self.x {
                    let fi = i as f64;
                    ax[i] = if fi > ci as f64 - sigma && fi < ci as f64 + sigma {
                        1.0
                    } else {
                        0.0
                    };
                }
                for j in 0..self.y {
                    let fj = j as f64;
                    ay[j] = if fj > cj as f64 - sigma && fj < cj as f64 + sigma {
                        1.0
                    } else {
                        0.0
                    };
                }
                for i in 0..self.x {
                    for j in 0..self.y {
                        h[i * self.y + j] = ax[i] * ay[j];
                    }
                }
            }
            3 => {
                // triangle
                for i in 0..self.x {
                    let v = (-(ci - i as i64).abs()) as f64 + sigma;
                    ax[i] = if v < 0.0 { 0.0 } else { v };
                }
                for j in 0..self.y {
                    let v = (-(cj - j as i64).abs()) as f64 + sigma;
                    ay[j] = if v < 0.0 { 0.0 } else { v };
                }
                for i in 0..self.x {
                    for j in 0..self.y {
                        h[i * self.y + j] = ax[i] * ay[j];
                    }
                }
            }
            _ => {
                // gaussian, separable
                let d = 2.0 * sigma * sigma;
                let (xc, yc) = (self.xx[c], self.yy[c]);
                for m in 0..self.ux.len() {
                    ax[m] = scalar_exp(-square(self.ux[m] - xc) / d);
                }
                for m in 0..self.uy.len() {
                    ay[m] = scalar_exp(-square(self.uy[m] - yc) / d);
                }
                for k in 0..self.k {
                    h[k] = ax[self.ix[k]] * ay[self.iy[k]];
                }
            }
        }
    }

    fn scratch(&self) -> (Vec<f64>, Vec<f64>) {
        (
            vec![0.0; self.ux.len().max(self.x)],
            vec![0.0; self.uy.len().max(self.y)],
        )
    }
}

/// The decays and neighborhood of a training.
#[derive(Clone, Copy, Debug)]
pub struct Schedule {
    pub learning_rate: f64,
    pub sigma: f64,
    pub lr_decay: i32,
    pub sigma_decay: i32,
    pub neighborhood: i32,
}

/// MiniSom.train(): `weights` (K x D) updated in place; `order` the sample of
/// every step (or one epoch's order, with `use_epochs`).
pub fn train_online(
    weights: &mut [f64],
    data: &[f64],
    d_n: usize,
    order: &[usize],
    num_iteration: usize,
    use_epochs: bool,
    s: &Schedule,
    grid: &Grid,
) {
    let k_n = grid.k;
    let epochs = if use_epochs { num_iteration } else { 1 };
    let max_iter = num_iteration as f64;
    let mut buf = vec![0.0; d_n];
    let mut h = vec![0.0; k_n];
    let (mut ax, mut ay) = grid.scratch();
    for e in 0..epochs {
        for (step, &sample) in order.iter().enumerate() {
            let xs = &data[sample * d_n..(sample + 1) * d_n];
            let bmu = find_bmu(xs, weights, k_n, &mut buf);
            // MiniSom's "t": the step, or the epoch with use_epochs
            let t = if use_epochs { e } else { step } as f64;
            let eta = decay_learning_rate(s.lr_decay, s.learning_rate, t, max_iter);
            let sig = decay_sigma(s.sigma_decay, s.sigma, t, max_iter);
            grid.neighborhood(s.neighborhood, bmu, sig, &mut h, &mut ax, &mut ay);
            for k in 0..k_n {
                let gk = h[k] * eta;
                if gk == 0.0 {
                    continue; // w += 0 * (x - w) leaves w unchanged
                }
                let wk = &mut weights[k * d_n..(k + 1) * d_n];
                for d in 0..d_n {
                    wk[d] += gk * (xs[d] - wk[d]);
                }
            }
        }
    }
}

/// Neurons accumulated together by one task in the batch update.
const NEURON_BLOCK: usize = 8;

/// MiniSom.train_batch_offline(): `weights` (K x D) updated in place. Each
/// neuron's sums run in sample order, so the result does not depend on the
/// thread count.
pub fn train_batch(
    weights: &mut [f64],
    data: &[f64],
    d_n: usize,
    num_iteration: usize,
    s: &Schedule,
    grid: &Grid,
    n_jobs: i32,
) {
    let k_n = grid.k;
    let n = data.len() / d_n;
    let threads = parallel::threads(n_jobs);
    let mut bmu = vec![0usize; n];
    let mut gt = vec![0.0; k_n * k_n]; // gt[k * K + c]: the neighborhood at k of winner c
    let mut num = vec![0.0; k_n * d_n];
    let mut den = vec![0.0; k_n];
    let state = BatchState {
        data,
        d_n,
        k_n,
        n,
        num_iteration,
        s,
        grid,
    };
    if threads == 1 {
        let mut buf = vec![0.0; d_n];
        let (mut h, mut is_winner) = (vec![0.0; k_n], vec![false; k_n]);
        for it in 0..num_iteration {
            for (i, b) in bmu.iter_mut().enumerate() {
                *b = find_bmu(&data[i * d_n..(i + 1) * d_n], weights, k_n, &mut buf);
            }
            state.neighborhoods(it, &bmu, &mut gt, &mut h, &mut is_winner);
            let blocks = num.chunks_mut(NEURON_BLOCK * d_n).zip(den.chunks_mut(NEURON_BLOCK));
            for (blk, (num_b, den_b)) in blocks.enumerate() {
                state.accumulate(blk, &bmu, &gt, num_b, den_b);
            }
            state.update(it, weights, &num, &den);
        }
        return;
    }

    // A team for the whole training, as the C++'s OpenMP: the phases of an
    // iteration are too short for a parallel loop each (rayon's workers fall
    // asleep and are woken between them, which cost more than the phases).
    let (w_sh, bmu_sh, gt_sh) = (
        parallel::Shared::new(weights),
        parallel::Shared::new(&mut bmu),
        parallel::Shared::new(&mut gt),
    );
    let (num_sh, den_sh) = (parallel::Shared::new(&mut num), parallel::Shared::new(&mut den));
    let next_sample = AtomicUsize::new(0);
    let next_block = AtomicUsize::new(0);
    let n_blocks = k_n.div_ceil(NEURON_BLOCK);
    parallel::team(threads, |tid, barrier| {
        let mut buf = vec![0.0; d_n];
        let (mut h, mut is_winner) = (vec![0.0; k_n], vec![false; k_n]);
        for it in 0..num_iteration {
            // the winners, in chunks of 512 samples
            loop {
                let start = next_sample.fetch_add(512, Ordering::Relaxed);
                if start >= n {
                    break;
                }
                let end = (start + 512).min(n);
                // SAFETY: the weights are written only by thread 0 between barriers; the
                // chunk of bmu is this thread's alone
                let (w, chunk) = unsafe { (w_sh.slice(), bmu_sh.slice_mut(start..end)) };
                for (i, b) in (start..end).zip(chunk.iter_mut()) {
                    *b = find_bmu(&data[i * d_n..(i + 1) * d_n], w, k_n, &mut buf);
                }
            }
            barrier.wait();
            if tid == 0 {
                next_sample.store(0, Ordering::Relaxed);
                // SAFETY: the other threads wait at the barrier
                unsafe {
                    state.neighborhoods(
                        it,
                        bmu_sh.slice(),
                        gt_sh.slice_mut(0..k_n * k_n),
                        &mut h,
                        &mut is_winner,
                    )
                };
            }
            barrier.wait();
            // the sums, by blocks of neurons
            loop {
                let blk = next_block.fetch_add(1, Ordering::Relaxed);
                if blk >= n_blocks {
                    break;
                }
                let (k0, k1) = (blk * NEURON_BLOCK, ((blk + 1) * NEURON_BLOCK).min(k_n));
                // SAFETY: the block of num and den is this thread's alone; bmu and gt are not
                // written until the next barrier
                unsafe {
                    state.accumulate(
                        blk,
                        bmu_sh.slice(),
                        gt_sh.slice(),
                        num_sh.slice_mut(k0 * d_n..k1 * d_n),
                        den_sh.slice_mut(k0..k1),
                    );
                }
            }
            barrier.wait();
            if tid == 0 {
                next_block.store(0, Ordering::Relaxed);
                // SAFETY: the other threads wait at the barrier
                unsafe { state.update(it, w_sh.slice_mut(0..k_n * d_n), num_sh.slice(), den_sh.slice()) };
            }
            barrier.wait();
        }
    });
}

/// The fixed inputs of a batch training, and its phases.
struct BatchState<'a> {
    data: &'a [f64],
    d_n: usize,
    k_n: usize,
    n: usize,
    num_iteration: usize,
    s: &'a Schedule,
    grid: &'a Grid,
}

impl BatchState<'_> {
    fn decays(&self, it: usize) -> (f64, f64) {
        let (t, max_iter) = (it as f64, self.num_iteration as f64);
        (
            decay_learning_rate(self.s.lr_decay, self.s.learning_rate, t, max_iter),
            decay_sigma(self.s.sigma_decay, self.s.sigma, t, max_iter),
        )
    }

    /// The neighborhood of every winning neuron, into gt[k * K + c].
    fn neighborhoods(&self, it: usize, bmu: &[usize], gt: &mut [f64], h: &mut [f64], is_winner: &mut [bool]) {
        let (_, sig) = self.decays(it);
        let k_n = self.k_n;
        is_winner.fill(false);
        for &b in bmu {
            is_winner[b] = true;
        }
        let (mut ax, mut ay) = self.grid.scratch();
        for c in 0..k_n {
            if !is_winner[c] {
                continue;
            }
            self.grid.neighborhood(self.s.neighborhood, c, sig, h, &mut ax, &mut ay);
            for k in 0..k_n {
                gt[k * k_n + c] = h[k];
            }
        }
    }

    /// numerator += g * sample; denominator += g, in sample order, for the
    /// neurons of block `blk`, summed in buffers of its own then copied.
    fn accumulate(&self, blk: usize, bmu: &[usize], gt: &[f64], num_b: &mut [f64], den_b: &mut [f64]) {
        let (d_n, k_n) = (self.d_n, self.k_n);
        let k0 = blk * NEURON_BLOCK;
        let nk = den_b.len();
        let mut acc = vec![0.0; nk * d_n];
        let mut acc_den = [0.0; NEURON_BLOCK];
        for i in 0..self.n {
            let xs = &self.data[i * d_n..(i + 1) * d_n];
            let c = bmu[i];
            for j in 0..nk {
                let g = gt[(k0 + j) * k_n + c];
                if g == 0.0 {
                    continue; // adds exact zeros
                }
                let a = &mut acc[j * d_n..(j + 1) * d_n];
                for d in 0..d_n {
                    a[d] += g * xs[d];
                }
                acc_den[j] += g;
            }
        }
        num_b.copy_from_slice(&acc);
        den_b.copy_from_slice(&acc_den[..nk]);
    }

    /// The new weights: (1 - lr) w + lr num / den.
    fn update(&self, it: usize, weights: &mut [f64], num: &[f64], den: &[f64]) {
        let (lr, _) = self.decays(it);
        let d_n = self.d_n;
        for k in 0..self.k_n {
            if !(den[k] > 0.0) {
                continue;
            }
            let wk = &mut weights[k * d_n..(k + 1) * d_n];
            let nk = &num[k * d_n..(k + 1) * d_n];
            for d in 0..d_n {
                let new_w = nk[d] / den[k];
                wk[d] = (1.0 - lr) * wk[d] + lr * new_w;
            }
        }
    }
}

/// The flat index (i * y + j) of the best matching unit of every sample.
pub fn predict_bmus(data: &[f64], d_n: usize, weights: &[f64], k_n: usize, n_jobs: i32) -> Vec<i32> {
    let n = data.len().checked_div(d_n).unwrap_or(0);
    let mut out = vec![0i32; n];
    parallel::for_each_row(
        &mut out,
        1,
        n_jobs,
        || vec![0.0; d_n],
        |buf, i, o| o[0] = find_bmu(&data[i * d_n..(i + 1) * d_n], weights, k_n, buf) as i32,
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pairwise_sum_matches_a_plain_sum_on_integers() {
        let a: Vec<f64> = (0..1000).map(|i| i as f64).collect();
        assert_eq!(pairwise_sum(&a), 499500.0);
    }

    #[test]
    fn batch_training_separates_two_clusters() {
        let data: Vec<f64> = (0..200)
            .flat_map(|i| if i % 2 == 0 { [0.0, 0.0] } else { [10.0, 10.0] })
            .collect();
        let grid = Grid::new(1, 2, &[0.0, 1.0], &[0.0, 0.0]).unwrap();
        let mut w = vec![1.0, 1.0, 9.0, 9.0];
        let s = Schedule {
            learning_rate: 0.5,
            sigma: 0.3,
            lr_decay: 0,
            sigma_decay: 0,
            neighborhood: 0,
        };
        for n_jobs in [1, 3] {
            let mut ww = w.clone();
            train_batch(&mut ww, &data, 2, 20, &s, &grid, n_jobs);
            assert!(ww[0] < 1.0 && ww[2] > 9.0, "{ww:?}");
            w = ww;
        }
        assert_eq!(predict_bmus(&[0.5, 0.5, 9.5, 9.5], 2, &w, 2, 1), vec![0, 1]);
    }
}
