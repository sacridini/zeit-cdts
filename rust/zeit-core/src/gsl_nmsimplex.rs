//! The Nelder-Mead simplex minimizer `gsl_multimin_fminimizer_nmsimplex2` of GSL 2.7 (`multimin/simplex2.c`, Copyright
//! (C) 2007-2009 Brian Gough, 2002 Tuomo Keskitalo and Ivo Alxneit, GPL-3.0-or-later), for a function of ONE variable
//! (n = 1, a simplex of two points: [`NmSimplex1`]), which is how FORCE's water vapour retrieval (`gas-ll.c`) uses it,
//! and for n variables ([`NmSimplex`], with GSL's reference `dnrm2`, `ddot` and `daxpy` written out), which is how the
//! polynomial AOD fit of `aod-ll.c` uses it (n = 3).
//!
//! Ported operation by operation: the BLAS calls of the C (`dscal`, `daxpy`, `ddot`, `dnrm2`) on vectors of one element
//! are the plain IEEE operations written here (GSL's reference CBLAS, no fused multiply-add; `dnrm2` of one element is
//! `fabs`), in the same order, so the iterates equal GSL's bit for bit. The simplex size is the running RMS value GSL
//! keeps (`S2`), as in the C. Where the C would call GSL's error handler (non-finite value of the function when the
//! simplex is set, a failed contraction), which aborts the program unless it is turned off, this port returns an `Err`
//! / `false`.

/// A simplex of two points `x[0], x[1]` of one variable, with the function values `y`.
pub struct NmSimplex1 {
    x1: [f64; 2],
    y1: [f64; 2],
    center: f64,
    s2: f64,
    /// The lowest point of the simplex (GSL's `s->x`).
    pub x: f64,
    /// Its function value (`s->fval`).
    pub fval: f64,
    /// The size of the simplex (`s->size`).
    pub size: f64,
}

const P: f64 = 2.0; // the number of points, n + 1 (`size_t P` of the C, used as a double)

impl NmSimplex1 {
    /// `gsl_multimin_fminimizer_set`: the simplex `{x0, x0 + step}`. `Err` if the function is not finite at either.
    pub fn set(f: &mut impl FnMut(f64) -> f64, x0: f64, step: f64) -> Result<Self, ()> {
        let val = f(x0);
        if !val.is_finite() {
            return Err(());
        }
        let xt = x0 + step;
        let val1 = f(xt);
        if !val1.is_finite() {
            return Err(());
        }
        let mut s = NmSimplex1 { x1: [x0, xt], y1: [val, val1], center: 0.0, s2: 0.0, x: x0, fval: 0.0, size: 0.0 };
        s.compute_center();
        s.size = s.compute_size();
        Ok(s)
    }

    fn compute_center(&mut self) {
        // set_zero, then daxpy(1.0, row) for each point, then dscal(1/P)
        let mut c = 0.0f64;
        for i in 0..2 {
            c += 1.0 * self.x1[i];
        }
        self.center = c * (1.0 / P);
    }

    fn compute_size(&mut self) -> f64 {
        let mut ss = 0.0f64;
        for i in 0..2 {
            let s = self.x1[i] + (-1.0) * self.center; // daxpy(-1, center, s)
            let t = s.abs(); // dnrm2 of one element
            ss += t * t;
        }
        self.s2 = ss / P;
        (ss / P).sqrt()
    }

    /// `try_corner_move`: the corner `corner` moved by `coeff`, and the function there.
    fn try_corner_move(&self, coeff: f64, corner: usize, f: &mut impl FnMut(f64) -> f64) -> (f64, f64) {
        let alpha = (1.0 - coeff) * P / (P - 1.0);
        let beta = (P * coeff - 1.0) / (P - 1.0);
        let mut xc = self.center; // memcpy
        xc *= alpha; // dscal
        xc += beta * self.x1[corner]; // daxpy
        (xc, f(xc))
    }

    fn update_point(&mut self, i: usize, x: f64, val: f64) {
        let x_orig = self.x1[i];
        let delta = x + (-1.0) * x_orig;
        let xmc = x_orig + (-1.0) * self.center;
        let d = delta.abs();
        let xmcd = 0.0 + xmc * delta; // ddot
        self.s2 += (2.0 / P) * xmcd + ((P - 1.0) / P) * (d * d / P);
        let alpha = 1.0 / P;
        self.center += (-alpha) * x_orig;
        self.center += alpha * x;
        self.x1[i] = x;
        self.y1[i] = val;
    }

    /// `contract_by_best`: false if the function is not finite at a moved point (GSL_EBADFUNC).
    fn contract_by_best(&mut self, best: usize, f: &mut impl FnMut(f64) -> f64) -> bool {
        let mut ok = true;
        for i in 0..2 {
            if i != best {
                self.x1[i] = 0.5 * (self.x1[i] + self.x1[best]);
                let v = f(self.x1[i]);
                self.y1[i] = v;
                if !v.is_finite() {
                    ok = false;
                }
            }
        }
        self.compute_center();
        self.compute_size();
        ok
    }

    /// `gsl_multimin_fminimizer_iterate`: one step; false if GSL would have failed (a contraction onto a non-finite
    /// value), in which case `x`, `fval` and `size` are not updated.
    pub fn iterate(&mut self, f: &mut impl FnMut(f64) -> f64) -> bool {
        // the highest, second highest and lowest point
        let mut dhi = self.y1[0];
        let mut dlo = self.y1[0];
        let (mut hi, mut lo) = (0usize, 0usize);
        let mut ds_hi = self.y1[1];
        let mut s_hi = 1usize;
        for i in 1..2 {
            let val = self.y1[i];
            if val < dlo {
                dlo = val;
                lo = i;
            } else if val > dhi {
                ds_hi = dhi;
                s_hi = hi;
                dhi = val;
                hi = i;
            } else if val > ds_hi {
                ds_hi = val;
                s_hi = i;
            }
        }
        let _ = (dhi, dlo, ds_hi);

        // reflect the highest point
        let (xc, val) = self.try_corner_move(-1.0, hi, f);
        if val.is_finite() && val < self.y1[lo] {
            // the reflected point is the lowest: try the expansion
            let (xc2, val2) = self.try_corner_move(-2.0, hi, f);
            if val2.is_finite() && val2 < self.y1[lo] {
                self.update_point(hi, xc2, val2);
            } else {
                self.update_point(hi, xc, val);
            }
        } else if !val.is_finite() || val > self.y1[s_hi] {
            if val.is_finite() && val <= self.y1[hi] {
                self.update_point(hi, xc, val);
            }
            // one-dimensional contraction
            let (xc2, val2) = self.try_corner_move(0.5, hi, f);
            if val2.is_finite() && val2 <= self.y1[hi] {
                self.update_point(hi, xc2, val2);
            } else if !self.contract_by_best(lo, f) {
                return false;
            }
        } else {
            self.update_point(hi, xc, val);
        }

        // the lowest point of the simplex (gsl_vector_min_index: the first minimum; a NaN is never one)
        let mut lo = 0usize;
        if self.y1[1] < self.y1[0] {
            lo = 1;
        }
        self.x = self.x1[lo];
        self.fval = self.y1[lo];
        self.size = if self.s2 > 0.0 { self.s2.sqrt() } else { self.compute_size() };
        true
    }
}

/// The loop of FORCE: iterate until the size is below `epsabs` (`gsl_multimin_test_size`) or `max_iter` iterations,
/// stopping at a failed iteration. Returns the lowest point (`s->x`) after the last iteration (the starting point if
/// the function is not finite at the first simplex, which the C would abort on: `None`).
pub fn minimize(f: &mut impl FnMut(f64) -> f64, x0: f64, step: f64, epsabs: f64, max_iter: usize) -> Option<f64> {
    let mut s = NmSimplex1::set(f, x0, step).ok()?;
    let mut iter = 0usize;
    loop {
        iter += 1;
        if !s.iterate(f) {
            break;
        }
        // status = test_size(size, eps): GSL_SUCCESS (0) if size < eps, else GSL_CONTINUE
        if s.size < epsabs || iter >= max_iter {
            break;
        }
    }
    Some(s.x)
}

// ---------------------------------------------------------------------------------------------------------------
// The same minimizer for any number of variables (the polynomial AOD fit of aod-ll.c has three)
// ---------------------------------------------------------------------------------------------------------------

/// GSL's reference CBLAS `dnrm2` (`cblas/source_nrm2_r.h`): the scaled sum of squares (`fabs` for one element).
fn dnrm2(x: &[f64]) -> f64 {
    let n = x.len();
    if n == 0 {
        return 0.0;
    } else if n == 1 {
        return x[0].abs();
    }
    let mut scale = 0.0f64;
    let mut ssq = 1.0f64;
    for &xi in x {
        if xi != 0.0 {
            let ax = xi.abs();
            if scale < ax {
                ssq = 1.0 + ssq * (scale / ax) * (scale / ax);
                scale = ax;
            } else {
                ssq += (ax / scale) * (ax / scale);
            }
        }
    }
    scale * ssq.sqrt()
}

/// `cblas_daxpy` for unit strides: `y += alpha * x` (a zero `alpha` returns at once, as in the C).
fn daxpy(alpha: f64, x: &[f64], y: &mut [f64]) {
    if alpha == 0.0 {
        return;
    }
    for (yi, &xi) in y.iter_mut().zip(x) {
        *yi += alpha * xi;
    }
}

/// `cblas_ddot`: the sum starts at zero and adds the products in order.
fn ddot(x: &[f64], y: &[f64]) -> f64 {
    let mut r = 0.0f64;
    for (&xi, &yi) in x.iter().zip(y) {
        r += xi * yi;
    }
    r
}

/// The simplex of `gsl_multimin_fminimizer_nmsimplex2` for `n` variables (`n + 1` points), operation by operation
/// as in [`NmSimplex1`] (which is this with `n = 1`).
pub struct NmSimplex {
    n: usize,
    /// The corner points, row-major, `n + 1` rows of `n` (`x1`).
    x1: Vec<f64>,
    y1: Vec<f64>,
    center: Vec<f64>,
    s2: f64,
    /// The lowest point of the simplex (`s->x`; the starting point until the first successful iteration).
    pub x: Vec<f64>,
    /// Its function value (`s->fval`).
    pub fval: f64,
    /// The size of the simplex (`s->size`).
    pub size: f64,
}

impl NmSimplex {
    fn p(&self) -> f64 {
        (self.n + 1) as f64
    }

    fn row(&self, i: usize) -> &[f64] {
        &self.x1[i * self.n..(i + 1) * self.n]
    }

    /// `gsl_multimin_fminimizer_set` (`nmsimplex_set`): the simplex `{x0, x0 + step_i e_i}`. `Err` if the function is
    /// not finite at one of the points (`GSL_EBADFUNC`, which aborts unless the error handler is off).
    pub fn set(f: &mut impl FnMut(&[f64]) -> f64, x0: &[f64], step: &[f64]) -> Result<Self, ()> {
        let n = x0.len();
        assert!(n > 0 && step.len() == n);
        let mut x1 = vec![0.0f64; (n + 1) * n];
        let mut y1 = vec![0.0f64; n + 1];
        let val = f(x0);
        if !val.is_finite() {
            return Err(());
        }
        x1[..n].copy_from_slice(x0);
        y1[0] = val;
        for i in 0..n {
            let mut xtemp = x0.to_vec();
            xtemp[i] = x0[i] + step[i];
            let val = f(&xtemp);
            if !val.is_finite() {
                return Err(());
            }
            x1[(i + 1) * n..(i + 2) * n].copy_from_slice(&xtemp);
            y1[i + 1] = val;
        }
        let mut s = NmSimplex { n, x1, y1, center: vec![0.0; n], s2: 0.0, x: x0.to_vec(), fval: 0.0, size: 0.0 };
        s.compute_center();
        s.size = s.compute_size();
        Ok(s)
    }

    fn compute_center(&mut self) {
        let n = self.n;
        let mut c = vec![0.0f64; n];
        for i in 0..=n {
            daxpy(1.0, self.row(i), &mut c);
        }
        let alpha = 1.0 / self.p();
        for v in c.iter_mut() {
            *v *= alpha;
        }
        self.center = c;
    }

    fn compute_size(&mut self) -> f64 {
        let n = self.n;
        let p = self.p();
        let mut ss = 0.0f64;
        for i in 0..=n {
            let mut s = self.row(i).to_vec();
            daxpy(-1.0, &self.center, &mut s);
            let t = dnrm2(&s);
            ss += t * t;
        }
        self.s2 = ss / p;
        (ss / p).sqrt()
    }

    /// `try_corner_move`: the corner `corner` moved by `coeff`, and the function there.
    fn try_corner_move(&self, coeff: f64, corner: usize, f: &mut impl FnMut(&[f64]) -> f64) -> (Vec<f64>, f64) {
        let p = self.p();
        let alpha = (1.0 - coeff) * p / (p - 1.0);
        let beta = (p * coeff - 1.0) / (p - 1.0);
        let mut xc = self.center.clone();
        for v in xc.iter_mut() {
            *v *= alpha;
        }
        daxpy(beta, self.row(corner), &mut xc);
        let val = f(&xc);
        (xc, val)
    }

    fn update_point(&mut self, i: usize, x: &[f64], val: f64) {
        let n = self.n;
        let p = self.p();
        let x_orig = self.row(i).to_vec();
        let mut delta = x.to_vec();
        daxpy(-1.0, &x_orig, &mut delta);
        let mut xmc = x_orig.clone();
        daxpy(-1.0, &self.center, &mut xmc);
        let d = dnrm2(&delta);
        let xmcd = ddot(&xmc, &delta);
        self.s2 += (2.0 / p) * xmcd + ((p - 1.0) / p) * (d * d / p);
        let alpha = 1.0 / p;
        daxpy(-alpha, &x_orig, &mut self.center);
        daxpy(alpha, x, &mut self.center);
        self.x1[i * n..(i + 1) * n].copy_from_slice(x);
        self.y1[i] = val;
    }

    /// `contract_by_best`: false if the function is not finite at a moved point (`GSL_EBADFUNC`).
    fn contract_by_best(&mut self, best: usize, f: &mut impl FnMut(&[f64]) -> f64) -> bool {
        let n = self.n;
        let mut ok = true;
        for i in 0..=n {
            if i != best {
                for j in 0..n {
                    self.x1[i * n + j] = 0.5 * (self.x1[i * n + j] + self.x1[best * n + j]);
                }
                let xc = self.row(i).to_vec();
                let v = f(&xc);
                self.y1[i] = v;
                if !v.is_finite() {
                    ok = false;
                }
            }
        }
        self.compute_center();
        self.compute_size();
        ok
    }

    /// `gsl_multimin_fminimizer_iterate`: one step; false if GSL would have failed (a contraction onto a non-finite
    /// value), in which case `x`, `fval` and `size` are not updated.
    pub fn iterate(&mut self, f: &mut impl FnMut(&[f64]) -> f64) -> bool {
        let n = self.n + 1;
        // the highest, second highest and lowest point
        let mut dhi = self.y1[0];
        let mut dlo = self.y1[0];
        let (mut hi, mut lo) = (0usize, 0usize);
        let mut ds_hi = self.y1[1];
        let mut s_hi = 1usize;
        for i in 1..n {
            let val = self.y1[i];
            if val < dlo {
                dlo = val;
                lo = i;
            } else if val > dhi {
                ds_hi = dhi;
                s_hi = hi;
                dhi = val;
                hi = i;
            } else if val > ds_hi {
                ds_hi = val;
                s_hi = i;
            }
        }
        let _ = (dhi, dlo, ds_hi);

        // reflect the highest point
        let (xc, val) = self.try_corner_move(-1.0, hi, f);
        if val.is_finite() && val < self.y1[lo] {
            // the reflected point is the lowest: try the expansion
            let (xc2, val2) = self.try_corner_move(-2.0, hi, f);
            if val2.is_finite() && val2 < self.y1[lo] {
                self.update_point(hi, &xc2, val2);
            } else {
                self.update_point(hi, &xc, val);
            }
        } else if !val.is_finite() || val > self.y1[s_hi] {
            if val.is_finite() && val <= self.y1[hi] {
                self.update_point(hi, &xc, val);
            }
            // one-dimensional contraction
            let (xc2, val2) = self.try_corner_move(0.5, hi, f);
            if val2.is_finite() && val2 <= self.y1[hi] {
                self.update_point(hi, &xc2, val2);
            } else if !self.contract_by_best(lo, f) {
                return false;
            }
        } else {
            self.update_point(hi, &xc, val);
        }

        // the lowest point of the simplex (gsl_vector_min_index: the first minimum; a NaN ends the search)
        let mut imin = 0usize;
        let mut min = self.y1[0];
        for i in 0..n {
            let v = self.y1[i];
            if v < min {
                min = v;
                imin = i;
            }
            if v.is_nan() {
                imin = i;
                break;
            }
        }
        self.x = self.row(imin).to_vec();
        self.fval = self.y1[imin];
        self.size = if self.s2 > 0.0 { self.s2.sqrt() } else { self.compute_size() };
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_the_minimum_of_a_parabola() {
        let mut f = |x: f64| (x - 2.5).abs();
        let x = minimize(&mut f, 3.0, 0.25, 1e-2, 100).unwrap();
        assert!((x - 2.5).abs() < 0.02, "{x}");
    }

    #[test]
    fn the_n_dimensional_simplex_is_the_one_dimensional_one_for_n_1() {
        let f = |x: f64| (x - 2.5).abs() + 0.1 * (x - 1.0) * (x - 1.0);
        let mut f1 = |x: f64| f(x);
        let mut fn_ = |v: &[f64]| f(v[0]);
        let mut a = NmSimplex1::set(&mut f1, 3.0, 0.25).unwrap();
        let mut b = NmSimplex::set(&mut fn_, &[3.0], &[0.25]).unwrap();
        assert_eq!(a.size.to_bits(), b.size.to_bits());
        for _ in 0..40 {
            assert!(a.iterate(&mut f1) && b.iterate(&mut fn_));
            assert_eq!((a.x.to_bits(), a.fval.to_bits(), a.size.to_bits()), (b.x[0].to_bits(), b.fval.to_bits(), b.size.to_bits()));
        }
    }

    #[test]
    fn three_variables_converge_on_a_quadratic() {
        // the RMSE of a quadratic in three coefficients, as the polynomial AOD fit minimizes
        let mut f = |v: &[f64]| ((v[0] - 0.3).powi(2) + (v[1] + 1.2).powi(2) + 4.0 * (v[2] + 0.2).powi(2)).sqrt();
        let mut s = NmSimplex::set(&mut f, &[0.0, 0.0, 0.0], &[0.01, 0.01, 0.02]).unwrap();
        let mut it = 0;
        while s.iterate(&mut f) && s.size >= 1e-4 && it < 500 {
            it += 1;
        }
        assert!((s.x[0] - 0.3).abs() < 1e-3 && (s.x[1] + 1.2).abs() < 1e-3 && (s.x[2] + 0.2).abs() < 1e-3, "{:?}", s.x);
        // a non-finite function at the start is refused
        let mut g = |_: &[f64]| f64::NAN;
        assert!(NmSimplex::set(&mut g, &[0.0, 0.0], &[0.1, 0.1]).is_err());
    }

    #[test]
    fn a_non_finite_start_is_none() {
        let mut f = |_x: f64| f64::NAN;
        assert!(minimize(&mut f, 3.0, 0.25, 1e-2, 100).is_none());
    }
}
