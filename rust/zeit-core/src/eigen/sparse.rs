//! Eigen 3.4's SimplicialLLT<SparseMatrix<double>> (lower, AMD ordering) of a
//! symmetric sparse matrix, with the same operations in the same order as the
//! C++ build: the approximate minimum degree ordering of
//! OrderingMethods/Amd.h, the permuted upper triangle as
//! permute_symm_to_symm lays it out, the up-looking factorization of
//! SimplicialCholesky_impl.h and the two sparse triangular solves.

// The index juggling is CSparse's, kept as it is.
#![allow(clippy::needless_range_loop)]

/// A square sparse matrix in compressed columns, row indices sorted.
#[derive(Clone, Debug)]
pub struct Csc {
    pub n: usize,
    pub outer: Vec<usize>,
    pub inner: Vec<usize>,
    pub values: Vec<f64>,
}

#[inline(always)]
fn flip(i: isize) -> isize {
    -i - 2
}

/// cs_wclear.
fn wclear(mut mark: isize, lemax: isize, w: &mut [isize], n: usize) -> isize {
    if mark < 2 || mark + lemax < 0 {
        for k in 0..n {
            if w[k] != 0 {
                w[k] = 1;
            }
        }
        mark = 2;
    }
    mark
}

/// cs_tdfs: postorder of the tree rooted at `j` from `k` on.
fn tdfs(j: isize, mut k: isize, head: &mut [isize], next: &[isize], post: &mut [isize], stack: &mut [isize]) -> isize {
    let mut top: isize = 0;
    stack[0] = j;
    while top >= 0 {
        let p = stack[top as usize];
        let i = head[p as usize];
        if i == -1 {
            top -= 1;
            post[k as usize] = p;
            k += 1;
        } else {
            head[p as usize] = next[i as usize];
            top += 1;
            stack[top as usize] = i;
        }
    }
    k
}

/// internal::minimum_degree_ordering of the pattern of a symmetric matrix
/// (both triangles and the diagonal, rows sorted): the inverse permutation
/// (AMDOrdering's output, SimplicialCholesky's m_Pinv).
pub fn amd(outer: &[usize], inner: &[usize], n: usize) -> Vec<usize> {
    let ni = n as isize;
    let mut dense = std::cmp::max(16, (10.0 * (n as f64).sqrt()) as isize);
    dense = std::cmp::min(ni - 2, dense);

    let mut cnz = inner.len() as isize;
    let t = cnz + cnz / 5 + 2 * ni;
    let mut cp: Vec<isize> = outer.iter().map(|&v| v as isize).collect();
    let mut ci: Vec<isize> = vec![0; t as usize];
    for (d, &s) in ci.iter_mut().zip(inner) {
        *d = s as isize;
    }
    let nn = n + 1;
    let mut len = vec![0isize; nn];
    let mut nv = vec![0isize; nn];
    let mut next = vec![0isize; nn];
    let mut head = vec![0isize; nn];
    let mut elen = vec![0isize; nn];
    let mut degree = vec![0isize; nn];
    let mut w = vec![0isize; nn];
    let mut hhead = vec![0isize; nn];
    let mut last = vec![0isize; nn]; // the permutation's storage in the C++

    for k in 0..n {
        len[k] = cp[k + 1] - cp[k];
    }
    len[n] = 0;
    let nzmax = t;
    for i in 0..=n {
        head[i] = -1;
        last[i] = -1;
        next[i] = -1;
        hhead[i] = -1;
        nv[i] = 1;
        w[i] = 1;
        elen[i] = 0;
        degree[i] = len[i];
    }
    let mut mark = wclear(0, 0, &mut w, n);

    let mut nel: isize = 0;
    for i in 0..n {
        let has_diag = (cp[i]..cp[i + 1]).any(|p| ci[p as usize] == i as isize);
        let d = degree[i];
        if d == 1 && has_diag {
            elen[i] = -2;
            nel += 1;
            cp[i] = -1;
            w[i] = 0;
        } else if d > dense || !has_diag {
            nv[i] = 0;
            elen[i] = -1;
            nel += 1;
            cp[i] = flip(ni);
            nv[n] += 1;
        } else {
            if head[d as usize] != -1 {
                last[head[d as usize] as usize] = i as isize;
            }
            next[i] = head[d as usize];
            head[d as usize] = i as isize;
        }
    }

    elen[n] = -2;
    cp[n] = -1;
    w[n] = 0;

    let (mut lemax, mut mindeg): (isize, isize) = (0, 0);
    while nel < ni {
        // select a node of minimum approximate degree
        let mut k: isize = -1;
        while mindeg < ni {
            k = head[mindeg as usize];
            if k != -1 {
                break;
            }
            mindeg += 1;
        }
        let ku = k as usize;
        if next[ku] != -1 {
            last[next[ku] as usize] = -1;
        }
        head[mindeg as usize] = next[ku];
        let elenk = elen[ku];
        let mut nvk = nv[ku];
        nel += nvk;

        // garbage collection
        if elenk > 0 && cnz + mindeg >= nzmax {
            for j in 0..n {
                let p = cp[j];
                if p >= 0 {
                    cp[j] = ci[p as usize];
                    ci[p as usize] = flip(j as isize);
                }
            }
            let (mut q, mut p) = (0isize, 0isize);
            while p < cnz {
                let j = flip(ci[p as usize]);
                p += 1;
                if j >= 0 {
                    ci[q as usize] = cp[j as usize];
                    cp[j as usize] = q;
                    q += 1;
                    for _ in 0..len[j as usize] - 1 {
                        ci[q as usize] = ci[p as usize];
                        q += 1;
                        p += 1;
                    }
                }
            }
            cnz = q;
        }

        // construct the new element
        let mut dk: isize = 0;
        nv[ku] = -nvk;
        let mut p = cp[ku];
        let pk1 = if elenk == 0 { p } else { cnz };
        let mut pk2 = pk1;
        for k1 in 1..=elenk + 1 {
            let (e, mut pj, ln);
            if k1 > elenk {
                e = k;
                pj = p;
                ln = len[ku] - elenk;
            } else {
                e = ci[p as usize];
                p += 1;
                pj = cp[e as usize];
                ln = len[e as usize];
            }
            for _ in 1..=ln {
                let i = ci[pj as usize];
                pj += 1;
                let iu = i as usize;
                let nvi = nv[iu];
                if nvi <= 0 {
                    continue;
                }
                dk += nvi;
                nv[iu] = -nvi;
                ci[pk2 as usize] = i;
                pk2 += 1;
                if next[iu] != -1 {
                    last[next[iu] as usize] = last[iu];
                }
                if last[iu] != -1 {
                    next[last[iu] as usize] = next[iu];
                } else {
                    head[degree[iu] as usize] = next[iu];
                }
            }
            if e != k {
                cp[e as usize] = flip(k);
                w[e as usize] = 0;
            }
        }
        if elenk != 0 {
            cnz = pk2;
        }
        degree[ku] = dk;
        cp[ku] = pk1;
        len[ku] = pk2 - pk1;
        elen[ku] = -2;

        // set differences
        mark = wclear(mark, lemax, &mut w, n);
        for pk in pk1..pk2 {
            let i = ci[pk as usize] as usize;
            let eln = elen[i];
            if eln <= 0 {
                continue;
            }
            let nvi = -nv[i];
            let wnvi = mark - nvi;
            let mut p = cp[i];
            while p < cp[i] + eln {
                let e = ci[p as usize] as usize;
                if w[e] >= mark {
                    w[e] -= nvi;
                } else if w[e] != 0 {
                    w[e] = degree[e] + wnvi;
                }
                p += 1;
            }
        }

        // degree update
        for pk in pk1..pk2 {
            let i = ci[pk as usize] as usize;
            let p1 = cp[i];
            let p2 = p1 + elen[i] - 1;
            let mut pn = p1;
            let (mut h, mut d): (isize, isize) = (0, 0);
            let mut p = p1;
            while p <= p2 {
                let e = ci[p as usize];
                let eu = e as usize;
                if w[eu] != 0 {
                    let dext = w[eu] - mark;
                    if dext > 0 {
                        d += dext;
                        ci[pn as usize] = e;
                        pn += 1;
                        h += e;
                    } else {
                        cp[eu] = flip(k);
                        w[eu] = 0;
                    }
                }
                p += 1;
            }
            elen[i] = pn - p1 + 1;
            let p3 = pn;
            let p4 = p1 + len[i];
            let mut p = p2 + 1;
            while p < p4 {
                let j = ci[p as usize];
                let nvj = nv[j as usize];
                p += 1;
                if nvj <= 0 {
                    continue;
                }
                d += nvj;
                ci[pn as usize] = j;
                pn += 1;
                h += j;
            }
            if d == 0 {
                cp[i] = flip(k);
                let nvi = -nv[i];
                dk -= nvi;
                nvk += nvi;
                nel += nvi;
                nv[i] = 0;
                elen[i] = -1;
            } else {
                degree[i] = std::cmp::min(degree[i], d);
                ci[pn as usize] = ci[p3 as usize];
                ci[p3 as usize] = ci[p1 as usize];
                ci[p1 as usize] = k;
                len[i] = pn - p1 + 1;
                h %= ni;
                next[i] = hhead[h as usize];
                hhead[h as usize] = i as isize;
                last[i] = h;
            }
        }
        degree[ku] = dk;
        lemax = std::cmp::max(lemax, dk);
        mark = wclear(mark + lemax, lemax, &mut w, n);

        // supernode detection
        for pk in pk1..pk2 {
            let i0 = ci[pk as usize] as usize;
            if nv[i0] >= 0 {
                continue;
            }
            let h = last[i0];
            let mut i = hhead[h as usize];
            hhead[h as usize] = -1;
            while i != -1 && next[i as usize] != -1 {
                let iu = i as usize;
                let ln = len[iu];
                let eln = elen[iu];
                let mut p = cp[iu] + 1;
                while p < cp[iu] + ln {
                    w[ci[p as usize] as usize] = mark;
                    p += 1;
                }
                let mut jlast = i;
                let mut j = next[iu];
                while j != -1 {
                    let ju = j as usize;
                    let mut ok = len[ju] == ln && elen[ju] == eln;
                    let mut p = cp[ju] + 1;
                    while ok && p < cp[ju] + ln {
                        if w[ci[p as usize] as usize] != mark {
                            ok = false;
                        }
                        p += 1;
                    }
                    if ok {
                        cp[ju] = flip(i);
                        nv[iu] += nv[ju];
                        nv[ju] = 0;
                        elen[ju] = -1;
                        j = next[ju];
                        next[jlast as usize] = j;
                    } else {
                        jlast = j;
                        j = next[ju];
                    }
                }
                i = next[iu];
                mark += 1;
            }
        }

        // finalize the new element
        let mut p = pk1;
        for pk in pk1..pk2 {
            let i = ci[pk as usize];
            let iu = i as usize;
            let nvi = -nv[iu];
            if nvi <= 0 {
                continue;
            }
            nv[iu] = nvi;
            let mut d = degree[iu] + dk - nvi;
            d = std::cmp::min(d, ni - nel - nvi);
            if head[d as usize] != -1 {
                last[head[d as usize] as usize] = i;
            }
            next[iu] = head[d as usize];
            last[iu] = -1;
            head[d as usize] = i;
            mindeg = std::cmp::min(mindeg, d);
            degree[iu] = d;
            ci[p as usize] = i;
            p += 1;
        }
        nv[ku] = nvk;
        len[ku] = p - pk1;
        if len[ku] == 0 {
            cp[ku] = -1;
            w[ku] = 0;
        }
        if elenk != 0 {
            cnz = p;
        }
    }

    // postordering
    for i in 0..n {
        cp[i] = flip(cp[i]);
    }
    for j in 0..=n {
        head[j] = -1;
    }
    for j in (0..=n).rev() {
        if nv[j] > 0 {
            continue;
        }
        next[j] = head[cp[j] as usize];
        head[cp[j] as usize] = j as isize;
    }
    for e in (0..=n).rev() {
        if nv[e] <= 0 {
            continue;
        }
        if cp[e] != -1 {
            next[e] = head[cp[e] as usize];
            head[cp[e] as usize] = e as isize;
        }
    }
    let mut post = vec![0isize; nn];
    let mut k: isize = 0;
    for i in 0..=n {
        if cp[i] == -1 {
            k = tdfs(i as isize, k, &mut head, &next, &mut post, &mut w);
        }
    }
    post[..n].iter().map(|&v| v as usize).collect()
}

/// SimplicialLLT<SparseMatrix<double>, Lower, AMDOrdering<int>>: the analysis
/// (analyzePattern) kept for the factorizations of matrices of its pattern.
pub struct SimplicialLlt {
    n: usize,
    /// m_P.indices(): row i of A goes to row p[i] of P A P^T.
    p: Vec<usize>,
    /// m_Pinv.indices().
    pinv: Vec<usize>,
    /// The permuted upper triangle (ap): columns, rows, and for each of its
    /// entries the index of the value of A's lower triangle it holds.
    ap_outer: Vec<usize>,
    ap_inner: Vec<usize>,
    ap_src: Vec<usize>,
    parent: Vec<isize>,
    lp: Vec<usize>,
    // the factor
    li: Vec<usize>,
    lx: Vec<f64>,
    nz: Vec<usize>,
    ap_values: Vec<f64>,
    y: Vec<f64>,
    pattern: Vec<usize>,
    tags: Vec<isize>,
}

impl SimplicialLlt {
    /// analyzePattern of `a`: full symmetric storage, rows sorted.
    pub fn analyze(a: &Csc) -> SimplicialLlt {
        let n = a.n;
        // C = a.selfadjointView<Lower>(), then the AMD of C^T + C: for a
        // symmetric pattern with sorted rows, the pattern of `a` itself
        let pinv = amd(&a.outer, &a.inner, n);
        let mut p = vec![0usize; n];
        for (i, &v) in pinv.iter().enumerate() {
            p[v] = i;
        }

        // ap = a.selfadjointView<Lower>().twistedBy(P), upper triangle
        let mut count = vec![0usize; n];
        for j in 0..n {
            let jp = p[j];
            for k in a.outer[j]..a.outer[j + 1] {
                let i = a.inner[k];
                if i < j {
                    continue;
                }
                count[p[i].max(jp)] += 1;
            }
        }
        let mut ap_outer = vec![0usize; n + 1];
        for j in 0..n {
            ap_outer[j + 1] = ap_outer[j] + count[j];
        }
        let nnz = ap_outer[n];
        let mut ap_inner = vec![0usize; nnz];
        let mut ap_src = vec![0usize; nnz];
        count[..n].copy_from_slice(&ap_outer[..n]);
        for j in 0..n {
            let jp = p[j];
            for k in a.outer[j]..a.outer[j + 1] {
                let i = a.inner[k];
                if i < j {
                    continue;
                }
                let ip = p[i];
                let dst = &mut count[ip.max(jp)];
                ap_inner[*dst] = ip.min(jp);
                ap_src[*dst] = k;
                *dst += 1;
            }
        }

        // the elimination tree and the column counts of L
        let mut parent = vec![-1isize; n];
        let mut nz = vec![0usize; n];
        let mut tags = vec![0isize; n];
        for k in 0..n {
            parent[k] = -1;
            tags[k] = k as isize;
            nz[k] = 0;
            for q in ap_outer[k]..ap_outer[k + 1] {
                let mut i = ap_inner[q];
                if i < k {
                    while tags[i] != k as isize {
                        if parent[i] == -1 {
                            parent[i] = k as isize;
                        }
                        nz[i] += 1;
                        tags[i] = k as isize;
                        i = parent[i] as usize;
                    }
                }
            }
        }
        let mut lp = vec![0usize; n + 1];
        for k in 0..n {
            lp[k + 1] = lp[k] + nz[k] + 1;
        }
        let lnz = lp[n];
        SimplicialLlt {
            n,
            p,
            pinv,
            ap_outer,
            ap_inner,
            ap_src,
            parent,
            lp,
            li: vec![0; lnz],
            lx: vec![0.0; lnz],
            nz,
            ap_values: vec![0.0; nnz],
            y: vec![0.0; n],
            pattern: vec![0; n],
            tags,
        }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// factorize of a matrix of the analyzed pattern: false when it is not
    /// positive definite (Eigen's NumericalIssue).
    pub fn factorize(&mut self, a: &Csc) -> bool {
        let n = self.n;
        for (v, &s) in self.ap_values.iter_mut().zip(&self.ap_src) {
            *v = a.values[s];
        }
        let (lp, li, lx) = (&self.lp, &mut self.li, &mut self.lx);
        let (y, pattern, tags, nz) = (&mut self.y, &mut self.pattern, &mut self.tags, &mut self.nz);
        for k in 0..n {
            y[k] = 0.0;
            let mut top = n;
            tags[k] = k as isize;
            nz[k] = 0;
            for q in self.ap_outer[k]..self.ap_outer[k + 1] {
                let mut i = self.ap_inner[q];
                if i <= k {
                    y[i] += self.ap_values[q];
                    let mut len = 0;
                    while tags[i] != k as isize {
                        pattern[len] = i;
                        len += 1;
                        tags[i] = k as isize;
                        i = self.parent[i] as usize;
                    }
                    while len > 0 {
                        top -= 1;
                        len -= 1;
                        pattern[top] = pattern[len];
                    }
                }
            }

            // the shift: d * 1 + 0
            let mut d = y[k] * 1.0 + 0.0;
            y[k] = 0.0;
            while top < n {
                let i = pattern[top];
                let mut yi = y[i];
                y[i] = 0.0;
                let l_ki = yi / lx[lp[i]];
                yi = l_ki;
                let p2 = lp[i] + nz[i];
                let mut p = lp[i] + 1;
                while p < p2 {
                    y[li[p]] -= lx[p] * yi;
                    p += 1;
                }
                d -= l_ki * yi;
                li[p] = k;
                lx[p] = l_ki;
                nz[i] += 1;
                top += 1;
            }
            let p = lp[k] + nz[k];
            nz[k] += 1;
            li[p] = k;
            if d <= 0.0 {
                return false;
            }
            lx[p] = d.sqrt();
        }
        true
    }

    /// solve(b) after a successful factorize: P^T L^-T L^-1 P b.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        let n = self.n;
        let mut x = vec![0.0; n];
        for i in 0..n {
            x[self.p[i]] = b[i];
        }
        if self.lp[n] > 0 {
            // L x = x: column by column, the diagonal first in each
            for i in 0..n {
                let mut tmp = x[i];
                if tmp != 0.0 {
                    let (start, end) = (self.lp[i], self.lp[i + 1]);
                    tmp /= self.lx[start];
                    x[i] = tmp;
                    for p in start + 1..end {
                        x[self.li[p]] -= tmp * self.lx[p];
                    }
                }
            }
            // L^T x = x: row i of L^T is column i of L
            for i in (0..n).rev() {
                let mut tmp = x[i];
                let (start, end) = (self.lp[i], self.lp[i + 1]);
                let l_ii = self.lx[start];
                for p in start + 1..end {
                    tmp -= self.lx[p] * x[self.li[p]];
                }
                x[i] = tmp / l_ii;
            }
        }
        let mut out = vec![0.0; n];
        for i in 0..n {
            out[self.pinv[i]] = x[i];
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The pentadiagonal W + lambda D^T D of the Whittaker smoother.
    fn penta(n: usize, lambda: f64) -> Csc {
        let mut outer = vec![0];
        let (mut inner, mut values) = (vec![], vec![]);
        for j in 0..n {
            for i in j.saturating_sub(2)..(j + 3).min(n) {
                inner.push(i);
                values.push(if i == j {
                    1.0 + 6.0 * lambda
                } else if i.abs_diff(j) == 1 {
                    -4.0 * lambda
                } else {
                    lambda
                });
            }
            outer.push(inner.len());
        }
        Csc {
            n,
            outer,
            inner,
            values,
        }
    }

    #[test]
    fn amd_is_a_permutation() {
        for n in [1, 2, 3, 5, 6, 7, 18, 19, 40, 200] {
            let a = penta(n, 2.0);
            let mut p = amd(&a.outer, &a.inner, n);
            p.sort_unstable();
            assert_eq!(p, (0..n).collect::<Vec<_>>(), "n = {n}");
        }
    }

    #[test]
    fn solves() {
        for n in [1, 2, 3, 6, 23, 100] {
            let a = penta(n, 3.0);
            let mut llt = SimplicialLlt::analyze(&a);
            assert!(llt.factorize(&a));
            let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin()).collect();
            // b = A x
            let mut b = vec![0.0; n];
            for j in 0..n {
                for k in a.outer[j]..a.outer[j + 1] {
                    b[a.inner[k]] += a.values[k] * x[j];
                }
            }
            let got = llt.solve(&b);
            for i in 0..n {
                assert!((got[i] - x[i]).abs() < 1e-9, "n = {n}");
            }
        }
    }
}
