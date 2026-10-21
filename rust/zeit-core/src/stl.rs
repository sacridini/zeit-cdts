//! STL (Cleveland et al. 1990), the periodic seasonal window with R's
//! non-robust defaults (what bfast calls: `stl(Yt, "periodic")`): a port of
//! src/stl_decompose.cpp (from R's stats/src/stl.c), same operations in the
//! same order, so the same bits. Indices follow the original's 1-based windows.

// The loops index the series as stl.c does.
#![allow(clippy::needless_range_loop, clippy::too_many_arguments)]

fn nextodd(x: f64) -> i64 {
    let xi = x.round() as i64; // std::lround: half away from zero
    if xi % 2 == 0 {
        xi + 1
    } else {
        xi
    }
}

/// The loess estimate at `xs` over the window [nleft, nright] (1-based), as
/// stl.c's stlest: None when the window carries no weight.
fn loess_est(
    y: &[f64],
    n: usize,
    len: usize,
    ideg: i32,
    xs: f64,
    nleft: usize,
    nright: usize,
    w: &mut [f64],
) -> Option<f64> {
    let range = (n - 1) as f64;
    let mut h = (xs - nleft as f64).max(nright as f64 - xs);
    if len > n {
        h += ((len - n) / 2) as f64;
    }
    let (h9, h1) = (h * 0.999, h * 0.001);
    let mut a = 0.0;
    for j in nleft - 1..nright {
        let r = ((j + 1) as f64 - xs).abs();
        if r <= h9 {
            w[j] = if r <= h1 {
                1.0
            } else {
                let u = r / h;
                let v = 1.0 - u * u * u;
                v * v * v
            };
            a += w[j];
        } else {
            w[j] = 0.0;
        }
    }
    if a <= 0.0 {
        return None;
    }
    for j in nleft - 1..nright {
        w[j] /= a;
    }
    if h > 0.0 && ideg > 0 {
        let mut xbar = 0.0;
        for j in nleft - 1..nright {
            xbar += w[j] * (j + 1) as f64;
        }
        let mut b = xs - xbar;
        let mut c = 0.0;
        for j in nleft - 1..nright {
            let d = (j + 1) as f64 - xbar;
            c += w[j] * d * d;
        }
        if c.sqrt() > range * 0.001 {
            b /= c;
            for j in nleft - 1..nright {
                w[j] *= b * ((j + 1) as f64 - xbar) + 1.0;
            }
        }
    }
    let mut ysum = 0.0;
    for j in nleft - 1..nright {
        ysum += w[j] * y[j];
    }
    Some(ysum)
}

/// The loess smoother over a series (stl.c's stless): computed every `njump`
/// points, linearly interpolated in between.
fn loess_smooth(y: &[f64], n: usize, len: usize, ideg: i32, njump: usize, ys: &mut [f64]) {
    if n < 2 {
        ys[0] = y[0];
        return;
    }
    let mut w = vec![0.0; n];
    let newnj = njump.min(n - 1);
    let (mut nleft, mut nright) = (1usize, 1usize);
    if len >= n {
        nleft = 1;
        nright = n;
        let mut i = 0;
        while i < n {
            ys[i] = loess_est(y, n, len, ideg, (i + 1) as f64, nleft, nright, &mut w).unwrap_or(y[i]);
            i += newnj;
        }
    } else if newnj == 1 {
        let nsh = len.div_ceil(2);
        nleft = 1;
        nright = len;
        for i in 0..n {
            if i + 1 > nsh && nright != n {
                nleft += 1;
                nright += 1;
            }
            ys[i] = loess_est(y, n, len, ideg, (i + 1) as f64, nleft, nright, &mut w).unwrap_or(y[i]);
        }
    } else {
        let nsh = len.div_ceil(2);
        let mut i = 0;
        while i < n {
            if i + 1 < nsh {
                nleft = 1;
                nright = len;
            } else if i >= n - nsh {
                nleft = n - len + 1;
                nright = n;
            } else {
                nleft = i + 1 - nsh + 1;
                nright = len + i + 1 - nsh;
            }
            ys[i] = loess_est(y, n, len, ideg, (i + 1) as f64, nleft, nright, &mut w).unwrap_or(y[i]);
            i += newnj;
        }
    }
    if newnj != 1 {
        let mut i = 0;
        while i + newnj < n {
            let delta = (ys[i + newnj] - ys[i]) / newnj as f64;
            for j in i + 1..i + newnj {
                ys[j] = ys[i] + delta * (j - i) as f64;
            }
            i += newnj;
        }
        let k = (n - 1) / newnj * newnj;
        if k != n - 1 {
            ys[n - 1] = loess_est(y, n, len, ideg, n as f64, nleft, nright, &mut w).unwrap_or(y[n - 1]);
            let delta = (ys[n - 1] - ys[k]) / (n - 1 - k) as f64;
            for j in k + 1..n - 1 {
                ys[j] = ys[k] + delta * (j - k) as f64;
            }
        }
    }
}

/// The running mean of window `len` (stl.c's stlma).
fn moving_average(x: &[f64], n: usize, len: usize, ave: &mut [f64]) {
    let flen = len as f64;
    let mut v = 0.0;
    for &xi in &x[..len] {
        v += xi;
    }
    ave[0] = v / flen;
    let newn = (n + 1).saturating_sub(len);
    if newn > 1 {
        let (mut k, mut m) = (len, 0);
        for j in 1..newn {
            v += x[k] - x[m];
            ave[j] = v / flen;
            k += 1;
            m += 1;
        }
    }
}

/// The low-pass filter: moving averages of np, np and 3 (stl.c's stlfts).
fn low_pass_filter(x: &[f64], n: usize, np: usize, trend: &mut [f64], work: &mut [f64]) {
    moving_average(x, n, np, trend);
    moving_average(trend, n - np + 1, np, work);
    moving_average(work, n - 2 * np + 2, 3, trend);
}

/// The cycle-subseries smoothing (stl.c's stlss), into season[m * np + j].
fn subseries_smooth(
    y: &[f64],
    n: usize,
    np: usize,
    ns: usize,
    isdeg: i32,
    nsjump: usize,
    season: &mut [f64],
    work1: &mut [f64],
    work2: &mut [f64],
    work4: &mut [f64],
) {
    for j in 0..np {
        let k = (n - (j + 1)) / np + 1;
        for i in 0..k {
            work1[i] = y[i * np + j];
        }
        loess_smooth(work1, k, ns, isdeg, nsjump, &mut work2[1..]);
        let nright = ns.min(k);
        work2[0] = loess_est(work1, k, ns, isdeg, 0.0, 1, nright, work4).unwrap_or(work2[1]);
        let nleft = (k as i64 - ns as i64 + 1).max(1) as usize;
        work2[k + 1] = loess_est(work1, k, ns, isdeg, (k + 1) as f64, nleft, k, work4).unwrap_or(work2[k]);
        for m in 0..k + 2 {
            season[m * np + j] = work2[m];
        }
    }
}

/// The inner loop of STL (stl.c's stlstp without the robustness weights).
fn inner_loop(
    y: &[f64],
    np: usize,
    (ns, nt, nl): (usize, usize, usize),
    (isdeg, itdeg, ildeg): (i32, i32, i32),
    (nsjump, ntjump, nljump): (usize, usize, usize),
    niter: usize,
    season: &mut [f64],
    trend: &mut [f64],
) {
    let n = y.len();
    let n2p = n + 2 * np;
    let (mut work, mut work2, mut work3, mut work5) = (vec![0.0; n2p], vec![0.0; n2p], vec![0.0; n2p], vec![0.0; n2p]);
    let mut sub_work1 = vec![0.0; n2p];
    for _ in 0..niter {
        for i in 0..n {
            work[i] = y[i] - trend[i];
        }
        subseries_smooth(
            &work,
            n,
            np,
            ns,
            isdeg,
            nsjump,
            &mut work2,
            &mut sub_work1,
            &mut work3,
            &mut work5,
        );
        low_pass_filter(&work2, n2p, np, &mut work3, &mut work);
        loess_smooth(&work3, n, nl, ildeg, nljump, &mut work);
        for i in 0..n {
            season[i] = work2[np + i] - work[i];
        }
        for i in 0..n {
            work[i] = y[i] - season[i];
        }
        loess_smooth(&work, n, nt, itdeg, ntjump, trend);
    }
}

/// R's `stl(ts(y, frequency = period), s.window = "periodic")` seasonal component.
pub fn periodic_seasonal(y: &[f64], period: i32) -> Result<Vec<f64>, &'static str> {
    let n = y.len();
    if period < 2 || n as i64 <= 2 * period as i64 {
        return Err("stl::periodic_seasonal: series is not periodic or has less than two periods");
    }
    let np = period as usize;
    // "periodic": s.window 10 n + 1 with s.degree 0, as R/stl.R
    let ns0 = 10 * n as i64 + 1;
    let twindow0 = nextodd((1.5 * np as f64 / (1.0 - 1.5 / ns0 as f64)).ceil());
    let lwindow0 = nextodd(np as f64);
    let fix = |w: i64| {
        let w = w.max(3);
        (if w % 2 == 0 { w + 1 } else { w }) as usize
    };
    let (ns, nt, nl) = (fix(ns0), fix(twindow0), fix(lwindow0));
    let jump = |w: usize| ((w as f64 / 10.0).ceil() as usize).max(1);
    let mut season = vec![0.0; n];
    let mut trend = vec![0.0; n];
    inner_loop(
        y,
        np,
        (ns, nt, nl),
        (0, 1, 1),
        (jump(ns), jump(nt), jump(nl)),
        2,
        &mut season,
        &mut trend,
    );

    // periodic post-processing (R/stl.R): the mean per cycle position
    let mut sum = vec![0.0; np];
    let mut cnt = vec![0usize; np];
    for (i, &s) in season.iter().enumerate() {
        sum[i % np] += s;
        cnt[i % np] += 1;
    }
    let mean: Vec<f64> = (0..np).map(|p| sum[p] / cnt[p] as f64).collect();
    Ok((0..n).map(|i| mean[i % np]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_a_pure_season() {
        let period = 12;
        let season: Vec<f64> = (0..period).map(|i| (i as f64 * 0.5).sin()).collect();
        let y: Vec<f64> = (0..10 * period).map(|i| season[i % period] + 0.01 * i as f64).collect();
        let s = periodic_seasonal(&y, period as i32).unwrap();
        let mean: f64 = season.iter().sum::<f64>() / period as f64;
        for i in 0..y.len() {
            assert!(
                (s[i] - (season[i % period] - mean)).abs() < 0.02,
                "{i}: {} vs {}",
                s[i],
                season[i % period] - mean
            );
        }
        assert!(periodic_seasonal(&y[..24], 12).is_err());
    }
}
