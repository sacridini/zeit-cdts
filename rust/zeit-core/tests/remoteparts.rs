//! zeit-core's ports of remotePARTS against the R package (tests/data/make_remoteparts_fixtures.R in
//! zeit's repository writes tests/data/remoteparts_*.json).

use serde_json::Value;
use zeit_core::ar_reml::{self, Gaps};
use zeit_core::gls::{self, Nugget};
use zeit_core::parts;
use zeit_core::spatial::{self, Correlation};

fn fixture(name: &str) -> Value {
    let path = format!("{}/tests/data/{name}", env!("CARGO_MANIFEST_DIR"));
    serde_json::from_str(&std::fs::read_to_string(&path).expect(&path)).unwrap()
}

/// A vector (jsonlite writes one of length 1 as a number).
fn numbers(v: &Value) -> Vec<f64> {
    match v.as_array() {
        Some(a) => a.iter().map(|e| e.as_f64().unwrap_or(f64::NAN)).collect(),
        None => vec![v.as_f64().unwrap_or(f64::NAN)],
    }
}

fn number(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

#[test]
fn gls_matches_fit_gls() {
    let data = fixture("remoteparts_gls.json");
    let mut worst = [0.0f64; 4];
    for case in data["gls"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (xc, yc) = (numbers(&case["x_coord"]), numbers(&case["y_coord"]));
        let n = xc.len();
        let range = number(&case["range"]);
        // R: exp(-as.matrix(dist(coords)) / range)
        let v: Vec<f64> = (0..n * n)
            .map(|e| {
                let (i, j) = (e % n, e / n);
                let (dx, dy) = (xc[i] - xc[j], yc[i] - yc[j]);
                (-(dx * dx + dy * dy).sqrt() / range).exp()
            })
            .collect();
        let cols: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(numbers).collect();
        let p = cols.len();
        let x = cols.concat();
        let y = numbers(&case["y"]);
        let nugget = case["nugget"].as_f64().map_or(Nugget::Ml, Nugget::Fixed);
        let f = gls::fit(&v, n, &x, p, &vec![1.0; n], 1, &y, nugget, 1).unwrap();
        let r = &case["fit"];
        // the nugget is found to within eps^(1/4) (1.2e-4), on a path that rounding may change
        let d = (f.nugget - number(&r["nugget"])).abs();
        assert!(d < 1e-8, "{name}: nugget {} vs R {}", f.nugget, number(&r["nugget"]));
        worst[0] = worst[0].max(d);
        let (coef, se, p_t) = (numbers(&r["coefficients"]), numbers(&r["se"]), numbers(&r["p"]));
        let mut check = |what: &str, d: f64, slot: usize| {
            assert!(d < 1e-8, "{name}: {what} differs by {d:e}");
            worst[slot] = worst[slot].max(d);
        };
        for j in 0..p {
            check("coef/se", (f.coefficients[j] - coef[j]).abs() / se[j], 1);
            check("se", rel(f.se[j], se[j]), 2);
            check("p", rel(f.p[j], p_t[j]), 3);
        }
        check("sse", rel(f.sse, number(&r["sse"])), 2);
        check("logdet", (f.logdet_v - number(&r["logdet_v"])).abs() / n as f64, 2);
        check("loglik", (f.loglik - number(&r["loglik"])).abs() / n as f64, 2);
        match (&f.null, r.get("null")) {
            (Some(n0), Some(r0)) => {
                check("sse0", rel(n0.sse, number(&r0["sse"])), 2);
                check("loglik0", (n0.loglik - number(&r0["loglik"])).abs() / n as f64, 2);
                check("msr", rel(n0.msr, number(&r0["msr"])), 2);
                check("F", rel(n0.f, number(&r0["f"])), 2);
                check("p(F)", rel(n0.p, number(&r0["p"])), 3);
                assert_eq!([n0.df1 as f64, n0.df2 as f64].to_vec(), numbers(&r0["df"]), "{name}");
                let (c0, s0) = (numbers(&r0["coefficients"]), numbers(&r0["se"]));
                check("coef0/se0", (n0.coefficients[0] - c0[0]).abs() / s0[0], 1);
                check("se0", rel(n0.se[0], s0[0]), 2);
            }
            (None, None) => {}
            _ => panic!("{name}: a null model on one side only"),
        }
    }
    println!("nugget, coef/se, relative, p: {worst:?}");
}

/// The largest difference of each statistic over the cases, for the message of a failure.
#[derive(Default, Debug)]
struct Worst {
    rho: f64,
    coef: f64,
    se: f64,
    t: f64,
    p: f64,
    mse: f64,
    loglik: f64,
}

fn rel(a: f64, b: f64) -> f64 {
    (a - b).abs() / b.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn ar_reml_matches_fit_ar() {
    let data = fixture("remoteparts_ar.json");
    let mut worst = Worst::default();
    for case in data["ar"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = numbers(&case["y"]);
        let cols: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(numbers).collect();
        let q = cols.len();
        let x: Vec<f64> = cols.concat();
        for (gaps, key) in [(Gaps::Ignore, "ignore"), (Gaps::Exact, "exact")] {
            let r = &case[key];
            let f = ar_reml::fit(&y, &x, q, gaps).unwrap_or_else(|| panic!("{name} {key}: no fit"));
            let (rho, coef, se) = (
                r["rho"].as_f64().unwrap(),
                numbers(&r["coefficients"]),
                numbers(&r["se"]),
            );
            let (abs_t, p) = (numbers(&r["abs_t"]), numbers(&r["p"]));
            // rho is found to the optimizer's tolerance (~3e-8), where the likelihood is flat and
            // rounding takes the last steps: R's two computations of a same fit (fitAR, and the exact
            // REML on a series without gaps) differ by 7e-8 in rho and 9e-6 in p. In short series
            // rho goes to the bound, the intercept is not identified (R's two computations of its
            // standard error differ by half), and only the other coefficients are compared.
            let at_bound = rho.abs() > 0.999;
            let d_rho = (f.rho - rho).abs();
            assert!(
                d_rho < if at_bound { 1e-5 } else { 1e-7 },
                "{name} {key}: rho {} vs R {rho}",
                f.rho
            );
            worst.rho = worst.rho.max(d_rho);
            let (tol, tol_p) = if at_bound { (1e-4, 1e-3) } else { (1e-6, 3e-5) };
            for j in usize::from(at_bound)..q {
                // coefficients against their standard errors: the precision that matters
                let d = (f.coefficients[j] - coef[j]).abs() / se[j];
                let checks = [
                    ("coef/se", d, tol, &mut worst.coef),
                    ("se", rel(f.se[j], se[j]), tol, &mut worst.se),
                    ("|t|", rel(f.t[j].abs(), abs_t[j]), tol, &mut worst.t),
                    ("p", rel(f.p[j], p[j]), tol_p, &mut worst.p),
                ];
                for (what, d, tol, w) in checks {
                    assert!(d < tol, "{name} {key}: {what} of coefficient {j} differs by {d:e}");
                    *w = w.max(d);
                }
            }
            let d = rel(f.mse, r["mse"].as_f64().unwrap());
            assert!(d < tol, "{name} {key}: mse differs by {d:e}");
            worst.mse = worst.mse.max(d);
            let d = (f.loglik - r["loglik"].as_f64().unwrap()).abs();
            assert!(d < tol, "{name} {key}: logLik differs by {d:e}");
            worst.loglik = worst.loglik.max(d);
        }
    }
    println!("{worst:?}");
}

/// A column-major R matrix or array written by the fixture's cm().
struct Cm {
    dim: Vec<usize>,
    values: Vec<f64>,
}

fn cm(v: &Value) -> Cm {
    Cm {
        dim: numbers(&v["dim"]).iter().map(|&d| d as usize).collect(),
        values: numbers(&v["values"]),
    }
}

impl Cm {
    fn at(&self, idx: &[usize]) -> f64 {
        let (mut k, mut stride) = (0, 1);
        for (i, d) in idx.iter().zip(&self.dim) {
            k += i * stride;
            stride *= d;
        }
        self.values[k]
    }
}

#[test]
fn fit_cor_matches() {
    let data = fixture("remoteparts_parts.json");
    let fc = &data["fitcor"];
    let res = cm(&fc["resids"]);
    let (n, nt) = (res.dim[0], res.dim[1]);
    let resids: Vec<f64> = (0..n * nt).map(|e| res.at(&[e / nt, e % nt])).collect();
    let (lng, lat) = (numbers(&data["lng"]), numbers(&data["lat"]));
    let (x, y) = (&lng[..n], &lat[..n]);
    let f = spatial::fit_correlation(&resids, nt, x, y, Correlation::Exponential { range: 0.1 }, true, -1).unwrap();
    let Correlation::Exponential { range } = f.cor else {
        panic!()
    };
    let want = number(&fc["exponential"]);
    // nls stops at a relative offset of 1e-5; here the least squares converge further
    assert!(rel(range, want) < 1e-5, "range {range} vs R {want}");
    let start = Correlation::ExponentialPower { range: 0.1, shape: 1.0 };
    let f = spatial::fit_correlation(&resids, nt, x, y, start, true, -1).unwrap();
    let Correlation::ExponentialPower { range, shape } = f.cor else {
        panic!()
    };
    let want = numbers(&fc["exponential_power"]);
    assert!(
        rel(range, want[0]) < 1e-5 && rel(shape, want[1]) < 1e-5,
        "{range} {shape} vs R {want:?}"
    );
}

#[test]
fn part_chisqr_matches() {
    for c in fixture("remoteparts_parts.json")["chisqr"].as_array().unwrap() {
        let (fmean, rssr) = (number(&c["fmean"]), number(&c["rssr"]));
        let (df1, npart) = (number(&c["df1"]) as usize, number(&c["npart"]) as usize);
        let (p, _) = parts::chisqr(fmean, rssr, df1, npart);
        // CompQuadForm's imhof integrates to 1e-6
        let want = number(&c["p"]);
        assert!((p - want).abs() < 1e-6, "{fmean} {rssr} {df1} {npart}: {p} vs R {want}");
    }
}

#[test]
fn fit_gls_partition_matches() {
    let data = fixture("remoteparts_parts.json");
    let (lng, lat, resp) = (numbers(&data["lng"]), numbers(&data["lat"]), numbers(&data["y"]));
    let npix = lng.len();
    let partitions: Vec<usize> = numbers(&data["partitions"]).iter().map(|&i| i as usize).collect();
    let partsize = number(&data["partsize"]) as usize;
    for case in data["parts"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cols: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(numbers).collect();
        let p = cols.len();
        let design = cols.concat();
        let ones = vec![1.0; npix];
        let input = parts::Data {
            x: &lng,
            y: &lat,
            response: &resp,
            design: &design,
            p,
            design0: &ones,
            p0: 1,
        };
        let ml = case["nugget"].is_null();
        let opts = parts::Options {
            cor: Correlation::Exponential { range: 0.1 },
            scaled: true,
            nugget: if ml {
                Nugget::Ml
            } else {
                Nugget::Fixed(number(&case["nugget"]))
            },
            ncross: 6,
            n_jobs: -1,
        };
        let f = parts::fit(&input, &partitions, partsize, &opts).unwrap();
        let close = |what: &str, a: f64, b: f64, tol: f64| {
            let d = (a - b).abs() / b.abs().max(1e-300);
            assert!(d < tol, "{name}: {what} {a} vs R {b} ({d:e})");
        };
        let (coef, se) = (cm(&case["coefficients"]), cm(&case["se"]));
        let nuggets = numbers(&case["nuggets"]);
        let fs = numbers(&case["f"]);
        for (k, part) in f.parts.iter().enumerate() {
            for c in 0..p {
                let d = (part.coefficients[c] - coef.at(&[k, c])).abs() / se.at(&[k, c]);
                assert!(d < 1e-8, "{name}: partition {k} coefficient {c} differs by {d:e} SE");
                close("se", part.se[c], se.at(&[k, c]), 1e-8);
            }
            assert!((part.nugget - nuggets[k]).abs() < 1e-8, "{name}: nugget of {k}");
            if let Some(null) = &part.null {
                close("F", null.f, fs[k], 1e-8);
            }
        }
        // remotePARTS crosses the partitions after the first without their nugget: with an ML
        // nugget only the pairs of the first partition are the same
        let (rcoefs, rssrs, rsses) = (cm(&case["rcoefs"]), numbers(&case["rssrs"]), numbers(&case["rsses"]));
        for (k, c) in f.cross.iter().enumerate() {
            if ml && c.i != 0 {
                continue;
            }
            for a in 0..p {
                for b in 0..p {
                    let (got, want) = (c.rcoef[b * p + a], rcoefs.at(&[k, a, b]));
                    assert!(
                        (got - want).abs() < 1e-8,
                        "{name}: rcoef {k} [{a}, {b}] {got} vs R {want}"
                    );
                }
            }
            if p > 1 {
                close("rSSR", c.rssr, rssrs[k], 1e-7);
            }
            close("rSSE", c.rsse, rsses[k], 1e-7);
        }
        let o = &case["overall"];
        let oc = numbers(&o["coefficients"]);
        for (got, want) in f.coefficients.iter().zip(&oc) {
            close("overall coefficient", *got, *want, 1e-8);
        }
        let dfs = numbers(&o["dfs"]);
        assert_eq!([f.df1 as f64, f.df2 as f64].to_vec(), dfs, "{name}");
        if ml {
            continue;
        }
        if p > 1 {
            close("F mean", f.f, number(&o["f"]), 1e-8);
            close("rSSR mean", f.rssr, number(&o["rssr"]), 1e-7);
        }
        close("rSSE mean", f.rsse, number(&o["rsse"]), 1e-7);
        let (t, covar) = (cm(&o["t"]), cm(&o["covar"]));
        for c in 0..p {
            close("t-test SE", f.t_test.se[c], t.at(&[c, 1]), 1e-7);
            close("t-test t", f.t_test.t[c].abs(), t.at(&[c, 2]).abs(), 1e-7);
            close("t-test p", f.t_test.p[c], t.at(&[c, 3]), 1e-6);
            for d in 0..p {
                close("covar", f.t_test.covar[d * p + c], covar.at(&[c, d]), 1e-7);
            }
        }
        match (f.p_chisqr, o.get("p_chisqr").and_then(Value::as_f64)) {
            (Some(a), Some(b)) => assert!((a - b).abs() < 1e-6, "{name}: chisqr {a} vs R {b}"),
            (None, None) => {}
            other => panic!("{name}: chisqr {other:?}"),
        }
    }
}
