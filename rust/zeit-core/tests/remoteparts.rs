//! zeit-core's ports of remotePARTS against the R package (tests/data/make_remoteparts_fixtures.R in
//! zeit's repository writes tests/data/remoteparts_*.json).

use serde_json::Value;
use zeit_core::ar_reml::{self, Gaps};
use zeit_core::gls::{self, Nugget};

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
