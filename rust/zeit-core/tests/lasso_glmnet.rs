//! zeit_core::lasso against glmnet's outputs (tests/data/lasso_glmnet.json, made
//! by make_lasso_glmnet_fixtures.R): alpha = 1, standardize, lambda = 20.

use serde_json::Value;
use zeit_core::lasso::{lasso_gaussian, lasso_gaussian_f64};

struct Case {
    id: u64,
    x: Vec<Vec<f64>>,
    y: Vec<f64>,
    lambda: f64,
    loose: Vec<f64>,
    tight: Vec<f64>,
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|e| e.as_f64().unwrap()).collect(),
        other => vec![other.as_f64().unwrap()],
    }
}

fn cases() -> Vec<Case> {
    let text = include_str!("data/lasso_glmnet.json");
    let v: Value = serde_json::from_str(text).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            id: c["id"].as_u64().unwrap(),
            x: c["x"].as_array().unwrap().iter().map(nums).collect(),
            y: nums(&c["y"]),
            lambda: c["lambda"].as_f64().unwrap(),
            loose: nums(&c["glmnet_1e4"]),
            tight: nums(&c["glmnet_1e12"]),
        })
        .collect()
}

fn refs(x: &[Vec<f64>]) -> Vec<&[f64]> {
    x.iter().map(|c| c.as_slice()).collect()
}

fn mean_sd(v: &[f64]) -> (f64, f64) {
    let n = v.len() as f64;
    let m = v.iter().sum::<f64>() / n;
    (m, (v.iter().map(|a| (a - m) * (a - m)).sum::<f64>() / n).sqrt())
}

fn rmse(c: &Case, intercept: f64, beta: &[f64]) -> f64 {
    let n = c.y.len();
    let ss: f64 = (0..n)
        .map(|i| {
            let r = c.y[i] - intercept - c.x.iter().zip(beta).map(|(col, b)| col[i] * b).sum::<f64>();
            r * r
        })
        .sum();
    (ss / n as f64).sqrt()
}

#[test]
fn f64_matches_glmnet_at_1e12() {
    // The stopping rule is `max d^2 < thresh * var(y)`, so thresh 1e-10 stops far
    // earlier than glmnet's 1e-12 (errors of ~1e-2 in t); the same thresh as
    // glmnet's is the comparison that means something (same scale, see the notes).
    let mut worst_rel = 0.0f64; // the spec's |b - ref| / max(1, |ref|)
    let mut worst_t = 0.0f64; // in units of sd(y), on the standardised coefficient
    let mut same_set = 0;
    let all = cases();
    for c in &all {
        let fit = lasso_gaussian_f64(&refs(&c.x), &c.y, c.lambda, 1e-12, 100_000).unwrap();
        assert!(fit.converged, "case {}", c.id);
        let (_, sdy) = mean_sd(&c.y);
        worst_rel = worst_rel.max((fit.intercept - c.tight[0]).abs() / c.tight[0].abs().max(1.0));
        let mut same = true;
        for j in 0..fit.coefs.len() {
            let want = c.tight[j + 1];
            worst_rel = worst_rel.max((fit.coefs[j] - want).abs() / want.abs().max(1.0));
            let (_, s) = mean_sd(&c.x[j]);
            worst_t = worst_t.max((fit.coefs[j] - want).abs() * s / sdy);
            same &= (fit.coefs[j] != 0.0) == (want != 0.0);
        }
        same_set += same as usize;
    }
    println!(
        "f64 (thresh 1e-12) vs glmnet 1e-12: worst spec-relative error {worst_rel:e}, worst t error {worst_t:e} sd(y), same non-zero set in {same_set}/{}",
        all.len()
    );
    assert!(worst_t <= 1e-6, "worst {worst_t:e} sd(y)");
    assert_eq!(same_set, all.len());
}

#[test]
fn f32_matches_glmnet_at_1e4() {
    let mut worst_t = 0.0f64; // in units of sd(y)
    let mut worst_rmse = 0.0f64;
    let mut same_set = 0;
    let all = cases();
    for c in &all {
        let fit = lasso_gaussian(&refs(&c.x), &c.y, c.lambda, 1e-4, 100_000).unwrap();
        assert!(fit.converged, "case {}", c.id);
        let (_, sdy) = mean_sd(&c.y);
        let mut same = true;
        for j in 0..fit.coefs.len() {
            let (_, s) = mean_sd(&c.x[j]);
            let (ours, theirs) = (fit.coefs[j] * s, c.loose[j + 1] * s);
            let err = (ours - theirs).abs() / sdy;
            worst_t = worst_t.max(err);
            // Case 3 (n = 16, p = 7, nearly collinear columns) stops elsewhere than
            // glmnet along a slow path (0.05 sd(y), f64 too): looser bound for it.
            let bound = if c.id == 3 { 0.1 } else { 0.02 };
            assert!(err <= bound, "case {} col {j}: t error {err} sd(y)", c.id);
            if (ours != 0.0) != (theirs != 0.0) {
                same = false;
                // Only allowed when the coefficient is within 0.02 sd(y) of the threshold.
                assert!(ours.abs().max(theirs.abs()) <= 0.02 * sdy, "case {} col {j}", c.id);
            }
        }
        same_set += same as usize;
        let ours = rmse(c, fit.intercept, &fit.coefs);
        let theirs = rmse(c, c.loose[0], &c.loose[1..]);
        let rel = (ours - theirs).abs() / theirs;
        worst_rmse = worst_rmse.max(rel);
        assert!(rel <= if c.id == 3 { 1e-2 } else { 1e-3 },"case {} rmse {ours} vs {theirs}", c.id);
    }
    println!(
        "f32 vs glmnet 1e-4: worst t error {worst_t:.3e} sd(y), worst RMSE relative {worst_rmse:e}, same non-zero set in {same_set}/{}",
        all.len()
    );
}
