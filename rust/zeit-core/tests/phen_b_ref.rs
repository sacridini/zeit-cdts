//! zeit_core::phen curve fits and metrics against the phenofit R package
//! (tests/data/phen_b_ref.json, made by make_phen_b_fixtures.R from 24
//! synthetic series of 46 points). Zeit's optimiser, parametrisations, bounds
//! and metric definitions are not phenofit's, so this measures the agreement
//! (run with `--nocapture` to see it) and checks it stays within realistic
//! bounds.

use serde_json::Value;
use zeit_core::phenology::curves::{evaluate_curve_at, CurveType};
use zeit_core::phenology::fit::{default_max_fev, fit_two_pass};
use zeit_core::phenology::metrics::{compute_gof, extract_metrics, METRIC_NAMES};

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|e| e.as_f64().unwrap_or(f64::NAN)).collect(),
        Value::Null => Vec::new(),
        other => vec![other.as_f64().unwrap_or(f64::NAN)],
    }
}

fn quantile(v: &mut [f64], q: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[((v.len() - 1) as f64 * q).round() as usize]
}

const CURVES: [(&str, CurveType, f64); 6] = [
    ("beck", CurveType::Beck, 0.02),
    ("elmore", CurveType::Elmore, 0.02),
    ("zhang", CurveType::Zhang, 0.02),
    ("gu", CurveType::Gu, 0.03),
    ("ag", CurveType::Ag, 0.03),
    ("klos", CurveType::Klos, 0.03),
];

#[test]
fn agreement_with_phenofit() {
    let v: Value = serde_json::from_str(include_str!("data/phen_b_ref.json")).unwrap();
    let t = nums(&v["t"]);
    let tout = nums(&v["tout"]);
    let cases = v["cases"].as_array().unwrap();
    let w1 = vec![1.0; t.len()];

    for (name, kind, curve_tol) in CURVES {
        let (mut n_fit, mut n_fail, mut n_ref) = (0, 0, 0);
        let mut rel_rms = Vec::new();
        let mut d_rmse = Vec::new();
        let (mut d_sos, mut d_eos, mut d_rsp, mut d_rau) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        let mut met_diffs: Vec<Vec<f64>> = vec![Vec::new(); 13];
        let mut met_nan_mismatch = [0usize; 13];
        for case in cases {
            let y = nums(&case["y"]);
            let Some(r) = case["fits"].get(name).filter(|r| r.is_object()) else {
                continue;
            };
            n_ref += 1;
            let ref_curve = nums(&r["curve"]);
            let mut params = Vec::new();
            let ok = fit_two_pass(&t, &y, &w1, &mut params, kind, default_max_fev(kind), 46, 0.5).is_some();
            if !ok {
                n_fail += 1;
                continue;
            }
            n_fit += 1;
            let ours = evaluate_curve_at(kind, &params, &tout, t[0]);
            let amp = ref_curve.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
                - ref_curve.iter().cloned().fold(f64::INFINITY, f64::min);
            let rms = (ours.iter().zip(&ref_curve).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / ours.len() as f64).sqrt();
            rel_rms.push(rms / amp);
            let at_obs = evaluate_curve_at(kind, &params, &t, t[0]);
            let (_, rmse) = compute_gof(&y, &at_obs, &w1);
            d_rmse.push((rmse - r["rmse"].as_f64().unwrap()).abs());

            if matches!(kind, CurveType::Beck | CurveType::Elmore | CurveType::Zhang) {
                let names: Vec<String> =
                    r["par_names"].as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect();
                let rp = nums(&r["par"]);
                let get = |n: &str| rp[names.iter().position(|x| x == n).unwrap()];
                let (sos, rsp, eos, rau) = match kind {
                    CurveType::Zhang | CurveType::Beck | CurveType::Elmore => (params[2], params[3], params[4], params[5]),
                    _ => unreachable!(),
                };
                d_sos.push((sos + t[0] - get("sos")).abs());
                d_eos.push((eos + t[0] - get("eos")).abs());
                d_rsp.push((rsp / get("rsp") - 1.0).abs());
                d_rau.push((rau / get("rau") - 1.0).abs());
            }

            let ref_m = nums(&r["metrics"]);
            let mine = extract_metrics(&params, kind, &tout);
            for k in 0..13 {
                let (a, b) = (mine[k], ref_m.get(k).copied().unwrap_or(f64::NAN));
                if a.is_finite() && b.is_finite() {
                    met_diffs[k].push((a - b).abs());
                } else if a.is_finite() != b.is_finite() {
                    met_nan_mismatch[k] += 1;
                }
            }
        }
        println!("== {name}: {n_fit}/{n_ref} fitted ({n_fail} failed)");
        println!(
            "   curve RMS/amplitude: median {:.4} p90 {:.4} max {:.4} (limit {curve_tol})",
            quantile(&mut rel_rms.clone(), 0.5),
            quantile(&mut rel_rms.clone(), 0.9),
            quantile(&mut rel_rms.clone(), 1.0)
        );
        println!(
            "   |RMSE zeit - RMSE phenofit|: median {:.5} max {:.5}",
            quantile(&mut d_rmse.clone(), 0.5),
            quantile(&mut d_rmse.clone(), 1.0)
        );
        if !d_sos.is_empty() {
            println!(
                "   |d sos| med {:.2} max {:.2}; |d eos| med {:.2} max {:.2}; rate rel diff rsp med {:.3} max {:.3}, rau med {:.3} max {:.3}",
                quantile(&mut d_sos.clone(), 0.5),
                quantile(&mut d_sos.clone(), 1.0),
                quantile(&mut d_eos.clone(), 0.5),
                quantile(&mut d_eos.clone(), 1.0),
                quantile(&mut d_rsp.clone(), 0.5),
                quantile(&mut d_rsp.clone(), 1.0),
                quantile(&mut d_rau.clone(), 0.5),
                quantile(&mut d_rau.clone(), 1.0)
            );
        }
        for k in 0..13 {
            let d = &met_diffs[k];
            let within3 = d.iter().filter(|&&x| x <= 3.0).count();
            let within8 = d.iter().filter(|&&x| x <= 8.0).count();
            println!(
                "   {:9} n={:2} within 3 d {:2} within 8 d {:2} median {:6.2} max {:7.2}; NaN mismatches {}",
                METRIC_NAMES[k],
                d.len(),
                within3,
                within8,
                quantile(&mut d.clone(), 0.5),
                quantile(&mut d.clone(), 1.0),
                met_nan_mismatch[k]
            );
        }
        assert!(n_ref > 0, "{name}: no reference fits");
        assert!(n_fail == 0, "{name}: {n_fail} fits failed");
        assert!(quantile(&mut rel_rms.clone(), 0.5) < curve_tol, "{name}: curve disagreement");
        // measured: the worst of the 24 series is 0.055 of the amplitude (Gu)
        assert!(quantile(&mut rel_rms.clone(), 1.0) < 0.07, "{name}: a curve far from phenofit's");
        // measured: 0.0026 at most for the plateau curves, 0.014 for Gu/Ag/Klos
        let rmse_limit = if matches!(kind, CurveType::Beck | CurveType::Elmore | CurveType::Zhang) { 0.003 } else { 0.02 };
        assert!(quantile(&mut d_rmse.clone(), 1.0) < rmse_limit, "{name}: RMSE");
        if matches!(kind, CurveType::Beck | CurveType::Zhang) {
            assert!(quantile(&mut d_sos.clone(), 1.0) <= 3.0 && quantile(&mut d_eos.clone(), 1.0) <= 3.0, "{name}: sos/eos");
            assert!(quantile(&mut d_rsp.clone(), 1.0) <= 0.4 && quantile(&mut d_rau.clone(), 1.0) <= 0.4, "{name}: rates");
        } else if kind == CurveType::Elmore {
            // Elmore's m7 trades with the plateau: the dates move more in the worst series
            assert!(quantile(&mut d_sos.clone(), 0.5) <= 1.0 && quantile(&mut d_eos.clone(), 0.5) <= 1.0, "{name}: sos/eos");
            assert!(quantile(&mut d_sos.clone(), 1.0) <= 8.0 && quantile(&mut d_eos.clone(), 1.0) <= 8.0, "{name}: sos/eos max");
        }
        // the dates: medians within 4 days (DER.pos of a flat-topped curve is ill-defined, skipped)
        for k in (0..13).filter(|&k| k != 7) {
            assert!(quantile(&mut met_diffs[k].clone(), 0.5) <= 4.0, "{name}: {} median", METRIC_NAMES[k]);
            assert_eq!(met_nan_mismatch[k], 0, "{name}: {} NaN against a date", METRIC_NAMES[k]);
        }
        // the 50% crossings, the common reference of the season, agree to 3 days nearly always
        let need = if matches!(kind, CurveType::Beck | CurveType::Elmore | CurveType::Zhang) { 24 } else { 21 };
        for k in [2, 3] {
            let within3 = met_diffs[k].iter().filter(|&&x| x <= 3.0).count();
            assert!(within3 >= need, "{name}: {} within 3 days only {within3} times", METRIC_NAMES[k]);
        }
    }
}
