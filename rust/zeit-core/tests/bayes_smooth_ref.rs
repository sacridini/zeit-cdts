//! zeit_core::bayes_smooth against bayesEO's bayes_smooth() (tests/data/bayes_smooth_ref.json,
//! made by make_bayes_smooth_fixtures.R): a synthetic cube of 3 classes, 60 x 80.

use serde_json::Value;
use zeit_core::bayes_smooth::{smooth_probs, smooth_probs_with, CountRule, Options};

struct Case {
    name: String,
    window: usize,
    frac: f64,
    smoothness: Vec<f64>,
    output: Vec<f64>,
}

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|e| e.as_f64().unwrap()).collect(),
        other => vec![other.as_f64().unwrap()],
    }
}

fn load() -> (usize, usize, usize, Vec<f64>, Vec<Case>) {
    let v: Value = serde_json::from_str(include_str!("data/bayes_smooth_ref.json")).unwrap();
    let cases = v["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| Case {
            name: c["name"].as_str().unwrap().to_string(),
            window: c["window"].as_u64().unwrap() as usize,
            frac: c["frac"].as_f64().unwrap(),
            smoothness: nums(&c["smoothness"]),
            output: nums(&c["output"]),
        })
        .collect();
    (
        v["n_classes"].as_u64().unwrap() as usize,
        v["nrows"].as_u64().unwrap() as usize,
        v["ncols"].as_u64().unwrap() as usize,
        nums(&v["input"]),
        cases,
    )
}

/// (largest difference over the image, over the interior beyond the window's
/// half, share of the pixels with the same argmax).
fn compare(out: &[f64], reference: &[f64], nk: usize, nr: usize, nc: usize, window: usize) -> (f64, f64, f64) {
    let h = window / 2;
    let (mut all, mut inner, mut same) = (0f64, 0f64, 0usize);
    for r in 0..nr {
        for c in 0..nc {
            let i = r * nc + c;
            let mut best = [0usize; 2];
            for k in 0..nk {
                let d = (out[k * nr * nc + i] - reference[k * nr * nc + i]).abs();
                all = all.max(d);
                if r >= h && r + h < nr && c >= h && c + h < nc {
                    inner = inner.max(d);
                }
                if out[k * nr * nc + i] > out[best[0] * nr * nc + i] {
                    best[0] = k;
                }
                if reference[k * nr * nc + i] > reference[best[1] * nr * nc + i] {
                    best[1] = k;
                }
            }
            same += (best[0] == best[1]) as usize;
        }
    }
    (all, inner, same as f64 / (nr * nc) as f64)
}

#[test]
fn matches_bayeseo() {
    let (nk, nr, nc, input, cases) = load();
    assert_eq!(cases.len(), 3);
    for case in &cases {
        let out = smooth_probs(&input, nk, nr, nc, case.window, &case.smoothness, case.frac, 2);
        let (all, inner, same) = compare(&out, &case.output, nk, nr, nc, case.window);
        println!("{}: max diff {all:.3e} (interior {inner:.3e}), argmax agreement {same:.5}", case.name);
        assert!(all < 2.5e-3, "{}: {all}", case.name);
        assert!(same >= 0.999, "{}: {same}", case.name);
    }
}

/// The alternatives for the choices the method's description leaves open are all
/// further from bayesEO than the defaults (the numbers that decided them: see
/// the report of the clean-room task).
#[test]
fn choices_are_the_closest_to_bayeseo() {
    let (nk, nr, nc, input, cases) = load();
    let base = Options::default();
    let alternatives: Vec<(&str, Options)> = vec![
        ("population variance", Options { sample_variance: false, ..base }),
        ("count floor", Options { count_rule: CountRule::Floor, ..base }),
        ("reflect 101", Options { reflect101: true, ..base }),
        ("eps 1e-6", Options { eps: 1e-6, ..base }),
        ("eps 1e-3", Options { eps: 1e-3, ..base }),
    ];
    for case in &cases {
        let run = |o: &Options| {
            let out = smooth_probs_with(&input, nk, nr, nc, case.window, &case.smoothness, case.frac, 2, o);
            compare(&out, &case.output, nk, nr, nc, case.window).0
        };
        let best = run(&base);
        for (label, opts) in &alternatives {
            let worse = run(opts);
            assert!(worse > 2.0 * best, "{}: {label} {worse} vs default {best}", case.name);
        }
    }
}
