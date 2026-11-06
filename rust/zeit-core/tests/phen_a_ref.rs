//! zeit_core::phen smoothing and season split against the phenofit R package
//! (tests/data/phen_a_ref.json, made by make_phen_a_fixtures.R from 200 synthetic
//! series of 3 years of 16-day data). The rules of Zeit are not phenofit's, so
//! this measures the agreement and checks it stays within realistic bounds.

use serde_json::Value;
use zeit_core::phenology::seasons::{split_growing_seasons, SeasonSettings};
use zeit_core::phenology::smooth::{whittaker_envelope, wtsm};

fn nums(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(a) => a.iter().map(|e| e.as_f64().unwrap_or(f64::NAN)).collect(),
        Value::Null => Vec::new(),
        other => vec![other.as_f64().unwrap()],
    }
}

fn ranks(v: &[f64]) -> Vec<f64> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[a].total_cmp(&v[b]));
    let mut r = vec![0.0; v.len()];
    let mut i = 0;
    while i < idx.len() {
        let mut j = i;
        while j + 1 < idx.len() && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        for k in i..=j {
            r[idx[k]] = 0.5 * (i + j) as f64;
        }
        i = j + 1;
    }
    r
}

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
    let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
    for i in 0..a.len() {
        sab += (a[i] - ma) * (b[i] - mb);
        saa += (a[i] - ma).powi(2);
        sbb += (b[i] - mb).powi(2);
    }
    if saa == 0.0 || sbb == 0.0 { 1.0 } else { sab / (saa * sbb).sqrt() }
}

#[test]
fn agreement_with_phenofit() {
    let v: Value = serde_json::from_str(include_str!("data/phen_a_ref.json")).unwrap();
    let npt = v["nptperyear"].as_u64().unwrap() as usize;
    let lambda = v["lambda"].as_f64().unwrap();
    let t = nums(&v["t"]);
    let cases = v["cases"].as_array().unwrap();
    let settings = SeasonSettings::default();

    let (mut same_count, mut off_by_one, mut total) = (0, 0, 0);
    let (mut peak_ok, mut peak_n, mut edge_ok, mut edge_n) = (0, 0, 0, 0);
    let (mut rmse_rel, mut corr_min, mut spearman_min) = (Vec::new(), 1.0f64, 1.0f64);
    let mut rust_more = 0;
    let mut worst_one = 0.0f64;
    let mut spearmans: Vec<f64> = Vec::new();
    let (mut low_theirs, mut low_both, mut low_ours) = (0usize, 0usize, 0usize);
    let (mut err_ours, mut err_theirs): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
    for c in cases {
        let y = nums(&c["y_input"]);
        // the weights check_input() gave (it lowers the spikes)
        let w = nums(&c["w_input"]);
        let ph_z = nums(&c["z"]);
        let env = whittaker_envelope(Some(&t), &y, &w, lambda, 3).unwrap();
        let amp = ph_z.iter().cloned().fold(f64::MIN, f64::max) - ph_z.iter().cloned().fold(f64::MAX, f64::min);
        let rmse = (env.z.iter().zip(&ph_z).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / y.len() as f64).sqrt();
        rmse_rel.push(rmse / amp);
        // both curves against the noise-free series
        let truth = nums(&c["truth"]);
        let err = |z: &[f64]| (z.iter().zip(&truth).map(|(a, b)| (a - b).powi(2)).sum::<f64>() / z.len() as f64).sqrt() / amp;
        err_ours.push(err(&env.z));
        err_theirs.push(err(&ph_z));
        corr_min = corr_min.min(pearson(&env.z, &ph_z));

        // wTSM: only the order, from the same fit and the same weights
        let z1 = nums(&c["z_one"]);
        let one = zeit_core::phenology::smooth::whittaker_weighted(&y, &w, lambda).unwrap();
        let d1 = one.iter().zip(&z1).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        if d1 > worst_one {
            worst_one = d1;
            let i = (0..y.len()).max_by(|&a, &b| (one[a] - z1[a]).abs().total_cmp(&(one[b] - z1[b]).abs())).unwrap();
            println!("worst one-step: i {i} y {} w {} ours {} theirs {}", y[i], w[i], one[i], z1[i]);
        }
        let ours = wtsm(&y, &z1, &w, 2, npt, 0.5);
        let theirs = nums(&c["w_tsm"]);
        let sp = pearson(&ranks(&ours), &ranks(&theirs));
        spearman_min = spearman_min.min(sp);
        spearmans.push(sp);
        // the points both lower (a cloud for either): do they agree?
        for i in 0..y.len() {
            low_theirs += (theirs[i] < 0.99 * w[i]) as usize;
            low_both += (theirs[i] < 0.99 * w[i] && ours[i] < 0.99 * w[i]) as usize;
            low_ours += (ours[i] < 0.99 * w[i]) as usize;
        }

        // seasons: ours on our own envelope curve
        let ours_s = split_growing_seasons(&env.z, &t, &settings, npt, None);
        let (beg, peak, end) = (nums(&c["beg"]), nums(&c["peak"]), nums(&c["end"]));
        total += 1;
        match (ours_s.len() as i64 - peak.len() as i64).abs() {
            0 => same_count += 1,
            1 => off_by_one += 1,
            _ => {}
        }
        if ours_s.len() > peak.len() {
            rust_more += 1;
        }
        for s in &ours_s {
            // the nearest phenofit peak, within 8 observations
            let best = (0..peak.len()).min_by(|&a, &b| {
                (peak[a] - s.peak as f64).abs().total_cmp(&(peak[b] - s.peak as f64).abs())
            });
            let Some(k) = best else { continue };
            if (peak[k] - s.peak as f64).abs() > 8.0 {
                continue;
            }
            peak_n += 1;
            peak_ok += ((peak[k] - s.peak as f64).abs() <= 1.0) as usize;
            // start and end (the ends of the series are not comparable: phenofit
            // extends them)
            edge_n += 1;
            edge_ok += ((beg[k] - s.start as f64).abs() <= 3.0 && (end[k] - s.end as f64).abs() <= 3.0) as usize;
        }
    }
    rmse_rel.sort_by(f64::total_cmp);
    let med = rmse_rel[rmse_rel.len() / 2];
    let p90 = rmse_rel[rmse_rel.len() * 9 / 10];
    let f = |a: usize, b: usize| a as f64 / b.max(1) as f64;
    println!("series {total}");
    println!("curve: rmse/amplitude median {med:.4}, p90 {p90:.4}, min correlation {corr_min:.4}");
    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len() as f64;
    println!(
        "error against the truth (rmse / amplitude, mean of the series): Zeit {:.4}, phenofit {:.4}",
        mean(&err_ours),
        mean(&err_theirs)
    );
    println!("one-step Whittaker, largest difference to phenofit's first pass: {worst_one:.2e}");
    spearmans.sort_by(f64::total_cmp);
    let sp_med = spearmans[spearmans.len() / 2];
    println!("wTSM: Spearman min {spearman_min:.3}, median {sp_med:.3}");
    println!(
        "wTSM: points lowered by phenofit {low_theirs}, by Zeit {low_ours}, by both {low_both} ({:.1}% of phenofit's)",
        100.0 * f(low_both, low_theirs)
    );
    println!(
        "seasons: same count {:.1}%, off by one {:.1}%, Zeit finds more in {rust_more}",
        100.0 * f(same_count, total),
        100.0 * f(off_by_one, total)
    );
    println!(
        "peaks within 1 obs: {:.1}% of {peak_n}; start and end within 3 obs: {:.1}% of {edge_n}",
        100.0 * f(peak_ok, peak_n),
        100.0 * f(edge_ok, edge_n)
    );

    assert!(total >= 150);
    // The curves differ (the weight rules are not the same: phenofit lowers a few
    // points to 0.5, Zeit every point below the curve in proportion), but Zeit's
    // is no farther from the truth than phenofit's and they agree on the clouds.
    assert!(p90 <= 0.2, "curve rmse p90 {p90}");
    assert!(corr_min >= 0.8, "curve correlation {corr_min}");
    assert!(mean(&err_ours) <= 1.1 * mean(&err_theirs), "error against the truth");
    assert!(f(low_both, low_theirs) >= 0.9, "points lowered by both");
    assert!(f(same_count + off_by_one, total) >= 0.9, "season count agreement");
    assert!(f(same_count, total) >= 0.9, "season count equal");
    assert!(f(peak_ok, peak_n) >= 0.9, "peak positions");
    assert!(f(edge_ok, edge_n) >= 0.8, "start and end positions");
}
