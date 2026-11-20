//! Which fine images pair with which coarse images, which dates are output, and which pairs predict each of them.
//!
//! Days are days since 1970-01-01 (UTC dates). A pair is a fine date matched to a coarse image; its date for the
//! temporal distances is the fine date.

/// The two methods differ in how many pairs a date uses and in what happens at the ends of the series.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Starfm,
    Estarfm,
}

/// How a fine date finds its coarse image.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairMatch {
    /// Composite windows when the coarse series has them, else the nearest date within the tolerance.
    Auto,
    /// The composite whose `[start, end]` contains the fine date.
    Window,
    /// The nearest coarse date within the tolerance.
    Nearest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PairsMode {
    /// The nearest pair before and the nearest after (STARFM).
    Bracket,
    /// The nearest pair (STARFM).
    Nearest,
}

#[derive(Clone, Debug)]
pub struct DateRule {
    /// Output dates (each snapped to a coarse date), or all coarse dates of `[start, end]`.
    pub dates: Option<Vec<i64>>,
    pub start: Option<i64>,
    pub end: Option<i64>,
    pub include_fine_dates: bool,
    pub keep_observed: bool,
    /// Fine dates never used as pairs.
    pub exclude: Vec<i64>,
    pub pair_match: PairMatch,
    /// Days; `None` = half the median coarse interval, rounded down.
    pub tolerance: Option<i64>,
    pub min_valid: f64,
    pub max_gap: Option<i64>,
    pub pairs: PairsMode,
}

impl Default for DateRule {
    fn default() -> Self {
        DateRule {
            dates: None,
            start: None,
            end: None,
            include_fine_dates: true,
            keep_observed: true,
            exclude: Vec::new(),
            pair_match: PairMatch::Auto,
            tolerance: None,
            min_valid: 0.3,
            max_gap: None,
            pairs: PairsMode::Bracket,
        }
    }
}

/// One output date.
#[derive(Clone, Debug, PartialEq)]
pub struct Target {
    /// The output date.
    pub day: i64,
    /// The coarse image of the date.
    pub coarse: usize,
    /// `(fine index, coarse index)` of the pairs used, in date order.
    pub pairs: Vec<(usize, usize)>,
    /// A fine date whose valid pixels are written as observed.
    pub observed: Option<usize>,
    pub extrapolated: bool,
}

#[derive(Clone, Debug, Default)]
pub struct Plan {
    pub targets: Vec<Target>,
    /// Dates asked for that could not be output, with the reason.
    pub skipped: Vec<(i64, String)>,
    /// Fine dates that are not pairs, with the reason.
    pub unpaired: Vec<(usize, String)>,
}

impl Plan {
    /// The fine indices the plan reads, sorted.
    pub fn fine_used(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .targets
            .iter()
            .flat_map(|t| t.pairs.iter().map(|p| p.0).chain(t.observed))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
    /// The coarse indices the plan reads, sorted.
    pub fn coarse_used(&self) -> Vec<usize> {
        let mut v: Vec<usize> = self
            .targets
            .iter()
            .flat_map(|t| t.pairs.iter().map(|p| p.1).chain(std::iter::once(t.coarse)))
            .collect();
        v.sort_unstable();
        v.dedup();
        v
    }
}

fn median_interval(days: &[i64]) -> f64 {
    let mut d: Vec<i64> = days.windows(2).map(|w| w[1] - w[0]).collect();
    if d.is_empty() {
        return 1.0;
    }
    d.sort_unstable();
    let n = d.len();
    if n % 2 == 1 {
        d[n / 2] as f64
    } else {
        (d[n / 2 - 1] + d[n / 2]) as f64 / 2.0
    }
}

/// The nearest coarse date to `day` within `tol` days (ties: earlier).
fn nearest(coarse_days: &[i64], day: i64, tol: i64) -> Option<usize> {
    let mut best: Option<(i64, usize)> = None;
    for (j, &g) in coarse_days.iter().enumerate() {
        let d = (g - day).abs();
        if d <= tol && best.is_none_or(|(bd, bj)| d < bd || (d == bd && g < coarse_days[bj])) {
            best = Some((d, j));
        }
    }
    best.map(|b| b.1)
}

/// Pairs and output dates (spec section 6). `fine_valid` is the valid fraction of each fine image over the
/// processing extent; `coarse_windows` the `[start, end]` of each coarse composite.
pub fn plan(
    fine_days: &[i64],
    fine_valid: &[f64],
    coarse_days: &[i64],
    coarse_windows: Option<&[(i64, i64)]>,
    rule: &DateRule,
    method: Method,
) -> Result<Plan, String> {
    if fine_days.len() != fine_valid.len() {
        return Err("fine_days and fine_valid differ in length".into());
    }
    if coarse_days.is_empty() || fine_days.is_empty() {
        return Err("fusion needs at least one fine and one coarse date".into());
    }
    if coarse_windows.is_some_and(|w| w.len() != coarse_days.len()) {
        return Err("coarse_windows and coarse_days differ in length".into());
    }
    let tol = rule
        .tolerance
        .unwrap_or_else(|| (median_interval(coarse_days) / 2.0).floor() as i64);
    let use_window = match rule.pair_match {
        PairMatch::Window => {
            if coarse_windows.is_none() {
                return Err("pair_match = window needs the composite windows of the coarse series".into());
            }
            true
        }
        PairMatch::Nearest => false,
        PairMatch::Auto => coarse_windows.is_some(),
    };
    let match_fine = |day: i64| -> Option<usize> {
        if use_window {
            let w = coarse_windows.unwrap();
            let mut best: Option<(i64, usize)> = None;
            for j in 0..coarse_days.len() {
                if w[j].0 <= day && day <= w[j].1 {
                    let d = (coarse_days[j] - day).abs();
                    if best.is_none_or(|(bd, bj)| d < bd || (d == bd && coarse_days[j] < coarse_days[bj])) {
                        best = Some((d, j));
                    }
                }
            }
            best.map(|b| b.1)
        } else {
            nearest(coarse_days, day, tol)
        }
    };

    let mut out = Plan::default();
    // The fine dates that can be pairs: (fine, coarse).
    let mut matched: Vec<Option<usize>> = Vec::with_capacity(fine_days.len());
    let mut cand: Vec<(usize, usize)> = Vec::new();
    for (i, &d) in fine_days.iter().enumerate() {
        let m = match_fine(d);
        matched.push(m);
        match m {
            None => out.unpaired.push((i, "no coarse image of that date".into())),
            Some(j) => {
                if rule.exclude.contains(&d) {
                    out.unpaired.push((i, "excluded".into()));
                } else if fine_valid[i] < rule.min_valid {
                    out.unpaired
                        .push((i, format!("only {:.0}% valid", fine_valid[i] * 100.0)));
                } else {
                    cand.push((i, j));
                }
            }
        }
    }
    cand.sort_by_key(|&(i, _)| (fine_days[i], i));

    let first = rule.start.or_else(|| fine_days.iter().copied().min());
    let last = rule.end.or_else(|| fine_days.iter().copied().max());

    // The output dates: (day, coarse, observed).
    let mut outs: Vec<(i64, usize, Option<usize>)> = Vec::new();
    let (lo, hi);
    if let Some(list) = &rule.dates {
        for &d in list {
            match nearest(coarse_days, d, tol) {
                Some(j) => outs.push((coarse_days[j], j, None)),
                None => return Err(format!("date {d} has no coarse image within {tol} days")),
            }
        }
        lo = list.iter().copied().min().unwrap_or(0);
        hi = list.iter().copied().max().unwrap_or(0);
    } else {
        lo = first.unwrap();
        hi = last.unwrap();
        for (j, &g) in coarse_days.iter().enumerate() {
            if g >= lo && g <= hi {
                outs.push((g, j, None));
            }
        }
    }
    if rule.include_fine_dates {
        for (i, &d) in fine_days.iter().enumerate() {
            if d < lo || d > hi {
                continue;
            }
            match matched[i] {
                Some(j) => {
                    // The fine date replaces the coarse output of its own day.
                    outs.retain(|o| !(o.0 == d && o.2.is_none()));
                    outs.push((d, j, if rule.keep_observed { Some(i) } else { None }));
                }
                None => out.skipped.push((d, "fine date without a coarse image".into())),
            }
        }
    }
    outs.sort_by_key(|o| (o.0, o.2.is_some()));
    outs.dedup_by_key(|o| (o.0, o.2));

    for (day, coarse, observed) in outs {
        let usable: Vec<(usize, usize)> = cand
            .iter()
            .copied()
            .filter(|&(i, _)| Some(i) != observed)
            .filter(|&(i, _)| rule.max_gap.is_none_or(|g| (fine_days[i] - day).abs() <= g))
            .collect();
        let (pairs, extrapolated) = match method {
            Method::Starfm => (starfm_pairs(&usable, fine_days, day, rule.pairs), false),
            Method::Estarfm => estarfm_pairs(&usable, fine_days, day),
        };
        let need = if method == Method::Estarfm { 2 } else { 1 };
        if pairs.len() < need {
            out.skipped
                .push((day, format!("{} pair(s) available, {need} needed", pairs.len())));
            continue;
        }
        out.targets.push(Target {
            day,
            coarse,
            pairs,
            observed,
            extrapolated,
        });
    }
    if out.targets.is_empty() {
        let why: Vec<String> = out.skipped.iter().map(|s| format!("day {}: {}", s.0, s.1)).collect();
        return Err(format!("no date can be predicted ({})", why.join("; ")));
    }
    Ok(out)
}

fn starfm_pairs(usable: &[(usize, usize)], fine_days: &[i64], day: i64, mode: PairsMode) -> Vec<(usize, usize)> {
    match mode {
        PairsMode::Nearest => {
            let mut best: Option<(i64, i64, (usize, usize))> = None;
            for &p in usable {
                let key = ((fine_days[p.0] - day).abs(), fine_days[p.0]);
                if best.is_none_or(|(d, f, _)| key < (d, f)) {
                    best = Some((key.0, key.1, p));
                }
            }
            best.map(|b| vec![b.2]).unwrap_or_default()
        }
        PairsMode::Bracket => {
            let before = usable
                .iter()
                .copied()
                .filter(|p| fine_days[p.0] <= day)
                .max_by_key(|p| (fine_days[p.0], p.0));
            let after = usable
                .iter()
                .copied()
                .filter(|p| fine_days[p.0] >= day)
                .min_by_key(|p| (fine_days[p.0], p.0));
            let mut v: Vec<(usize, usize)> = Vec::new();
            for p in [before, after].into_iter().flatten() {
                if !v.contains(&p) {
                    v.push(p);
                }
            }
            v.sort_by_key(|p| (fine_days[p.0], p.0));
            v
        }
    }
}

/// The two bracketing pairs, or the two nearest on one side (`extrapolated`).
fn estarfm_pairs(usable: &[(usize, usize)], fine_days: &[i64], day: i64) -> (Vec<(usize, usize)>, bool) {
    let mut before: Vec<(usize, usize)> = usable.iter().copied().filter(|p| fine_days[p.0] <= day).collect();
    let mut after: Vec<(usize, usize)> = usable.iter().copied().filter(|p| fine_days[p.0] > day).collect();
    before.sort_by_key(|p| std::cmp::Reverse((fine_days[p.0], p.0)));
    after.sort_by_key(|p| (fine_days[p.0], p.0));
    let mut v: Vec<(usize, usize)>;
    let mut extra = false;
    if !before.is_empty() && !after.is_empty() {
        v = vec![before[0], after[0]];
    } else if before.is_empty() {
        v = after.iter().take(2).copied().collect();
        extra = true;
    } else {
        v = before.iter().take(2).copied().collect();
        extra = true;
    }
    v.sort_by_key(|p| (fine_days[p.0], p.0));
    (v, extra)
}
