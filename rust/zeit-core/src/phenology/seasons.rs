//! The growing seasons of a smoothed series (phenofit's findpeaks,
//! removeClosedExtreme, check_GS_HeadTail and season filter): a port of
//! split_growing_seasons of src/phenology_math.cpp.

use crate::eigen::{std_max, std_min};

/// A season: indices of its start, peak and end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Season {
    pub start: usize,
    pub peak: usize,
    pub end: usize,
}

#[derive(Clone, Copy)]
struct Extreme {
    pos: usize,
    /// 1 a maximum, -1 a minimum
    kind: i32,
    val: f64,
}

/// The maxima (`kind` 1) or minima (-1): the middle of each plateau after a
/// rise and before a fall (of more than 1e-9).
fn find_extremes(y: &[f64], kind: i32) -> Vec<Extreme> {
    let mut extr = vec![];
    let (mut dir, mut last_change) = (0, 0usize);
    for i in 1..y.len() {
        let diff = if kind == 1 { y[i] - y[i - 1] } else { y[i - 1] - y[i] };
        let cur = if diff > 1e-9 {
            1
        } else if diff < -1e-9 {
            -1
        } else {
            0
        };
        if cur == 1 {
            dir = 1;
            last_change = i - 1;
        } else if cur == -1 {
            if dir == 1 {
                let pos = (last_change + 1 + i - 1) / 2;
                extr.push(Extreme { pos, kind, val: y[pos] });
            }
            dir = -1;
        }
    }
    extr
}

fn split_once(
    y: &[f64],
    dates: &[f64],
    min_season_length: i32,
    min_amplitude: f64,
    rtrough_max: f64,
    r_min_filter: f64,
) -> Vec<Season> {
    let n = y.len();
    if n < 3 {
        return vec![];
    }
    let (mut min_y, mut max_y) = (y[0], y[0]);
    for &v in y {
        if v < min_y {
            min_y = v;
        }
        if v > max_y {
            max_y = v;
        }
    }
    let a = max_y - min_y;
    if a == 0.0 {
        return vec![];
    }

    let troughs = find_extremes(y, -1);
    let peaks = find_extremes(y, 1);
    let mut pos: Vec<Extreme> = troughs
        .into_iter()
        .filter(|t| t.val - min_y <= rtrough_max * a)
        .collect();
    pos.extend(peaks);
    // std::sort by position: no two extremes share one
    pos.sort_by_key(|e| e.pos);
    if pos.is_empty() {
        return vec![];
    }

    // removeClosedExtreme
    let y_min = 0.05 * a;
    let mut kept = Vec::with_capacity(pos.len());
    let mut i = 0;
    while i < pos.len() {
        if i < pos.len() - 1 && (pos[i + 1].val - pos[i].val).abs() < y_min {
            kept.push(if pos[i].kind == 1 {
                if pos[i].val > pos[i + 1].val {
                    pos[i]
                } else {
                    pos[i + 1]
                }
            } else if pos[i].val < pos[i + 1].val {
                pos[i]
            } else {
                pos[i + 1]
            });
            i += 1;
        } else {
            kept.push(pos[i]);
        }
        i += 1;
    }
    let mut pos = kept;

    // check_GS_HeadTail. With a single extreme, a maximum, the C++ reads
    // outside its vector (pos[-1], pos[1]): no trough is added here.
    let minlen = (23 / 3) as i64;
    if pos.len() >= 2 && pos[pos.len() - 1].kind == 1 {
        let prev = pos[pos.len() - 2];
        if (n as i64 - prev.pos as i64) > minlen && (y[n - 1] - prev.val).abs() < 0.15 * a {
            pos.push(Extreme {
                pos: n - 1,
                kind: -1,
                val: y[n - 1],
            });
        }
    }
    if pos.len() >= 2 && pos[0].kind == 1 && pos[1].pos as i64 > minlen && (y[0] - pos[1].val).abs() < 0.15 * a {
        pos.insert(
            0,
            Extreme {
                pos: 0,
                kind: -1,
                val: y[0],
            },
        );
    }

    // from the first trough to the last
    let first = pos.iter().position(|e| e.kind == -1);
    let last = pos.iter().rposition(|e| e.kind == -1);
    let (first, last) = match (first, last) {
        (Some(f), Some(l)) if f != l => (f, l),
        _ => return vec![],
    };
    let pos = &pos[first..=last];

    let mut seasons = vec![];
    for i in 0..pos.len().saturating_sub(2) {
        if pos[i].kind == -1 && pos[i + 1].kind == 1 && pos[i + 2].kind == -1 {
            seasons.push(Season {
                start: pos[i].pos,
                peak: pos[i + 1].pos,
                end: pos[i + 2].pos,
            });
        }
    }

    // rcpp_season_filter: merge close seasons
    let (days_max_diff, days_max_2gs) = (10i64, 40i64);
    let mut i = 0;
    while i + 1 < seasons.len() {
        let t_diff = seasons[i + 1].start as i64 - seasons[i].end as i64;
        let t2s = seasons[i + 1].end as i64 - seasons[i].start as i64;
        if t_diff < 0 {
            seasons[i].end = seasons[i + 1].start;
        }
        let is_closed = (0..=days_max_diff).contains(&t_diff);
        if is_closed && t2s <= days_max_2gs {
            let (s1, s2) = (seasons[i], seasons[i + 1]);
            let t1_min = std_min(y[s1.start], y[s1.end]);
            let t2_min = std_min(y[s2.start], y[s2.end]);
            let max_yy = std_max(y[s1.peak], y[s2.peak]);
            let min_yy = std_min(t1_min, t2_min);
            let local_a = max_yy - min_yy;
            let t2_h_left = y[s2.peak] - y[s2.start];
            let t1_h_right = y[s1.peak] - y[s1.end];
            let trs = local_a * r_min_filter;
            let trs2 = local_a * (r_min_filter + 0.1);
            let con_left = t1_h_right <= trs && y[s1.start] < y[s2.start] && (y[s1.peak] > trs2 + t1_min);
            let con_right = t2_h_left <= trs && y[s1.end] > y[s2.end] && (y[s2.peak] > trs2 + t2_min);
            if (y[s1.end] >= local_a * 0.7 + t1_min) || con_right || con_left {
                seasons[i].end = s2.end;
                if y[seasons[i].peak] < y[s2.peak] {
                    seasons[i].peak = s2.peak;
                }
                seasons.remove(i + 1);
                continue; // recheck
            }
        }
        i += 1;
    }

    seasons
        .into_iter()
        .filter(|s| {
            dates[s.end] - dates[s.start] >= min_season_length as f64
                && y[s.peak] - std_max(y[s.start], y[s.end]) >= min_amplitude
        })
        .collect()
}

/// split_growing_seasons: with `retry_on_empty`, a second pass with a looser
/// trough threshold when the first finds no season (phenofit's season_mov).
pub fn split_growing_seasons(
    y: &[f64],
    dates: &[f64],
    min_season_length: i32,
    min_amplitude: f64,
    rtrough_max: f64,
    r_min_filter: f64,
    retry_on_empty: bool,
) -> Vec<Season> {
    let mut seasons = split_once(y, dates, min_season_length, min_amplitude, rtrough_max, r_min_filter);
    if seasons.is_empty() && retry_on_empty {
        let relaxed = std_min(rtrough_max + 0.2, 0.95);
        if relaxed > rtrough_max {
            seasons = split_once(y, dates, min_season_length, min_amplitude, relaxed, r_min_filter);
        }
    }
    seasons
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_seasons_with_their_head_and_tail() {
        let y: Vec<f64> = (0..69)
            .map(|i| 0.2 + 0.6 * (i as f64 * std::f64::consts::PI / 23.0).sin().powi(2))
            .collect();
        let dates: Vec<f64> = (0..69).map(|i| i as f64 * 16.0).collect();
        let s = split_growing_seasons(&y, &dates, 0, 0.0, 0.6, 0.02, true);
        let bounds: Vec<(usize, usize)> = s.iter().map(|s| (s.start, s.end)).collect();
        assert_eq!(bounds, [(0, 23), (23, 46), (46, 68)]);
    }
}
