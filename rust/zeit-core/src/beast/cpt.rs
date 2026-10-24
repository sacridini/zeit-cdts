//! The changepoints read off a changepoint-occurrence probability curve:
//! `FindChangepoint_LeftRightMargins` of Rbeast's `abc_common.c`, with the
//! introsort of `abc_sort_template.h` that orders them (its order of ties is
//! part of the result).

use super::kern::{sum, sumfilter};

/// The introsort of `abc_sort_template.h` on `arr` with `index` carried along:
/// ascending (`f32a_introSort_index`) or descending (`f32d_introSort_index`).
pub fn introsort_index(arr: &mut [f32], index: &mut [i32], descending: bool) {
    let n = arr.len();
    if n == 0 {
        return;
    }
    let mut depth = 0;
    let mut m = n;
    loop {
        m >>= 1;
        if m == 0 {
            break;
        }
        depth += 1;
    }
    let mut s = Sorter { arr, index, descending };
    s.introsort(0, n as i64 - 1, depth);
}

struct Sorter<'a> {
    arr: &'a mut [f32],
    index: &'a mut [i32],
    descending: bool,
}

impl Sorter<'_> {
    #[inline]
    fn less(&self, a: f32, b: f32) -> bool {
        if self.descending {
            a > b
        } else {
            a < b
        }
    }

    #[inline]
    fn swap(&mut self, i: i64, j: i64) {
        self.arr.swap(i as usize, j as usize);
        self.index.swap(i as usize, j as usize);
    }

    #[inline]
    fn a(&self, i: i64) -> f32 {
        self.arr[i as usize]
    }

    fn sort2(&mut self, a: i64, b: i64) {
        if self.less(self.a(b), self.a(a)) {
            self.swap(a, b);
        }
    }

    fn sort3(&mut self, a: i64, b: i64, c: i64) {
        self.sort2(a, b);
        if self.less(self.a(c), self.a(b)) {
            self.swap(b, c);
            self.sort2(a, b);
        }
    }

    fn binary_search(&self, key: f32, mut low: i64, mut high: i64) -> i64 {
        while high > low {
            let mid = (low + high) / 2;
            if self.less(key, self.a(mid)) {
                high = mid - 1;
            } else {
                low = mid + 1;
            }
        }
        if high < low {
            low
        } else if self.less(key, self.a(low)) {
            low
        } else {
            low + 1
        }
    }

    fn insertion_by_three(&mut self, low: i64, high: i64) {
        let n = high - low + 1;
        if n == 1 {
            return;
        }
        self.sort2(low, low + 1);
        let mut skip = n % 3;
        if skip == 0 {
            if self.less(self.a(low + 2), self.a(low + 1)) {
                self.swap(low + 1, low + 2);
                self.sort2(low, low + 1);
            }
            skip = 3;
        }
        let mut i = low + skip;
        while i <= high - 2 {
            self.sort3(i, i + 1, i + 2);
            let (k1, k2, k3) = (self.a(i), self.a(i + 1), self.a(i + 2));
            let loc1 = self.binary_search(k1, low, i - 1);
            if loc1 == i {
                i += 3;
                continue;
            }
            let loc2 = self.binary_search(k2, loc1, i - 1);
            let loc3 = if loc2 == i {
                i
            } else {
                self.binary_search(k3, loc2, i - 1)
            };
            let (i1, i2, i3) = (
                self.index[i as usize],
                self.index[i as usize + 1],
                self.index[i as usize + 2],
            );
            let mut j = i - 1;
            while j >= loc3 {
                self.arr[(j + 3) as usize] = self.arr[j as usize];
                self.index[(j + 3) as usize] = self.index[j as usize];
                j -= 1;
            }
            self.arr[(loc3 + 2) as usize] = k3;
            self.index[(loc3 + 2) as usize] = i3;
            let mut j = loc3 - 1;
            while j >= loc2 {
                self.arr[(j + 2) as usize] = self.arr[j as usize];
                self.index[(j + 2) as usize] = self.index[j as usize];
                j -= 1;
            }
            self.arr[(loc2 + 1) as usize] = k2;
            self.index[(loc2 + 1) as usize] = i2;
            let mut j = loc2 - 1;
            while j >= loc1 {
                self.arr[(j + 1) as usize] = self.arr[j as usize];
                self.index[(j + 1) as usize] = self.index[j as usize];
                j -= 1;
            }
            self.arr[loc1 as usize] = k1;
            self.index[loc1 as usize] = i1;
            i += 3;
        }
    }

    fn partition_right(&mut self, low: i64, high: i64) -> (i64, bool) {
        let n = high - low + 1;
        if n == 2 {
            self.sort2(low, high);
            return (low, false);
        }
        let mid = (low + high) / 2;
        let mut first = low;
        let mut last;
        if n > 512 {
            self.sort3(low, mid - 1, high - 2);
            self.sort3(low + 1, mid, high - 1);
            self.sort3(low + 2, mid + 1, high);
            self.sort3(mid - 1, mid, mid + 1);
            self.swap(low, mid);
            last = high + 1;
        } else {
            self.sort3(mid, low, high);
            last = high;
        }
        let pivot = self.a(low);
        loop {
            first += 1;
            if !self.less(self.a(first), pivot) {
                break;
            }
        }
        if first == low + 1 && !self.less(self.a(mid), pivot) && !self.less(self.a(low + 2), pivot) {
            while last > first {
                last -= 1;
                if self.less(self.a(last), pivot) {
                    break;
                }
            }
        } else {
            loop {
                last -= 1;
                if self.less(self.a(last), pivot) {
                    break;
                }
            }
        }
        let no_swap = first >= last;
        while first < last {
            self.swap(first, last);
            loop {
                first += 1;
                if !self.less(self.a(first), pivot) {
                    break;
                }
            }
            loop {
                last -= 1;
                if self.less(self.a(last), pivot) {
                    break;
                }
            }
        }
        let p = first - 1;
        self.swap(low, p);
        (p, no_swap)
    }

    fn heapify(&mut self, base: i64, n: i64, i: i64) {
        let mut i = i;
        loop {
            let mut largest = i;
            let left = 2 * i + 1;
            let right = 2 * i + 2;
            if left < n && self.less(self.a(base + largest), self.a(base + left)) {
                largest = left;
            }
            if right < n && self.less(self.a(base + largest), self.a(base + right)) {
                largest = right;
            }
            if largest == i {
                break;
            }
            self.swap(base + i, base + largest);
            i = largest;
        }
    }

    fn heapsort(&mut self, low: i64, high: i64) {
        let n = high - low + 1;
        let mut i = n / 2 - 1;
        while i >= 0 {
            self.heapify(low, n, i);
            i -= 1;
        }
        let mut i = n - 1;
        while i >= 0 {
            self.swap(low, low + i);
            self.heapify(low, i, 0);
            i -= 1;
        }
    }

    fn is_ordered(&self, low: i64, high: i64) -> bool {
        let mut i = low + 1;
        while i <= high {
            if self.less(self.a(i), self.a(i - 1)) {
                return false;
            }
            i += 1;
        }
        true
    }

    fn introsort(&mut self, low: i64, high: i64, depth: i64) {
        let size = high - low + 1;
        if size <= 1 {
            return;
        }
        if size <= 32 {
            self.insertion_by_three(low, high);
            return;
        }
        if depth == 0 {
            self.heapsort(low, high);
            return;
        }
        let (p, no_swap) = self.partition_right(low, high);
        let (mut left_ordered, mut right_ordered) = (false, false);
        if no_swap {
            left_ordered = self.is_ordered(low, p - 1);
            right_ordered = self.is_ordered(p + 1, high);
        }
        let l_size = p - low;
        let r_size = high - (p + 1);
        let unbalanced = (l_size < size / 8 || r_size < size / 8) as i64;
        if !left_ordered {
            self.introsort(low, p - 1, depth - unbalanced);
        }
        if !right_ordered {
            self.introsort(p + 1, high, depth - unbalanced);
        }
    }
}

/// `confidenceInterval` of `abc_common.c`: how far into a half segment of
/// probabilities, sorted, 95% of their sum lies.
fn confidence_interval(half: &[f32], right: bool) -> f32 {
    let n = half.len();
    let total = sum(half);
    let mut cum = 0f32;
    if right {
        let mut j = 0;
        while j < n {
            cum += half[j];
            if cum / total >= 0.95 {
                break;
            }
            j += 1;
        }
        let j = j.min(n - 1);
        let jj = j as f32 + 1.0;
        jj - (cum - 0.95f32 * total) / half[j]
    } else {
        let mut j = n as i64 - 1;
        while j >= 0 {
            cum += half[j as usize];
            if cum / total >= 0.95 {
                break;
            }
            j -= 1;
        }
        let j = j.max(0);
        let jj = (n as i64 - j) as f32;
        jj - (cum - 0.95f32 * total) / half[j as usize]
    }
}

/// The changepoints found in `prob` (the occurrence probability of each
/// date): their positions (0-based) and probabilities, in the order of their
/// probabilities, and their 95% intervals (`ci[i]` the lower end and
/// `ci[n_cp + i]` the upper of the i-th).
pub struct Changepoints {
    pub pos: Vec<i32>,
    pub prob: Vec<f32>,
    pub ci: Vec<f32>,
}

/// `FindChangepoint_LeftRightMargins(prob, mem, 0.01, ...)`.
pub fn find(
    prob: &[f32],
    threshold: f32,
    min_sep: i32,
    max_cpt: i32,
    left_margin: i32,
    right_margin: i32,
) -> Changepoints {
    let n = prob.len() as i32;
    let empty = Changepoints {
        pos: vec![],
        prob: vec![],
        ci: vec![],
    };
    if max_cpt == 0 {
        return empty;
    }
    if min_sep == 0 {
        return find_minseg_one(prob, threshold, max_cpt);
    }
    let min_seg = min_sep + 1;
    let mut sump = vec![0f32; n as usize];
    sumfilter(prob, &mut sump, min_seg as usize);
    sump[0] = 0.0;
    let w0 = min_seg / 2;
    let w1 = min_seg - w0 - 1;
    let local_peak = |j: i32, p: &[f32]| {
        let j = j as usize;
        (p[j] > p[j - 1] && p[j] >= p[j + 1]) || (p[j] >= p[j - 1] && p[j] > p[j + 1])
    };
    let peak4 = |j: i32, p: &[f32]| {
        let j = j as usize;
        p[j] >= p[j - 1] && p[j] >= p[j + 1] && p[j] >= p[j - 2] && p[j] >= p[j + 2]
    };
    let mut sum_pos: Vec<i32> = Vec::new();
    let mut sum_val: Vec<f32> = Vec::new();
    let lower = left_margin + 1;
    let upper = (n - 1) - right_margin;
    let mut i = lower.max(1);
    while i <= upper {
        let iu = i as usize;
        if !(sump[iu] < threshold) {
            let has_right = i < n - 1;
            let has_two = i > 1 && i < n - 2;
            let skip = (has_right && !local_peak(i, &sump))
                || (min_seg > 4 && has_two && !peak4(i, &sump))
                || (!has_right && !(sump[iu] > sump[iu - 1]));
            if !skip {
                let k = sum_pos.len();
                let dist = if k > 0 { i - sum_pos[k - 1] } else { 0 };
                if k == 0 || dist > min_sep || dist < -min_sep {
                    sum_pos.push(i);
                    sum_val.push(sump[iu]);
                } else if sump[iu] > sum_val[k - 1] {
                    sum_pos[k - 1] = i;
                    sum_val[k - 1] = sump[iu];
                }
            }
        }
        i += 1;
    }
    let num = sum_pos.len();
    let mut prob_pos = vec![0i32; num];
    if min_sep <= 4 {
        for (k, &cpt) in sum_pos.iter().enumerate() {
            let lo = (cpt - w0).max(1);
            let hi = (cpt + w1).min(n - 1);
            let mut max_idx = cpt;
            let mut max_val = prob[cpt as usize];
            for j in lo..=hi {
                if prob[j as usize] > max_val {
                    max_idx = j;
                    max_val = prob[j as usize];
                }
            }
            prob_pos[k] = max_idx;
        }
    } else {
        for (k, &cpt) in sum_pos.iter().enumerate() {
            let lo = (cpt - w0).max(1);
            let hi = (cpt + w1).min(n - 1);
            let mut loc: Vec<i32> = Vec::with_capacity(15);
            let mut prb: Vec<f32> = Vec::with_capacity(15);
            for j in lo..=hi {
                let ju = j as usize;
                let cur = prob[ju];
                let has_right = j < n - 1;
                if (has_right && local_peak(j, prob)) || (!has_right && cur > prob[ju - 1]) {
                    if loc.len() < 15 {
                        loc.push(j);
                        prb.push(cur);
                    } else {
                        let mut min_p = 100f32;
                        let mut min_i = 0;
                        for (q, &p) in prb.iter().enumerate() {
                            if min_p > p {
                                min_i = q;
                                min_p = p;
                            }
                        }
                        if cur > min_p {
                            loc[min_i] = j;
                            prb[min_i] = cur;
                        }
                    }
                }
            }
            prob_pos[k] = match loc.len() {
                0 => cpt,
                1 => loc[0],
                _ => {
                    introsort_index(&mut prb, &mut loc, true);
                    if prb[0] as f64 > 1.5 * prb[1] as f64 {
                        loc[0]
                    } else {
                        let mut best_k = 0;
                        let mut best = -9999f32;
                        for (q, &c) in loc.iter().enumerate() {
                            let c = c as usize;
                            let p3 = if c as i32 == n - 1 {
                                prob[c] + prob[c - 1]
                            } else {
                                prob[c] + prob[c + 1] + prob[c - 1]
                            };
                            if p3 > best {
                                best = p3;
                                best_k = q;
                            }
                        }
                        loc[best_k]
                    }
                }
            };
        }
    }
    if num == 0 {
        return empty;
    }
    finish(prob, sum_val, prob_pos, max_cpt)
}

/// The end of the changepoint search: the changepoints sorted by summed
/// probability, at most `max_cpt` of them, and their intervals.
fn finish(prob: &[f32], mut sum_val: Vec<f32>, mut prob_pos: Vec<i32>, max_cpt: i32) -> Changepoints {
    let n = prob.len();
    introsort_index(&mut sum_val, &mut prob_pos, true);
    let num = sum_val.len().min(max_cpt.max(0) as usize);
    let cpt: Vec<i32> = prob_pos[..num].to_vec();
    let mut order: Vec<i32> = (0..num as i32).collect();
    let mut cpt_f: Vec<f32> = cpt.iter().map(|&c| c as f32).collect();
    introsort_index(&mut cpt_f, &mut order, false);
    let mut ci = vec![-9999f32; 2 * num];
    let mut seg: Vec<f32> = Vec::with_capacity(n);
    for i in 0..num {
        let end = cpt_f[i] as i64;
        let start = if i == 0 { 0 } else { cpt_f[i - 1] as i64 };
        let start = (start + end) / 2;
        seg.clear();
        seg.extend_from_slice(&prob[start as usize..=end as usize]);
        seg.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        ci[i] = confidence_interval(&seg, false);
        let start = cpt_f[i] as i64;
        let end = if i == num - 1 {
            n as i64 - 1
        } else {
            cpt_f[i + 1] as i64
        };
        let end = (start + end) / 2;
        seg.clear();
        seg.extend_from_slice(&prob[start as usize..=end as usize]);
        seg.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        ci[num + i] = confidence_interval(&seg, true);
    }
    let backup = ci.clone();
    let mut out_ci = ci;
    let mut out_prob = vec![0f32; num];
    for i in 0..num {
        let idx = order[i] as usize;
        out_ci[idx] = cpt_f[i] - backup[i];
        out_ci[num + idx] = cpt_f[i] + backup[num + i];
        out_prob[i] = if 1.0 < sum_val[i] as f64 { 1.0 } else { sum_val[i] };
    }
    Changepoints {
        pos: cpt,
        prob: out_prob,
        ci: out_ci,
    }
}

/// `FindChangepoint_minseg_is_1`: with no minimum separation, every date but
/// the last at or above the threshold, its interval the date itself.
fn find_minseg_one(prob: &[f32], threshold: f32, max_cpt: i32) -> Changepoints {
    let n = prob.len();
    let mut pos: Vec<i32> = Vec::new();
    let mut val: Vec<f32> = Vec::new();
    for (i, &p) in prob.iter().enumerate().take(n.saturating_sub(1)) {
        if p < threshold {
            continue;
        }
        pos.push(i as i32);
        val.push(p);
    }
    if pos.is_empty() {
        return Changepoints {
            pos: vec![],
            prob: vec![],
            ci: vec![],
        };
    }
    introsort_index(&mut val, &mut pos, true);
    let num = val.len().min(max_cpt.max(0) as usize);
    let cpt: Vec<i32> = pos[..num].to_vec();
    let mut order: Vec<i32> = (0..num as i32).collect();
    let mut cpt_f: Vec<f32> = cpt.iter().map(|&c| c as f32).collect();
    introsort_index(&mut cpt_f, &mut order, false);
    let mut ci = vec![0f32; 2 * num];
    let mut out_prob = vec![0f32; num];
    for i in 0..num {
        let idx = order[i] as usize;
        ci[idx] = cpt_f[i];
        ci[num + idx] = cpt_f[i];
        out_prob[i] = if val[i] > 1.0 { 1.0 } else { val[i] };
    }
    Changepoints {
        pos: cpt,
        prob: out_prob,
        ci,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn introsort_sorts_with_index() {
        for &n in &[1usize, 2, 3, 7, 31, 32, 33, 100, 600, 1500] {
            let mut v: Vec<f32> = (0..n).map(|i| ((i * 7919) % 101) as f32).collect();
            let mut idx: Vec<i32> = (0..n as i32).collect();
            let orig = v.clone();
            introsort_index(&mut v, &mut idx, false);
            assert!(v.windows(2).all(|w| w[0] <= w[1]));
            for (k, &i) in idx.iter().enumerate() {
                assert_eq!(orig[i as usize], v[k]);
            }
            introsort_index(&mut v, &mut idx, true);
            assert!(v.windows(2).all(|w| w[0] >= w[1]));
        }
    }

    #[test]
    fn one_clear_changepoint() {
        let mut p = vec![0f32; 50];
        p[20] = 0.6;
        p[21] = 0.3;
        let c = find(&p, 0.01, 3, 5, 3, 3);
        assert_eq!(c.pos, vec![20]);
        assert!((c.prob[0] - 0.9).abs() < 1e-6);
    }
}
