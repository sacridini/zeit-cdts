//! SNIC superpixels (Achanta & Süsstrunk, CVPR 2017) of multiband images and
//! image time series: a port of src/snic.cpp, which gives the reference
//! implementation's labels from the same seeds. Same operations in the same
//! order, so both give the same bits: the 1-based min-heap with strict
//! comparisons (left child on ties), the 4-neighbours visited left, up, right,
//! down, and the colour distance summed as Eigen sums it (in packets for
//! float64 data; one value at a time for float32, whose cast to double Eigen
//! 3.4 does not vectorize).

use crate::eigen;
use crate::parallel;
use rayon::prelude::*;

/// A feature value: float32 or float64, accumulated in float64.
pub trait Feature: Copy + Send + Sync + 'static {
    fn to_f64(self) -> f64;
    /// The sum of the squared differences of `sum - n * x` over the features,
    /// in the order Eigen's `squaredNorm()` takes for this type.
    fn colordist(sum: &[f64], n: f64, x: &[Self]) -> f64;
}

impl Feature for f64 {
    #[inline]
    fn to_f64(self) -> f64 {
        self
    }
    #[inline(always)]
    fn colordist(sum: &[f64], n: f64, x: &[f64]) -> f64 {
        eigen::sum_zip(sum, x, |s, v| {
            let d = s - n * v;
            d * d
        })
    }
}

impl Feature for f32 {
    #[inline]
    fn to_f64(self) -> f64 {
        self as f64
    }
    #[inline(always)]
    fn colordist(sum: &[f64], n: f64, x: &[f32]) -> f64 {
        let x = &x[..sum.len()];
        let d0 = sum[0] - n * x[0] as f64;
        let mut res = d0 * d0;
        for (&s, &v) in sum[1..].iter().zip(&x[1..]) {
            let d = s - n * v as f64;
            res += d * d;
        }
        res
    }
}

#[derive(Clone, Copy)]
struct Node {
    i: u32,
    k: u32,
    d: f64,
}

/// 1-based binary min-heap: push sifts up while the parent is strictly
/// greater, pop sifts the last node down through the strictly smaller child
/// (the left one on ties). With equal distances this order decides which
/// cluster wins a pixel both reach at the same cost; it is the reference's.
struct Heap {
    nodes: Vec<Node>,
    len: usize,
}

impl Heap {
    fn new(capacity: usize) -> Heap {
        Heap {
            nodes: vec![Node { i: 0, k: 0, d: 0.0 }; capacity + 1],
            len: 0,
        }
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    // The indices below stay in 1..=len, and len < nodes.len() (push grows the
    // vector first): the accesses skip the bounds checks, the heap being the
    // hot loop of the segmentation.

    fn push(&mut self, ind: u32, klab: u32, dist: f64) {
        if self.len + 1 >= self.nodes.len() {
            let grown = self.nodes.len() * 2;
            self.nodes.resize(grown, Node { i: 0, k: 0, d: 0.0 });
        }
        self.len += 1;
        let nodes = self.nodes.as_mut_ptr();
        let mut i = self.len;
        let mut j = i / 2;
        // SAFETY: 1 <= j < i <= len < nodes.len()
        unsafe {
            while i > 1 && (*nodes.add(j)).d > dist {
                *nodes.add(i) = *nodes.add(j);
                i = j;
                j /= 2;
            }
            *nodes.add(i) = Node {
                i: ind,
                k: klab,
                d: dist,
            };
        }
    }

    fn pop(&mut self) -> Node {
        assert!(self.len > 0, "pop of an empty heap");
        let nodes = self.nodes.as_mut_ptr();
        // SAFETY: every index read or written is in 1..=len (before the
        // decrement), and len < nodes.len()
        unsafe {
            let top = *nodes.add(1);
            let last = *nodes.add(self.len);
            self.len -= 1;
            let len = self.len;
            let mut i = 1;
            // The C++'s sift-down, written with fewer branches: the smaller
            // child (the left one unless the right is strictly smaller) takes
            // the hole while strictly smaller than the last node; the same
            // child moves in every case, ties included.
            //
            // The choice of the child is arithmetic, not a branch: which one
            // is smaller is a coin toss the branch predictor misses half the
            // time, at every level (MSVC makes it a cmov; LLVM made it a
            // branch, and SNIC 30% slower than the C++). Reading the right
            // child when there is none is fine: index len + 1 is the old last
            // node, inside the vector, and the comparison is discarded.
            loop {
                let j = 2 * i;
                if j > len {
                    break;
                }
                let right_smaller = (j < len) & ((*nodes.add(j + 1)).d < (*nodes.add(j)).d);
                let c = j + right_smaller as usize;
                if (*nodes.add(c)).d < last.d {
                    *nodes.add(i) = *nodes.add(c);
                    i = c;
                } else {
                    break;
                }
            }
            *nodes.add(i) = last;
            top
        }
    }
}

const UNLABELED: i32 = -1;
const MASKED: i32 = -2;

/// A planar image: `data[f * rows * cols + r * cols + c]`.
#[derive(Clone, Copy)]
pub struct Image<'a, T> {
    pub data: &'a [T],
    pub features: usize,
    pub rows: usize,
    pub cols: usize,
}

struct Tile {
    r0: usize,
    c0: usize,
    h: usize,
    w: usize,
    /// Global seed indices, in input order.
    seed_ids: Vec<usize>,
}

/// What a tile gives back: its labels (global seed index or -1) and, per seed
/// of the tile, the feature sums, the coordinate sums and the size.
struct TileResult {
    labels: Vec<i32>,
    sums: Vec<f64>,
    sum_row: Vec<f64>,
    sum_col: Vec<f64>,
    size: Vec<f64>,
}

/// SNIC's growth from the seeds in `heap`, over a w x h tile of `n_valid`
/// unmasked pixels (`px` pixel-major, `f_n` features), into the labels and, per
/// cluster, the feature sums, coordinate sums and size.
///
/// A function of its own, the heap by value and the buffers as arguments: LLVM
/// then keeps more of the loop's state in registers (5% faster than inline).
#[allow(clippy::too_many_arguments)]
#[inline(never)]
fn grow<T: Feature>(
    px: &[T],
    labels: &mut [i32],
    mut heap: Heap,
    ksum: &mut [f64],
    kx: &mut [f64],
    ky: &mut [f64],
    ksize: &mut [f64],
    f_n: usize,
    w: usize,
    h: usize,
    n_valid: usize,
    invwt: f64,
) {
    let mut pixelcount = 0;
    while pixelcount < n_valid && !heap.is_empty() {
        let node = heap.pop();
        let (k, i) = (node.k as usize, node.i as usize);
        if labels[i] != UNLABELED {
            continue;
        }
        // 32-bit division, as the C++'s: tiles hold fewer than 2^32 pixels
        let (x, y) = ((node.i % w as u32) as usize, (node.i / w as u32) as usize);
        labels[i] = k as i32;
        pixelcount += 1;

        let kc = &mut ksum[k * f_n..(k + 1) * f_n];
        for (acc, v) in kc.iter_mut().zip(&px[i * f_n..(i + 1) * f_n]) {
            *acc += v.to_f64();
        }
        kx[k] += x as f64;
        ky[k] += y as f64;
        ksize[k] += 1.0;
        let (sx, sy, nk) = (kx[k], ky[k], ksize[k]);
        let kc = &ksum[k * f_n..(k + 1) * f_n];

        // A macro, not a closure: LLVM kept the closure out of line, a call
        // for every neighbour of every pixel.
        macro_rules! visit {
            ($ii:expr, $xx:expr, $yy:expr) => {{
                let ii: usize = $ii;
                if labels[ii] == UNLABELED {
                    let colordist = T::colordist(kc, nk, &px[ii * f_n..(ii + 1) * f_n]);
                    let xdiff = sx - $xx as f64 * nk;
                    let ydiff = sy - $yy as f64 * nk;
                    let xydist = xdiff * xdiff + ydiff * ydiff;
                    // normalised by n^2 once, instead of dividing every sum by n
                    heap.push(ii as u32, k as u32, (colordist + xydist * invwt) / (nk * nk));
                }
            }};
        }
        // the 4-neighbours: left, up, right, down
        if x > 0 {
            visit!(i - 1, x - 1, y);
        }
        if y > 0 {
            visit!(i - w, x, y - 1);
        }
        if x + 1 < w {
            visit!(i + 1, x + 1, y);
        }
        if y + 1 < h {
            visit!(i + w, x, y + 1);
        }
    }
}

fn segment_tile<T: Feature>(
    img: Image<T>,
    tile: &Tile,
    seeds: &[(usize, usize)],
    compactness: f64,
    gather_jobs: i32,
) -> TileResult {
    let (f_n, w, h) = (img.features, tile.w, tile.h);
    let n = w * h;
    let plane = img.rows * img.cols;
    let k_n = tile.seed_ids.len();

    // The tile pixel-major ([pixel, feature]), so every distance reads one
    // contiguous run of features, and the pixels with any NaN masked.
    let mut px: Vec<T> = vec![img.data[0]; n * f_n];
    let mut labels = vec![UNLABELED; n];
    let gather = |r: usize, row: &mut [T], lab: &mut [i32]| {
        let src_row = (tile.r0 + r) * img.cols + tile.c0;
        for f in 0..f_n {
            let src = &img.data[f * plane + src_row..f * plane + src_row + w];
            for (c, &v) in src.iter().enumerate() {
                row[c * f_n + f] = v;
            }
        }
        for c in 0..w {
            if row[c * f_n..(c + 1) * f_n].iter().any(|v| v.to_f64().is_nan()) {
                lab[c] = MASKED;
            }
        }
    };
    if parallel::threads(gather_jobs) > 1 {
        parallel::pool(parallel::threads(gather_jobs)).install(|| {
            px.par_chunks_mut(w * f_n)
                .zip(labels.par_chunks_mut(w))
                .enumerate()
                .for_each(|(r, (row, lab))| gather(r, row, lab));
        });
    } else {
        for (r, (row, lab)) in px.chunks_mut(w * f_n).zip(labels.chunks_mut(w)).enumerate() {
            gather(r, row, lab);
        }
    }
    let n_valid = labels.iter().filter(|&&l| l == UNLABELED).count();

    // Seeds on masked pixels never enter the heap (their segment stays empty).
    let mut heap = Heap::new(n);
    let mut k_valid = 0usize;
    for (k, &s) in tile.seed_ids.iter().enumerate() {
        let (r, c) = (seeds[s].0 - tile.r0, seeds[s].1 - tile.c0);
        let i = r * w + c;
        if labels[i] == MASKED {
            continue;
        }
        heap.push(i as u32, k as u32, 0.0);
        k_valid += 1;
    }

    let mut ksum = vec![0.0; k_n * f_n];
    let (mut kx, mut ky, mut ksize) = (vec![0.0; k_n], vec![0.0; k_n], vec![0.0; k_n]);
    if n_valid == 0 || k_valid == 0 {
        return TileResult {
            labels: vec![-1; n],
            sums: ksum,
            sum_row: ky,
            sum_col: kx,
            size: ksize,
        };
    }

    // Spatial weight M^2 * K / N = (M / S)^2, S = sqrt(N / K) the grid step.
    let invwt = (compactness * compactness * k_valid as f64) / n_valid as f64;
    grow(
        &px,
        &mut labels,
        heap,
        &mut ksum,
        &mut kx,
        &mut ky,
        &mut ksize,
        f_n,
        w,
        h,
        n_valid,
        invwt,
    );

    let labels = labels
        .iter()
        .map(|&l| if l >= 0 { tile.seed_ids[l as usize] as i32 } else { -1 })
        .collect();
    let sum_row = (0..k_n).map(|k| ky[k] + tile.r0 as f64 * ksize[k]).collect();
    let sum_col = (0..k_n).map(|k| kx[k] + tile.c0 as f64 * ksize[k]).collect();
    TileResult {
        labels,
        sums: ksum,
        sum_row,
        sum_col,
        size: ksize,
    }
}

/// The segmentation: labels (rows x cols, the seed index or -1), means
/// (seeds x features), centroids (seeds x 2: row, col) and sizes (seeds).
pub struct Segments {
    pub labels: Vec<i32>,
    pub means: Vec<f64>,
    pub centroids: Vec<f64>,
    pub sizes: Vec<i64>,
}

/// SNIC of `img` from `seeds` (row, col). The image is split into tiles of
/// `tile_height` x `tile_width` (0: the whole extent), segmented independently
/// and in parallel, each seed in the tile that holds it (sits_segment's
/// block-wise strategy); with one tile the result is SNIC on the whole image.
/// Segments whose seed fell on a masked pixel have size 0 and NaN mean and
/// centroid. Errors are the C++'s invalid_argument messages.
pub fn segment<T: Feature>(
    img: Image<T>,
    seeds: &[(i64, i64)],
    compactness: f64,
    tile_height: i64,
    tile_width: i64,
    n_jobs: i32,
) -> Result<Segments, String> {
    if !compactness.is_finite() || compactness < 0.0 {
        return Err("compactness must be finite and >= 0".into());
    }
    let (f_n, rows, cols) = (img.features, img.rows, img.cols);
    if f_n < 1 || rows < 1 || cols < 1 {
        return Err("data must have at least one feature, row and column".into());
    }
    let k_n = seeds.len();
    if k_n < 1 {
        return Err("at least one seed is required".into());
    }
    if k_n > i32::MAX as usize {
        return Err("too many seeds".into());
    }
    let th = if tile_height > 0 {
        (tile_height as usize).min(rows)
    } else {
        rows
    };
    let tw = if tile_width > 0 {
        (tile_width as usize).min(cols)
    } else {
        cols
    };
    if (th as u64) * (tw as u64) >= u32::MAX as u64 {
        return Err("tile too large (>= 2^32 pixels); set tile_height/tile_width".into());
    }

    let (ntr, ntc) = (rows.div_ceil(th), cols.div_ceil(tw));
    let mut tiles: Vec<Tile> = (0..ntr * ntc)
        .map(|t| {
            let (r0, c0) = ((t / ntc) * th, (t % ntc) * tw);
            Tile {
                r0,
                c0,
                h: th.min(rows - r0),
                w: tw.min(cols - c0),
                seed_ids: Vec::new(),
            }
        })
        .collect();
    let mut seeds_rc = Vec::with_capacity(k_n);
    for (s, &(r, c)) in seeds.iter().enumerate() {
        if r < 0 || r >= rows as i64 || c < 0 || c >= cols as i64 {
            return Err("seed outside the image".into());
        }
        let (r, c) = (r as usize, c as usize);
        seeds_rc.push((r, c));
        tiles[(r / th) * ntc + c / tw].seed_ids.push(s);
    }

    let results: Vec<TileResult> = if tiles.len() == 1 {
        vec![segment_tile(img, &tiles[0], &seeds_rc, compactness, n_jobs)]
    } else {
        let threads = parallel::threads(n_jobs);
        let run = |t: &Tile| segment_tile(img, t, &seeds_rc, compactness, 1);
        if threads == 1 {
            tiles.iter().map(run).collect()
        } else {
            parallel::pool(threads).install(|| tiles.par_iter().with_max_len(1).map(run).collect())
        }
    };

    let mut labels = vec![-1i32; rows * cols];
    let mut means = vec![0.0; k_n * f_n];
    let (mut sum_row, mut sum_col) = (vec![0.0; k_n], vec![0.0; k_n]);
    let mut sizes = vec![0i64; k_n];
    for (tile, res) in tiles.iter().zip(&results) {
        for r in 0..tile.h {
            let dst = (tile.r0 + r) * cols + tile.c0;
            labels[dst..dst + tile.w].copy_from_slice(&res.labels[r * tile.w..(r + 1) * tile.w]);
        }
        for (k, &s) in tile.seed_ids.iter().enumerate() {
            means[s * f_n..(s + 1) * f_n].copy_from_slice(&res.sums[k * f_n..(k + 1) * f_n]);
            sum_row[s] = res.sum_row[k];
            sum_col[s] = res.sum_col[k];
            sizes[s] = res.size[k] as i64;
        }
    }

    let mut centroids = vec![0.0; k_n * 2];
    for s in 0..k_n {
        let sz = sizes[s] as f64;
        if sz > 0.0 {
            for v in &mut means[s * f_n..(s + 1) * f_n] {
                *v /= sz;
            }
            centroids[2 * s] = sum_row[s] / sz;
            centroids[2 * s + 1] = sum_col[s] / sz;
        } else {
            means[s * f_n..(s + 1) * f_n].fill(f64::NAN);
            centroids[2 * s] = f64::NAN;
            centroids[2 * s + 1] = f64::NAN;
        }
    }
    Ok(Segments {
        labels,
        means,
        centroids,
        sizes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heap_pops_in_order_with_ties_left_first() {
        let mut h = Heap::new(2);
        for (i, d) in [3.0, 1.0, 2.0, 1.0, 0.5, 2.0].iter().enumerate() {
            h.push(i as u32, 0, *d);
        }
        let order: Vec<u32> = (0..6).map(|_| h.pop().i).collect();
        let ds: Vec<f64> = order
            .iter()
            .map(|&i| [3.0, 1.0, 2.0, 1.0, 0.5, 2.0][i as usize])
            .collect();
        assert!(ds.windows(2).all(|w| w[0] <= w[1]));
        assert!(h.is_empty());
    }

    #[test]
    fn two_halves_two_segments() {
        let (rows, cols) = (8, 8);
        let data: Vec<f64> = (0..rows * cols)
            .map(|i| if i % cols < 4 { 0.0 } else { 10.0 })
            .collect();
        let img = Image {
            data: &data,
            features: 1,
            rows,
            cols,
        };
        for (th, tw) in [(0, 0), (4, 4)] {
            let s = segment(img, &[(4, 1), (4, 6)], 0.01, th, tw, 2).unwrap();
            if th == 0 {
                assert!((0..rows * cols).all(|i| s.labels[i] == if i % cols < 4 { 0 } else { 1 }));
                assert_eq!(s.sizes, vec![32, 32]);
                assert_eq!(s.means, vec![0.0, 10.0]);
            }
        }
    }

    #[test]
    fn errors() {
        let data = vec![0.0f32; 4];
        let img = Image {
            data: &data,
            features: 1,
            rows: 2,
            cols: 2,
        };
        assert!(segment(img, &[], 1.0, 0, 0, 1).is_err());
        assert!(segment(img, &[(2, 0)], 1.0, 0, 0, 1).is_err());
        assert!(segment(img, &[(0, 0)], f64::NAN, 0, 0, 1).is_err());
        assert!(segment(img, &[(0, 0)], 1.0, 0, 0, 1).is_ok());
    }
}
