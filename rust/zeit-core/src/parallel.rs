//! Threads, as OpenMP gave them to the C++ core: `n_jobs` > 0 threads, or all
//! the processors but one for `n_jobs` <= 0.
//!
//! One rayon pool per thread count, made on first use and kept: a pool costs
//! tens of microseconds to start, too much for every call on small arrays.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

/// The number of threads `n_jobs` asks for.
pub fn threads(n_jobs: i32) -> usize {
    if n_jobs > 0 {
        n_jobs as usize
    } else {
        let procs = std::thread::available_parallelism().map_or(1, |n| n.get());
        procs.saturating_sub(1).max(1)
    }
}

/// The pool of `threads` threads.
pub fn pool(threads: usize) -> Arc<rayon::ThreadPool> {
    static POOLS: OnceLock<Mutex<HashMap<usize, Arc<rayon::ThreadPool>>>> = OnceLock::new();
    let mut pools = POOLS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    pools
        .entry(threads)
        .or_insert_with(|| {
            Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .thread_name(|i| format!("zeit-{i}"))
                    .build()
                    .expect("could not start the thread pool"),
            )
        })
        .clone()
}

/// Calls `f(&mut state, i, row)` for every row `i` of `out` (`width` values each),
/// on the threads of `n_jobs`, with one `state` from `init` per task: the scratch
/// buffers OpenMP's threads allocated once in the C++ core. With one thread, the
/// rows run in order on the calling thread.
pub fn for_each_row<T, S, I, F>(out: &mut [T], width: usize, n_jobs: i32, init: I, f: F)
where
    T: Send,
    I: Fn() -> S + Sync + Send,
    F: Fn(&mut S, usize, &mut [T]) + Sync + Send,
{
    if width == 0 || out.is_empty() {
        return;
    }
    let threads = threads(n_jobs);
    if threads == 1 {
        let mut state = init();
        for (i, row) in out.chunks_mut(width).enumerate() {
            f(&mut state, i, row);
        }
    } else {
        pool(threads).install(|| {
            out.par_chunks_mut(width)
                .enumerate()
                .for_each_init(&init, |state, (i, row)| f(state, i, row));
        });
    }
}

/// `rows` x `cols` values, row by row, as `cols` x `rows`.
pub fn transpose<T: Copy>(values: &[T], rows: usize, cols: usize) -> Vec<T> {
    debug_assert_eq!(values.len(), rows * cols);
    let mut out = Vec::with_capacity(values.len());
    for c in 0..cols {
        out.extend((0..rows).map(|r| values[r * cols + c]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_see_their_index_with_any_threads() {
        for n_jobs in [1, 3, -1] {
            let mut out = vec![0usize; 1000 * 3];
            for_each_row(&mut out, 3, n_jobs, || (), |_, i, row| row.fill(i));
            assert!(out.chunks(3).enumerate().all(|(i, r)| r.iter().all(|&v| v == i)));
        }
    }

    #[test]
    fn transposes() {
        assert_eq!(transpose(&[1, 2, 3, 4, 5, 6], 2, 3), vec![1, 4, 2, 5, 3, 6]);
    }
}
