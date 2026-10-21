//! Threads, as OpenMP gave them to the C++ core: `n_jobs` > 0 threads, or all
//! the processors but one for `n_jobs` <= 0.
//!
//! One rayon pool per thread count, made on first use and kept: a pool costs
//! tens of microseconds to start, too much for every call on small arrays.

use rayon::prelude::*;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

/// The number of threads `n_jobs` asks for.
pub fn threads(n_jobs: i32) -> usize {
    if n_jobs > 0 {
        n_jobs as usize
    } else {
        let procs = std::thread::available_parallelism().map_or(1, |n| n.get());
        procs.saturating_sub(1).max(1)
    }
}

/// The `n_jobs` of every processor: the thread count of OpenMP's default
/// team, which the C++ functions without an `n_jobs` used.
pub fn all_processors() -> i32 {
    std::thread::available_parallelism().map_or(1, |n| n.get() as i32)
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

/// A barrier that spins a little before sleeping, as OpenMP's: the phases of
/// an iterative algorithm are often too short for threads to fall asleep and be
/// woken between them, but spinning long starves the threads still working
/// when there are more threads than free processors.
pub struct SpinBarrier {
    n: usize,
    count: AtomicUsize,
    generation: AtomicUsize,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
}

impl SpinBarrier {
    /// Spins before sleeping (a few tens of microseconds).
    const SPINS: u32 = 1 << 10;

    pub fn new(n: usize) -> SpinBarrier {
        SpinBarrier {
            n,
            count: AtomicUsize::new(0),
            generation: AtomicUsize::new(0),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
        }
    }

    /// Waits for the `n` threads; what each wrote before is seen by all after.
    pub fn wait(&self) {
        let generation = self.generation.load(Ordering::Acquire);
        if self.count.fetch_add(1, Ordering::AcqRel) + 1 == self.n {
            self.count.store(0, Ordering::Relaxed);
            {
                // under the lock, so that a thread about to sleep sees it or is woken
                let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
                self.generation.fetch_add(1, Ordering::Release);
            }
            if self.sleepers.load(Ordering::Acquire) > 0 {
                self.wake.notify_all();
            }
            return;
        }
        for _ in 0..Self::SPINS {
            if self.generation.load(Ordering::Acquire) != generation {
                return;
            }
            std::hint::spin_loop();
        }
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.sleepers.fetch_add(1, Ordering::AcqRel);
        while self.generation.load(Ordering::Acquire) == generation {
            guard = self.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        self.sleepers.fetch_sub(1, Ordering::AcqRel);
    }
}

/// A team of threads for the whole of an iterative computation (OpenMP's
/// parallel region): `f(tid, &barrier)` runs once on each of the `threads`
/// threads of the pool, which stay up and meet at the barrier between phases.
pub fn team<F: Fn(usize, &SpinBarrier) + Sync>(threads: usize, f: F) {
    let barrier = SpinBarrier::new(threads);
    pool(threads).broadcast(|ctx| f(ctx.index(), &barrier));
}

/// A slice that the threads of a team write in disjoint parts between
/// barriers (OpenMP's shared arrays).
#[derive(Clone, Copy)]
pub struct Shared<T> {
    ptr: *mut T,
    len: usize,
}

// SAFETY: the users of a Shared write disjoint elements between barriers, and
// read only what the barriers have published.
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}

impl<T> Shared<T> {
    pub fn new(slice: &mut [T]) -> Shared<T> {
        Shared {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// The elements `range`, mutable.
    ///
    /// # Safety
    /// No other thread reads or writes them until the next barrier.
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn slice_mut(&self, range: std::ops::Range<usize>) -> &mut [T] {
        assert!(range.start <= range.end && range.end <= self.len);
        unsafe { std::slice::from_raw_parts_mut(self.ptr.add(range.start), range.end - range.start) }
    }

    /// All the elements, shared.
    ///
    /// # Safety
    /// No thread writes them until the next barrier.
    pub unsafe fn slice(&self) -> &[T] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
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
