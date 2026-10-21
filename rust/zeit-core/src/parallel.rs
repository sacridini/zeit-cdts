//! Threads, as OpenMP gave them to the C++ core: `n_jobs` > 0 threads, or all
//! the processors but one for `n_jobs` <= 0.
//!
//! One rayon pool per thread count, made on first use and kept: a pool costs
//! tens of microseconds to start, too much for every call on small arrays.

use rayon::prelude::*;
use std::collections::HashMap;
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

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

/// The threads of `region`: `threads - 1` workers kept for the life of the
/// process, the caller being the team's thread 0.
struct Team {
    workers: usize,
    /// The region running: its function, valid until every worker is done.
    job: AtomicPtr<()>,
    generation: AtomicUsize,
    done: AtomicUsize,
    panicked: AtomicBool,
    sleepers: AtomicUsize,
    lock: Mutex<()>,
    wake: Condvar,
    /// One region at a time.
    busy: Mutex<()>,
}

thread_local! {
    /// The index of the thread in the team of the region it runs, if any.
    static TEAM_INDEX: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

type Job<'a> = &'a (dyn Fn(usize) + Sync);

impl Team {
    /// How long the workers wait, spinning then yielding, for the next region
    /// before they sleep: OpenMP's threads (GOMP_SPINCOUNT) wait in the same
    /// way, so that calls one after the other find them awake.
    const SPIN: Duration = Duration::from_micros(50);
    const YIELD: Duration = Duration::from_millis(2);

    fn start(threads: usize) -> Arc<Team> {
        let team = Arc::new(Team {
            workers: threads - 1,
            job: AtomicPtr::new(std::ptr::null_mut()),
            generation: AtomicUsize::new(0),
            done: AtomicUsize::new(0),
            panicked: AtomicBool::new(false),
            sleepers: AtomicUsize::new(0),
            lock: Mutex::new(()),
            wake: Condvar::new(),
            busy: Mutex::new(()),
        });
        for tid in 1..threads {
            let team = team.clone();
            std::thread::Builder::new()
                .name(format!("zeit-team-{tid}"))
                .spawn(move || team.work(tid))
                .expect("could not start a thread");
        }
        team
    }

    fn work(&self, tid: usize) {
        TEAM_INDEX.with(|t| t.set(Some(tid)));
        let mut seen = 0;
        loop {
            seen = self.next_generation(seen);
            // SAFETY: `region` stored a `&Job` that outlives the region, and
            // waits for every worker before it returns.
            let job = unsafe { *(self.job.load(Ordering::Acquire) as *const Job) };
            if catch_unwind(AssertUnwindSafe(|| job(tid))).is_err() {
                self.panicked.store(true, Ordering::Relaxed);
            }
            self.done.fetch_add(1, Ordering::Release);
        }
    }

    /// Waits for a generation after `seen`: spins, yields, then sleeps.
    fn next_generation(&self, seen: usize) -> usize {
        let start = Instant::now();
        let mut k = 0u32;
        loop {
            let g = self.generation.load(Ordering::Acquire);
            if g != seen {
                return g;
            }
            k = k.wrapping_add(1);
            if k.is_multiple_of(64) {
                let waited = start.elapsed();
                if waited > Self::YIELD {
                    break;
                } else if waited > Self::SPIN {
                    std::thread::yield_now();
                    continue;
                }
            }
            std::hint::spin_loop();
        }
        let mut guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        self.sleepers.fetch_add(1, Ordering::AcqRel);
        while self.generation.load(Ordering::Acquire) == seen {
            guard = self.wake.wait(guard).unwrap_or_else(|e| e.into_inner());
        }
        self.sleepers.fetch_sub(1, Ordering::AcqRel);
        self.generation.load(Ordering::Acquire)
    }

    fn run(&self, f: Job) {
        self.done.store(0, Ordering::Relaxed);
        self.panicked.store(false, Ordering::Relaxed);
        self.job.store(&f as *const Job as *mut (), Ordering::Release);
        {
            // under the lock, so that a worker about to sleep sees it or is woken
            let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
            self.generation.fetch_add(1, Ordering::Release);
        }
        if self.sleepers.load(Ordering::Acquire) > 0 {
            self.wake.notify_all();
        }
        let mine = catch_unwind(AssertUnwindSafe(|| {
            TEAM_INDEX.with(|t| t.set(Some(0)));
            f(0);
        }));
        TEAM_INDEX.with(|t| t.set(None));
        let mut k = 0u32;
        while self.done.load(Ordering::Acquire) < self.workers {
            k = k.wrapping_add(1);
            if k < 1 << 12 {
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        if let Err(e) = mine {
            resume_unwind(e);
        }
        if self.panicked.load(Ordering::Relaxed) {
            panic!("a thread of the team panicked");
        }
    }
}

/// `f(tid)` on each of `threads` threads, the calling one being 0 (OpenMP's
/// parallel region). The threads are kept between calls and wait a little
/// for the next region before they sleep, so short calls one after another do
/// not pay for waking them; a rayon pool's threads fall asleep at once. When
/// the team is busy (a region from another thread), rayon's pool runs it.
pub fn region<F: Fn(usize) + Sync>(threads: usize, f: F) {
    if threads <= 1 {
        f(0);
        return;
    }
    static TEAMS: OnceLock<Mutex<HashMap<usize, Arc<Team>>>> = OnceLock::new();
    let team = TEAMS
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .entry(threads)
        .or_insert_with(|| Team::start(threads))
        .clone();
    let inside = TEAM_INDEX.with(|t| t.get()).is_some();
    match team.busy.try_lock() {
        Ok(_busy) if !inside => team.run(&f),
        _ => {
            pool(threads).broadcast(|ctx| f(ctx.index()));
        }
    };
}

/// The index of the calling thread in its `region` team, or in the rayon
/// pool running it; 0 elsewhere.
pub fn thread_index() -> usize {
    TEAM_INDEX
        .with(|t| t.get())
        .or_else(rayon::current_thread_index)
        .unwrap_or(0)
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
    fn a_region_runs_every_thread_once() {
        for threads in [1, 2, 5] {
            for _ in 0..200 {
                let seen: Vec<AtomicUsize> = (0..threads).map(|_| AtomicUsize::new(0)).collect();
                region(threads, |t| {
                    assert_eq!(thread_index(), t);
                    seen[t].fetch_add(1, Ordering::Relaxed);
                });
                assert!(seen.iter().all(|s| s.load(Ordering::Relaxed) == 1));
            }
        }
    }

    #[test]
    fn regions_nest_and_run_side_by_side() {
        let total = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    for _ in 0..50 {
                        region(3, |_| {
                            region(3, |_| {
                                total.fetch_add(1, Ordering::Relaxed);
                            })
                        });
                    }
                });
            }
        });
        assert_eq!(total.load(Ordering::Relaxed), 4 * 50 * 3 * 3);
    }

    #[test]
    fn a_panic_in_a_region_reaches_the_caller() {
        let r = std::panic::catch_unwind(|| region(4, |t| assert!(t != 2, "thread 2")));
        assert!(r.is_err());
        // the team is still usable
        let n = AtomicUsize::new(0);
        region(4, |_| {
            n.fetch_add(1, Ordering::Relaxed);
        });
        assert_eq!(n.load(Ordering::Relaxed), 4);
    }

    #[test]
    fn transposes() {
        assert_eq!(transpose(&[1, 2, 3, 4, 5, 6], 2, 3), vec![1, 4, 2, 5, 3, 6]);
    }
}
