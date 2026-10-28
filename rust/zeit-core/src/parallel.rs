//! Threads: `n_jobs` > 0 threads, or for `n_jobs` <= 0 the processors zeit may
//! use, decided by `detect` from (the first that applies):
//!
//! 1. ZEIT_NUM_THREADS, as it is (zeit's tiles set it for each tile's process);
//! 2. `zeit config threads`, the user's setting (config.toml of the config
//!    folder), outside a job;
//! 3. OMP_NUM_THREADS, as it is (as OpenMP's omp_get_max_threads);
//! 4. the job's processors from its scheduler's variables (SLURM_CPUS_PER_TASK,
//!    SLURM_CPUS_ON_NODE, SLURM_JOB_CPUS_PER_NODE; NCPUS, PBS_NUM_PPN;
//!    LSB_DJOB_NUMPROC), when the processors this process may run on are more
//!    (a cluster that does not confine its jobs with a cgroup);
//! 5. the processors this process may run on, its affinity and cgroup quota
//!    (std's available_parallelism): all of them in a job (Slurm, PBS, LSF) or
//!    when the process is confined to fewer than the machine has, where the
//!    others belong to someone else; on a machine several users are logged in
//!    to at once (Remote Desktop, SSH), an equal share for each; elsewhere all
//!    but one. On the login node of a cluster, at most `LOGIN_THREADS`.
//!
//! One rayon pool per thread count, made on first use and kept: a pool costs
//! tens of microseconds to start, too much for every call on small arrays.

use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// The variable that names the job of each batch scheduler.
pub const JOB_VARIABLES: [(&str, &str); 3] = [("Slurm", "SLURM_JOB_ID"), ("PBS", "PBS_JOBID"), ("LSF", "LSB_JOBID")];

/// The variables that give a job's processors on this node, by scheduler, the
/// first set counting: Slurm's without `--cpus-per-task` (`--ntasks=8`), PBS
/// Pro's and Torque's, LSF's.
pub const JOB_PROCESSORS: [(&str, &str); 6] = [
    ("Slurm", "SLURM_CPUS_PER_TASK"),
    ("Slurm", "SLURM_CPUS_ON_NODE"),
    ("Slurm", "SLURM_JOB_CPUS_PER_NODE"),
    ("PBS", "NCPUS"),
    ("PBS", "PBS_NUM_PPN"),
    ("LSF", "LSB_DJOB_NUMPROC"),
];

/// What `variable` is when `zeit config threads` gave the count.
pub const CONFIG: &str = "zeit config threads";

/// The most threads of `n_jobs` <= 0 on a cluster's login node, outside a job:
/// everyone's, and meant for editing and submitting.
pub const LOGIN_THREADS: usize = 4;

/// How long what `detect` reads of the machine (its users, the setting) is
/// kept before it is read again.
const REFRESH: Duration = Duration::from_secs(30);

/// What `detect` reads of the machine, besides the variables.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Host {
    /// The processors this process may run on (affinity and cgroup quota).
    pub available: usize,
    /// The processors of the machine.
    pub machine: usize,
    /// The distinct users with an active session and where they were read
    /// ("utmp", "logind", "WTS"), or None where that cannot be read.
    pub users: Option<(usize, &'static str)>,
    /// The scheduler whose configuration is on this machine, if any: outside a
    /// job, a login node.
    pub scheduler: Option<&'static str>,
    /// `zeit config threads`, if set.
    pub configured: Option<usize>,
}

/// What `detect` found: the threads of `n_jobs` <= 0 and where they came from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detected {
    /// The threads of `n_jobs` <= 0.
    pub threads: usize,
    /// Every processor zeit may use (`threads` with the one kept free).
    pub all: usize,
    /// The variable that gave the count (ZEIT_NUM_THREADS, `CONFIG`,
    /// OMP_NUM_THREADS or one of `JOB_PROCESSORS`), or None for the processors
    /// available.
    pub variable: Option<&'static str>,
    /// The processors this process may run on (affinity and cgroup quota).
    pub available: usize,
    /// The processors of the machine.
    pub machine: usize,
    /// The scheduler and id of the job this process runs in, if any.
    pub job: Option<(&'static str, String)>,
    /// The processors of the job (those available, or fewer from its
    /// scheduler's variables), or those available outside a job.
    pub job_processors: usize,
    /// One processor is kept free (a desktop, outside a job).
    pub reserved: bool,
    /// The distinct users with an active session and where they were read.
    pub users: Option<(usize, &'static str)>,
    /// The processors are shared among `users` (more than one).
    pub shared: bool,
    /// The scheduler of the cluster whose login node this is (outside a job).
    pub login_node: Option<&'static str>,
}

/// The processors this process may run on, and those of the machine: read once
/// (the cgroup files on Linux), as OpenMP does.
fn processors() -> (usize, usize) {
    static CPUS: OnceLock<(usize, usize)> = OnceLock::new();
    *CPUS.get_or_init(|| {
        let available = std::thread::available_parallelism().map_or(1, |n| n.get());
        (available, online_processors().unwrap_or(available).max(available))
    })
}

/// The processors online in the machine (Linux), from /sys/devices/system/cpu/online.
fn online_processors() -> Option<usize> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    cpu_list_len(std::fs::read_to_string("/sys/devices/system/cpu/online").ok()?.trim())
}

/// The number of processors of a Linux CPU list ("0-3,8-11", "0").
fn cpu_list_len(list: &str) -> Option<usize> {
    let mut n = 0;
    for part in list.split(',').map(str::trim).filter(|p| !p.is_empty()) {
        n += match part.split_once('-') {
            Some((a, b)) => b.parse::<usize>().ok()?.checked_sub(a.parse::<usize>().ok()?)? + 1,
            None => {
                part.parse::<usize>().ok()?;
                1
            }
        };
    }
    (n > 0).then_some(n)
}

/// A positive thread count from a variable's value ("8", or "8,4" for OpenMP's
/// nested levels, of which the first counts).
fn count(v: &str) -> Option<usize> {
    v.split(',').next()?.trim().parse::<usize>().ok().filter(|&n| n > 0)
}

/// The processors of each node from SLURM_JOB_CPUS_PER_NODE ("8", "8(x2)"),
/// when every node has as many ("8(x2),4" does not say which is this one).
fn uniform_count(v: &str) -> Option<usize> {
    let v = v.trim();
    if v.contains(',') {
        return None;
    }
    count(v.split_once('(').map_or(v, |(n, _)| n))
}

/// The rules of the module, on the variables `var` gives and what was read of
/// the machine.
pub fn decide(var: &dyn Fn(&str) -> Option<String>, host: &Host) -> Detected {
    let available = host.available.max(1);
    let machine = host.machine.max(available);
    let job = JOB_VARIABLES.iter().find_map(|&(name, v)| {
        var(v)
            .map(|id| id.trim().to_string())
            .filter(|id| !id.is_empty())
            .map(|id| (name, id))
    });
    let from_var = |v: &'static str| var(v).as_deref().and_then(count).map(|n| (n, v));
    // the setting is the user's default for their own sessions, not a job's
    let configured = host.configured.filter(|_| job.is_none()).map(|n| (n, CONFIG));
    let given = from_var("ZEIT_NUM_THREADS")
        .or(configured)
        .or_else(|| from_var("OMP_NUM_THREADS"));
    let from_job = job.as_ref().and_then(|(scheduler, _)| {
        JOB_PROCESSORS
            .iter()
            .filter(|(s, _)| s == scheduler)
            .find_map(|&(_, v)| {
                let value = var(v)?;
                let n = if v == "SLURM_JOB_CPUS_PER_NODE" {
                    uniform_count(&value)
                } else {
                    count(&value)
                };
                Some((n?, v))
            })
            .filter(|&(n, _)| n < available)
    });
    let job_processors = from_job.map_or(available, |(n, _)| n);
    let (all, variable) = match given.or(from_job) {
        Some((n, v)) => (n, Some(v)),
        None => (available, None),
    };
    // nothing chose the count: the processors available, and how to share them
    let default = variable.is_none() && job.is_none();
    let whole_machine = default && available >= machine;
    let users = host.users;
    let shared = whole_machine && users.is_some_and(|(n, _)| n > 1);
    let login_node = host.scheduler.filter(|_| default);
    let threads = match users {
        Some((n, _)) if shared => (all / n).max(1),
        _ if whole_machine => all.saturating_sub(1).max(1),
        _ => all,
    };
    let threads = if login_node.is_some() {
        threads.min(LOGIN_THREADS)
    } else {
        threads
    };
    let reserved = whole_machine && !shared && login_node.is_none();
    Detected {
        threads,
        all: if reserved { all } else { threads },
        variable,
        available,
        machine,
        job,
        job_processors,
        reserved,
        users,
        shared,
        login_node,
    }
}

/// The threads of `n_jobs` <= 0 now: the variables are read at every call (a
/// few lookups), the processors once, the users and the setting every
/// `REFRESH`.
pub fn detect() -> Detected {
    decide(&|v| std::env::var(v).ok(), &host())
}

static HOST: Mutex<Option<(Instant, Host)>> = Mutex::new(None);

/// What `detect` reads of the machine, from the last `REFRESH`.
pub fn host() -> Host {
    let mut seen = HOST.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((at, host)) = seen.as_ref() {
        if at.elapsed() < REFRESH {
            return host.clone();
        }
    }
    let (available, machine) = processors();
    let host = Host {
        available,
        machine,
        users: active_users(),
        scheduler: installed_scheduler(&|v| std::env::var_os(v), &|p| p.exists()),
        configured: configured(),
    };
    *seen = Some((Instant::now(), host.clone()));
    host
}

/// Reads the users and the setting again at the next `detect` (after `zeit
/// config threads` changed it).
pub fn refresh() {
    *HOST.lock().unwrap_or_else(|e| e.into_inner()) = None;
}

/// The folder of zeit's settings, as zeit/_config.py's `config_dir`:
/// ZEIT_CONFIG_DIR, or %APPDATA%\zeit on Windows, $XDG_CONFIG_HOME/zeit or
/// ~/.config/zeit elsewhere.
pub fn config_dir() -> Option<PathBuf> {
    let var = |v: &str| std::env::var_os(v).filter(|s| !s.is_empty()).map(PathBuf::from);
    if let Some(dir) = var("ZEIT_CONFIG_DIR") {
        return Some(dir);
    }
    if cfg!(windows) {
        return Some(var("APPDATA").or_else(|| var("USERPROFILE"))?.join("zeit"));
    }
    Some(
        var("XDG_CONFIG_HOME")
            .or_else(|| Some(var("HOME")?.join(".config")))?
            .join("zeit"),
    )
}

/// `zeit config threads` from config.toml of the config folder.
fn configured() -> Option<usize> {
    setting(
        &std::fs::read_to_string(config_dir()?.join("config.toml")).ok()?,
        "threads",
    )
    .and_then(|v| count(&v))
}

/// The value of `key` in the `key = value` lines of a config.toml.
fn setting(text: &str, key: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (k, v) = line.split('#').next()?.split_once('=')?;
        (k.trim() == key).then(|| v.trim().trim_matches('"').to_string())
    })
}

/// The configuration files of each scheduler: on a machine that has one, a job
/// can be submitted from it.
const SCHEDULER_FILES: [(&str, &str); 6] = [
    ("Slurm", "/etc/slurm/slurm.conf"),
    ("Slurm", "/etc/slurm-llnl/slurm.conf"),
    ("Slurm", "/usr/local/etc/slurm.conf"),
    ("Slurm", "/opt/slurm/etc/slurm.conf"),
    ("PBS", "/etc/pbs.conf"),
    ("LSF", "/etc/lsf.conf"),
];

/// The scheduler whose configuration is on this machine (Unix), from the
/// variables that point to it or its usual files.
fn installed_scheduler(
    var: &dyn Fn(&str) -> Option<std::ffi::OsString>,
    exists: &dyn Fn(&Path) -> bool,
) -> Option<&'static str> {
    if !cfg!(unix) {
        return None;
    }
    [("Slurm", "SLURM_CONF"), ("PBS", "PBS_CONF_FILE"), ("LSF", "LSF_ENVDIR")]
        .into_iter()
        .find_map(|(name, v)| var(v).filter(|s| !s.is_empty()).map(|_| name))
        .or_else(|| {
            SCHEDULER_FILES
                .iter()
                .find_map(|&(name, f)| exists(Path::new(f)).then_some(name))
        })
}

/// The distinct users with an active session: on Linux those of utmp (what
/// `who` lists) or, without it, systemd-logind's sessions; on Windows the
/// sessions in the WTSActive state (what `quser` lists as Active; a Remote
/// Desktop left without logging off is Disconnected and does not count).
/// Neither needs an administrator.
fn active_users() -> Option<(usize, &'static str)> {
    if cfg!(target_os = "linux") {
        if let Some(n) = ["/run/utmp", "/var/run/utmp"]
            .iter()
            .find_map(|f| std::fs::read(f).ok())
            .and_then(|b| utmp_users(&b))
        {
            return Some((n, "utmp"));
        }
        return logind_users(Path::new("/run/systemd/sessions")).map(|n| (n, "logind"));
    }
    windows_users().map(|n| (n, "WTS"))
}

/// The bytes of a struct utmp of glibc on Linux (the same on 64 and 32 bits),
/// and where its fields are.
const UTMP_SIZE: usize = 384;
const UTMP_USER: std::ops::Range<usize> = 44..76;
const USER_PROCESS: i16 = 7;

/// The distinct users of the USER_PROCESS records of utmp's bytes, or None
/// when there are none or the bytes are not glibc's records.
fn utmp_users(bytes: &[u8]) -> Option<usize> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(UTMP_SIZE) {
        return None;
    }
    let users: HashSet<&[u8]> = bytes
        .as_chunks::<UTMP_SIZE>()
        .0
        .iter()
        .filter(|r| i16::from_ne_bytes([r[0], r[1]]) == USER_PROCESS)
        .map(|r| {
            let name = &r[UTMP_USER];
            &name[..name.iter().position(|&b| b == 0).unwrap_or(name.len())]
        })
        .filter(|name| !name.is_empty())
        .collect();
    (!users.is_empty()).then_some(users.len())
}

/// The distinct users of systemd-logind's sessions (a file each, readable by
/// all, in /run/systemd/sessions) of a user's class that are not closing. The
/// FIFOs beside them (1.ref) are skipped: opening one waits for a writer.
fn logind_users(dir: &Path) -> Option<usize> {
    let mut users = HashSet::new();
    for entry in std::fs::read_dir(dir).ok()?.flatten() {
        if !entry.file_type().is_ok_and(|t| t.is_file()) {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(entry.path()) else {
            continue;
        };
        let field = |key: &str| text.lines().find_map(|l| l.strip_prefix(key)?.strip_prefix('='));
        let user = field("USER").or_else(|| field("UID"));
        if field("CLASS").is_some_and(|c| c.starts_with("user")) && field("STATE") != Some("closing") {
            users.extend(user.map(str::to_string));
        }
    }
    (!users.is_empty()).then_some(users.len())
}

#[cfg(not(windows))]
fn windows_users() -> Option<usize> {
    None
}

/// The distinct users (domain\name) of the active sessions of this Windows
/// machine, from the Remote Desktop Services API: a user may list the
/// sessions of the machine they are logged in to.
#[cfg(windows)]
fn windows_users() -> Option<usize> {
    use std::ffi::c_void;
    use std::ptr::null_mut;

    #[repr(C)]
    struct SessionInfo {
        id: u32,
        station: *mut u16,
        state: i32,
    }
    #[link(name = "wtsapi32")]
    extern "system" {
        fn WTSEnumerateSessionsW(
            server: *mut c_void,
            reserved: u32,
            version: u32,
            info: *mut *mut SessionInfo,
            count: *mut u32,
        ) -> i32;
        fn WTSQuerySessionInformationW(
            server: *mut c_void,
            session: u32,
            class: i32,
            buffer: *mut *mut u16,
            bytes: *mut u32,
        ) -> i32;
        fn WTSFreeMemory(memory: *mut c_void);
    }
    const WTS_ACTIVE: i32 = 0;
    const WTS_USER_NAME: i32 = 5;
    const WTS_DOMAIN_NAME: i32 = 7;

    let query = |id: u32, class: i32| -> Option<String> {
        let (mut buffer, mut bytes) = (null_mut(), 0u32);
        // SAFETY: the API allocates the buffer, a NUL-terminated UTF-16 string
        // of `bytes` bytes, which WTSFreeMemory releases.
        unsafe {
            if WTSQuerySessionInformationW(null_mut(), id, class, &mut buffer, &mut bytes) == 0 || buffer.is_null() {
                return None;
            }
            let units = std::slice::from_raw_parts(buffer, bytes as usize / 2);
            let text = String::from_utf16_lossy(&units[..units.iter().position(|&u| u == 0).unwrap_or(units.len())]);
            WTSFreeMemory(buffer.cast());
            Some(text)
        }
    };
    let (mut info, mut count) = (null_mut(), 0u32);
    // SAFETY: on success the API allocates `count` records, released below.
    if unsafe { WTSEnumerateSessionsW(null_mut(), 0, 1, &mut info, &mut count) } == 0 || info.is_null() {
        return None;
    }
    let active: Vec<u32> = unsafe { std::slice::from_raw_parts(info, count as usize) }
        .iter()
        .filter(|s| s.state == WTS_ACTIVE)
        .map(|s| s.id)
        .collect();
    unsafe { WTSFreeMemory(info.cast()) };
    let users: HashSet<String> = active
        .into_iter()
        .filter_map(|id| {
            let user = query(id, WTS_USER_NAME).filter(|u| !u.is_empty())?;
            Some(format!("{}\\{user}", query(id, WTS_DOMAIN_NAME).unwrap_or_default()).to_lowercase())
        })
        .collect();
    (!users.is_empty()).then_some(users.len())
}

/// The number of threads `n_jobs` asks for.
pub fn threads(n_jobs: i32) -> usize {
    if n_jobs > 0 {
        n_jobs as usize
    } else {
        detect().threads
    }
}

/// The `n_jobs` of every processor zeit may use, none kept free: the thread
/// count of OpenMP's default team, which the C++ functions without an `n_jobs`
/// used.
pub fn all_processors() -> i32 {
    detect().all as i32
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
    fn count_values() {
        assert_eq!(count("8"), Some(8));
        assert_eq!(count(" 4 ,2"), Some(4));
        assert_eq!(count("0"), None);
        assert_eq!(count("x"), None);
        assert_eq!(count(""), None);
    }

    #[test]
    fn cpu_lists() {
        assert_eq!(cpu_list_len("0-127"), Some(128));
        assert_eq!(cpu_list_len("0"), Some(1));
        assert_eq!(cpu_list_len("0-3,8-11"), Some(8));
        assert_eq!(cpu_list_len(""), None);
        assert_eq!(cpu_list_len("3-1"), None);
    }

    fn vars(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> {
        let pairs: Vec<(String, String)> = pairs.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect();
        move |name| pairs.iter().find(|(k, _)| k == name).map(|(_, v)| v.clone())
    }

    fn on(available: usize, machine: usize) -> Host {
        Host {
            available,
            machine,
            ..Host::default()
        }
    }

    #[test]
    fn a_desktop_keeps_one_processor() {
        let d = decide(&vars(&[]), &on(16, 16));
        assert_eq!((d.threads, d.all, d.variable, d.reserved), (15, 16, None, true));
        assert_eq!(decide(&vars(&[]), &on(1, 1)).threads, 1);
    }

    #[test]
    fn a_job_or_a_confined_process_uses_every_processor() {
        let d = decide(&vars(&[("SLURM_JOB_ID", "123456")]), &on(16, 128));
        assert_eq!((d.threads, d.reserved), (16, false));
        assert_eq!(d.job, Some(("Slurm", "123456".to_string())));
        assert_eq!(decide(&vars(&[("PBS_JOBID", "7.pbs")]), &on(8, 8)).threads, 8);
        assert_eq!(decide(&vars(&[("LSB_JOBID", "9")]), &on(8, 8)).threads, 8);
        // taskset or a container: the other processors belong to someone else
        assert_eq!(decide(&vars(&[]), &on(2, 64)).threads, 2);
    }

    #[test]
    fn the_variables_in_their_order() {
        let all = [
            ("ZEIT_NUM_THREADS", "6"),
            ("OMP_NUM_THREADS", "1"),
            ("SLURM_CPUS_PER_TASK", "4"),
            ("SLURM_JOB_ID", "1"),
        ];
        let d = decide(&vars(&all), &on(32, 32));
        assert_eq!((d.threads, d.variable), (6, Some("ZEIT_NUM_THREADS")));
        let d = decide(&vars(&all[1..]), &on(32, 32));
        assert_eq!((d.threads, d.variable), (1, Some("OMP_NUM_THREADS")));
        // as they are, on a desktop too
        assert_eq!(decide(&vars(&[("OMP_NUM_THREADS", "8")]), &on(16, 16)).threads, 8);
        // SLURM_CPUS_PER_TASK only where no cgroup confines the job
        let d = decide(&vars(&all[2..]), &on(32, 32));
        assert_eq!((d.threads, d.variable), (4, Some("SLURM_CPUS_PER_TASK")));
        let d = decide(&vars(&all[2..]), &on(4, 32));
        assert_eq!((d.threads, d.variable), (4, None));
        // values that are no count are ignored
        assert_eq!(
            decide(&vars(&[("ZEIT_NUM_THREADS", "0"), ("OMP_NUM_THREADS", "x")]), &on(4, 4)).threads,
            3
        );
    }

    #[test]
    fn a_job_without_cpus_per_task_uses_its_scheduler_variables() {
        // --ntasks=8 on a node that does not confine the job
        let slurm = [
            ("SLURM_JOB_ID", "1"),
            ("SLURM_CPUS_ON_NODE", "8"),
            ("SLURM_JOB_CPUS_PER_NODE", "6"),
        ];
        let d = decide(&vars(&slurm), &on(64, 64));
        assert_eq!(
            (d.threads, d.variable, d.job_processors),
            (8, Some("SLURM_CPUS_ON_NODE"), 8)
        );
        let d = decide(&vars(&[slurm[0], slurm[2]]), &on(64, 64));
        assert_eq!((d.threads, d.variable), (6, Some("SLURM_JOB_CPUS_PER_NODE")));
        // the processors of each node, when they are the same on all of them
        assert_eq!(
            decide(&vars(&[slurm[0], ("SLURM_JOB_CPUS_PER_NODE", "6(x2)")]), &on(64, 64)).threads,
            6
        );
        assert_eq!(
            decide(&vars(&[slurm[0], ("SLURM_JOB_CPUS_PER_NODE", "6(x2),4")]), &on(64, 64)).threads,
            64
        );
        // PBS Pro, Torque, LSF
        let d = decide(&vars(&[("PBS_JOBID", "7.pbs"), ("NCPUS", "12")]), &on(64, 64));
        assert_eq!((d.threads, d.variable), (12, Some("NCPUS")));
        assert_eq!(
            decide(&vars(&[("PBS_JOBID", "7"), ("PBS_NUM_PPN", "10")]), &on(64, 64)).threads,
            10
        );
        let d = decide(&vars(&[("LSB_JOBID", "9"), ("LSB_DJOB_NUMPROC", "4")]), &on(64, 64));
        assert_eq!(
            (d.threads, d.variable, d.job_processors),
            (4, Some("LSB_DJOB_NUMPROC"), 4)
        );
        // another scheduler's, or outside a job: ignored
        assert_eq!(
            decide(&vars(&[("LSB_JOBID", "9"), ("NCPUS", "4")]), &on(64, 64)).threads,
            64
        );
        assert_eq!(decide(&vars(&[("NCPUS", "4")]), &on(64, 64)).threads, 63);
        // a cgroup that confines the job already: the processors available
        let d = decide(&vars(&slurm), &on(4, 64));
        assert_eq!((d.threads, d.variable, d.job_processors), (4, None, 4));
        // the first set decides, even when the cgroup is the smaller
        let d = decide(
            &vars(&[
                ("SLURM_JOB_ID", "1"),
                ("SLURM_CPUS_PER_TASK", "16"),
                ("SLURM_CPUS_ON_NODE", "2"),
            ]),
            &on(8, 64),
        );
        assert_eq!((d.threads, d.variable), (8, None));
        // OMP_NUM_THREADS gives the count, and the job's processors are still known
        let d = decide(&vars(&[slurm[0], slurm[1], ("OMP_NUM_THREADS", "1")]), &on(64, 64));
        assert_eq!(
            (d.threads, d.variable, d.job_processors),
            (1, Some("OMP_NUM_THREADS"), 8)
        );
        assert_eq!(uniform_count(" 8 "), Some(8));
        assert_eq!(uniform_count("x"), None);
    }

    #[test]
    fn a_shared_machine_is_divided_among_its_users() {
        let host = |users: usize| Host {
            users: Some((users, "utmp")),
            ..on(48, 48)
        };
        let d = decide(&vars(&[]), &host(3));
        assert_eq!((d.threads, d.all, d.shared, d.reserved), (16, 16, true, false));
        assert_eq!(decide(&vars(&[]), &host(100)).threads, 1);
        // alone (or the same user twice): a desktop
        let d = decide(&vars(&[]), &host(1));
        assert_eq!((d.threads, d.all, d.shared, d.reserved), (47, 48, false, true));
        // a job, a variable, or a process confined to some of the processors: as before
        assert_eq!(decide(&vars(&[("SLURM_JOB_ID", "1")]), &host(3)).threads, 48);
        assert_eq!(decide(&vars(&[("ZEIT_NUM_THREADS", "40")]), &host(3)).threads, 40);
        let confined = Host {
            users: Some((3, "WTS")),
            ..on(8, 48)
        };
        let d = decide(&vars(&[]), &confined);
        assert_eq!((d.threads, d.shared), (8, false));
    }

    #[test]
    fn a_login_node_uses_few_threads() {
        let login = Host {
            scheduler: Some("Slurm"),
            ..on(64, 64)
        };
        let d = decide(&vars(&[]), &login);
        assert_eq!(
            (d.threads, d.all, d.login_node, d.reserved),
            (4, 4, Some("Slurm"), false)
        );
        let small = Host {
            users: Some((40, "utmp")),
            ..login.clone()
        };
        assert_eq!(decide(&vars(&[]), &small).threads, 1);
        // in a job, the job's; a variable, as it is
        let d = decide(&vars(&[("SLURM_JOB_ID", "1")]), &login);
        assert_eq!((d.threads, d.login_node), (64, None));
        assert_eq!(decide(&vars(&[("OMP_NUM_THREADS", "12")]), &login).threads, 12);
    }

    #[test]
    fn the_setting_comes_after_zeit_num_threads_outside_a_job() {
        let host = Host {
            configured: Some(12),
            users: Some((3, "WTS")),
            ..on(48, 48)
        };
        let d = decide(&vars(&[("OMP_NUM_THREADS", "1")]), &host);
        assert_eq!((d.threads, d.variable, d.shared), (12, Some(CONFIG), false));
        assert_eq!(decide(&vars(&[("ZEIT_NUM_THREADS", "2")]), &host).threads, 2);
        // a job's processors are the job's
        let d = decide(&vars(&[("SLURM_JOB_ID", "1")]), &host);
        assert_eq!((d.threads, d.variable), (48, None));
    }

    #[test]
    fn settings_are_read_from_config_toml() {
        let text = "# zeit's settings\nother = 1\nthreads = 12  # mine\n";
        assert_eq!(setting(text, "threads").as_deref(), Some("12"));
        assert_eq!(setting("threads = \"auto\"", "threads").as_deref(), Some("auto"));
        assert_eq!(setting("# threads = 3", "threads"), None);
        assert_eq!(setting("", "threads"), None);
    }

    #[test]
    fn utmp_records_give_the_distinct_users() {
        let record = |kind: i16, user: &str| {
            let mut r = vec![0u8; UTMP_SIZE];
            r[..2].copy_from_slice(&kind.to_ne_bytes());
            r[UTMP_USER.start..UTMP_USER.start + user.len()].copy_from_slice(user.as_bytes());
            r
        };
        let bytes: Vec<u8> = [
            record(2, "reboot"),
            record(USER_PROCESS, "ana"),
            record(USER_PROCESS, "bia"),
            record(USER_PROCESS, "ana"),
            record(8, "carla"),
        ]
        .concat();
        assert_eq!(utmp_users(&bytes), Some(2));
        assert_eq!(utmp_users(&record(2, "reboot")), None);
        assert_eq!(utmp_users(&bytes[..100]), None);
        assert_eq!(utmp_users(&[]), None);
    }

    #[test]
    fn logind_sessions_give_the_distinct_users() {
        let dir = std::env::temp_dir().join(format!("zeit-logind-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let sessions = [
            ("1", "UID=1000\nUSER=ana\nACTIVE=1\nSTATE=active\nCLASS=user\n"),
            ("2", "UID=1000\nUSER=ana\nSTATE=online\nCLASS=user\n"),
            ("3", "UID=1001\nUSER=bia\nSTATE=active\nCLASS=user\n"),
            ("4", "UID=1002\nUSER=carla\nSTATE=closing\nCLASS=user\n"),
            ("c1", "UID=120\nUSER=gdm\nSTATE=online\nCLASS=greeter\n"),
        ];
        for (name, text) in sessions {
            std::fs::write(dir.join(name), text).unwrap();
        }
        // each session has a FIFO beside it (1.ref), which opening would wait on
        if cfg!(unix) {
            let made = std::process::Command::new("mkfifo").arg(dir.join("1.ref")).status();
            assert!(made.is_err() || made.unwrap().success());
        }
        assert_eq!(logind_users(&dir), Some(2));
        std::fs::remove_dir_all(&dir).unwrap();
        assert_eq!(logind_users(&dir), None);
    }

    #[test]
    fn a_scheduler_installed_here() {
        let none = |_: &str| None;
        let files = |p: &Path| p == Path::new("/etc/pbs.conf");
        let found = installed_scheduler(&none, &files);
        let conf = installed_scheduler(&|v: &str| (v == "SLURM_CONF").then(|| "/x/slurm.conf".into()), &|_| {
            false
        });
        if cfg!(unix) {
            assert_eq!((found, conf), (Some("PBS"), Some("Slurm")));
        } else {
            assert_eq!((found, conf), (None, None));
        }
        assert_eq!(installed_scheduler(&none, &|_| false), None);
    }

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
