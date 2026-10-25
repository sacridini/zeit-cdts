//! zeit._core.parallel: the threads zeit uses, as zeit-core's parallel module
//! decides them.

use pyo3::prelude::*;
use pyo3::types::PyDict;
use zeit_core::parallel;

/// The threads of `n_jobs` <= 0 and where they come from: a dict with
/// `threads`, `all` (with the processor kept free), `variable` (the variable
/// that gave the count, or None), `available` (the processors of the
/// process's affinity and cgroup quota), `machine`, `scheduler` and `job` (None
/// outside a job) and `reserved`.
#[pyfunction]
fn detect(py: Python<'_>) -> PyResult<Bound<'_, PyDict>> {
    let d = parallel::detect();
    let out = PyDict::new(py);
    out.set_item("threads", d.threads)?;
    out.set_item("all", d.all)?;
    out.set_item("variable", d.variable)?;
    out.set_item("available", d.available)?;
    out.set_item("machine", d.machine)?;
    out.set_item("scheduler", d.job.as_ref().map(|j| j.0))?;
    out.set_item("job", d.job.as_ref().map(|j| j.1.clone()))?;
    out.set_item("reserved", d.reserved)?;
    Ok(out)
}

/// The number of threads `n_jobs` asks for, as every function of the core
/// counts them.
#[pyfunction]
fn threads(n_jobs: i32) -> usize {
    parallel::threads(n_jobs)
}

pub fn module(py: Python<'_>) -> PyResult<Bound<'_, PyModule>> {
    let m = PyModule::new(py, "parallel")?;
    m.add("__doc__", "The threads zeit uses")?;
    m.add_function(wrap_pyfunction!(detect, &m)?)?;
    m.add_function(wrap_pyfunction!(threads, &m)?)?;
    Ok(m)
}
