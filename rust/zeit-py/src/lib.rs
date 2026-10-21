//! zeit._rust: the Rust core, with the same submodules, functions and
//! arguments as zeit._core (the C++ one) for the modules already converted, so
//! zeit/_backend.py can hand either to the wrappers.

use pyo3::prelude::*;
use pyo3::types::PyModule;

// mimalloc for the Rust core's allocations: the modules that allocate per pixel
// (QR decompositions, robust fits) ran up to 3x slower on the system allocator,
// Windows' above all, and more so on many threads. It changes no value.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod arrays;
mod bfast;
mod bfastlite;
mod bfastmonitor;
mod coded;
mod landtrendr;
mod mannkendall;
mod sma;
mod smooth;
mod snic;
mod som;
mod tmask;
mod twdtw;
mod utils;

/// Adds `sub` to `parent` as `parent.name`, importable as `zeit._rust.name` too
/// (`from zeit._rust.mannkendall import ...`, as the wrappers do with _core).
fn add_submodule(parent: &Bound<'_, PyModule>, sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    parent.add_submodule(sub)?;
    let name = format!("zeit._rust.{}", sub.name()?);
    py.import("sys")?.getattr("modules")?.set_item(name, sub)?;
    Ok(())
}

#[pymodule]
fn _rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "Rust backend for zeit")?;
    add_submodule(m, &mannkendall::module(m.py())?)?;
    add_submodule(m, &twdtw::module(m.py())?)?;
    add_submodule(m, &snic::module(m.py())?)?;
    add_submodule(m, &smooth::module(m.py())?)?;
    add_submodule(m, &sma::module(m.py())?)?;
    add_submodule(m, &coded::module(m.py())?)?;
    add_submodule(m, &tmask::module(m.py())?)?;
    add_submodule(m, &bfastmonitor::module(m.py())?)?;
    add_submodule(m, &bfastlite::module(m.py())?)?;
    add_submodule(m, &bfast::module(m.py())?)?;
    add_submodule(m, &som::module(m.py())?)?;
    add_submodule(m, &utils::module(m.py())?)?;
    add_submodule(m, &landtrendr::module(m.py())?)?;
    Ok(())
}
