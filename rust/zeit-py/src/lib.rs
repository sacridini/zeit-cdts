//! zeit._rust: the Rust core, with the same submodules, functions and
//! arguments as zeit._core (the C++ one) for the modules already converted, so
//! zeit/_backend.py can hand either to the wrappers.

use pyo3::prelude::*;
use pyo3::types::PyModule;

mod arrays;
mod mannkendall;
mod snic;
mod twdtw;

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
    Ok(())
}
