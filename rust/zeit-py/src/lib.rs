//! zeit._core: zeit's compiled core, the Python bindings of zeit-core, one
//! submodule per algorithm (zeit._core.ccdc, zeit._core.landtrendr, ...).

use pyo3::prelude::*;
use pyo3::types::PyModule;

// mimalloc for the Rust core's allocations: the modules that allocate per pixel
// (QR decompositions, robust fits) ran up to 3x slower on the system allocator,
// Windows' above all, and more so on many threads. It changes no value.
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod aoa;
mod arrays;
mod bfast;
mod bfastlite;
mod bfastmonitor;
mod ccdc;
mod coded;
mod ewmacd;
mod landtrendr;
mod mannkendall;
mod parts;
mod phenology;
mod postproc;
mod recovery;
mod sma;
mod smooth;
mod snic;
mod som;
mod tmask;
mod twdtw;
mod utils;
mod warp;

/// Adds `sub` to `parent` as `parent.name`, importable as `zeit._core.name` too
/// (`from zeit._core.mannkendall import ...`).
fn add_submodule(parent: &Bound<'_, PyModule>, sub: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent.py();
    parent.add_submodule(sub)?;
    let name = format!("zeit._core.{}", sub.name()?);
    py.import("sys")?.getattr("modules")?.set_item(name, sub)?;
    Ok(())
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__doc__", "The compiled core of zeit, in Rust")?;
    add_submodule(m, &mannkendall::module(m.py())?)?;
    add_submodule(m, &parts::module(m.py())?)?;
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
    add_submodule(m, &ccdc::module(m.py())?)?;
    add_submodule(m, &warp::module(m.py())?)?;
    add_submodule(m, &phenology::module(m.py())?)?;
    add_submodule(m, &recovery::module(m.py())?)?;
    add_submodule(m, &ewmacd::module(m.py())?)?;
    add_submodule(m, &postproc::module(m.py())?)?;
    add_submodule(m, &aoa::module(m.py())?)?;
    Ok(())
}
