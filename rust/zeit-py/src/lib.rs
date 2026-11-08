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
mod bap;
mod beast;
mod bfast;
mod bfastlite;
mod bfastmonitor;
mod brdf;
mod ccdc;
mod coded;
mod coreg;
mod ewmacd;
mod aod;
mod atmo;
mod fmask;
mod fmask_run;
mod fmask_shadow;
mod s2l1c;
mod gas;
mod hotspots;
mod improphe;
mod knndm;
mod landtrendr;
mod mannkendall;
mod parallax;
mod parallel;
mod parts;
mod phenology;
mod postproc;
mod profile;
mod radtran;
mod recovery;
mod resmerge;
mod sccd;
mod sma;
mod smooth;
mod snic;
mod som;
mod specadjust;
mod tasseled;
mod texture;
mod tmask;
mod topo;
mod topocorr;
mod tsa;
mod twdtw;
mod utils;
mod variogram;
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
    add_submodule(m, &texture::module(m.py())?)?;
    add_submodule(m, &bap::module(m.py())?)?;
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
    add_submodule(m, &hotspots::module(m.py())?)?;
    add_submodule(m, &sccd::module(m.py())?)?;
    add_submodule(m, &beast::module(m.py())?)?;
    add_submodule(m, &parallel::module(m.py())?)?;
    add_submodule(m, &variogram::module(m.py())?)?;
    add_submodule(m, &knndm::module(m.py())?)?;
    add_submodule(m, &profile::module(m.py())?)?;
    add_submodule(m, &tsa::module(m.py())?)?;
    add_submodule(m, &specadjust::module(m.py())?)?;
    add_submodule(m, &brdf::module(m.py())?)?;
    add_submodule(m, &tasseled::module(m.py())?)?;
    add_submodule(m, &improphe::module(m.py())?)?;
    add_submodule(m, &parallax::module(m.py())?)?;
    add_submodule(m, &resmerge::module(m.py())?)?;
    add_submodule(m, &radtran::module(m.py())?)?;
    add_submodule(m, &aod::module(m.py())?)?;
    add_submodule(m, &atmo::module(m.py())?)?;
    add_submodule(m, &fmask::module(m.py())?)?;
    add_submodule(m, &fmask_run::module(m.py())?)?;
    add_submodule(m, &fmask_shadow::module(m.py())?)?;
    add_submodule(m, &s2l1c::module(m.py())?)?;
    add_submodule(m, &gas::module(m.py())?)?;
    add_submodule(m, &topo::module(m.py())?)?;
    add_submodule(m, &topocorr::module(m.py())?)?;
    add_submodule(m, &coreg::module(m.py())?)?;
    Ok(())
}
