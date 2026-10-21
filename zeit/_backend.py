"""The compiled core each algorithm runs on: zeit._core (C++) or zeit._rust (Rust).

zeit's core moves from C++ to Rust one module at a time (RUST_MIGRATION.md).
While both exist, the wrappers reach a module of the core through ``core``
(``core.mannkendall.fit_mann_kendall_batch``), which gives:

- with ``ZEIT_BACKEND=cpp``, the C++ module;
- with ``ZEIT_BACKEND=rust``, the Rust module wherever the Rust core has one;
- otherwise, the Rust module for the modules in ``RUST_BY_DEFAULT``, the C++ one
  for the rest.

Both cores take the same arguments and give the same results, so the choice is
made once, at import.
"""
import os

from . import _core as _cpp

try:
    from . import _rust
except ImportError:  # built without a Rust toolchain: the C++ core alone
    _rust = None

#: The modules of the Rust core that replace the C++ ones by default, once they
#: pass the gates of RUST_MIGRATION.md (differential tests against the C++, the
#: parity suites against the authors' code, timings on the three systems).
RUST_BY_DEFAULT = frozenset()


def _mode():
    mode = os.environ.get("ZEIT_BACKEND", "").strip().lower()
    if mode not in ("", "cpp", "rust"):
        raise ValueError(f"ZEIT_BACKEND must be 'cpp' or 'rust', not {mode!r}")
    if mode == "rust" and _rust is None:
        raise ImportError("ZEIT_BACKEND=rust, but this zeit was built without its Rust core (zeit._rust)")
    return mode


MODE = _mode()


def backend_of(name):
    """``'rust'`` or ``'cpp'``: the core that runs the module ``name``."""
    if _rust is not None and isinstance(getattr(_rust, name, None), type(_cpp)):
        if MODE == "rust" or (MODE == "" and name in RUST_BY_DEFAULT):
            return "rust"
    return "cpp"


class _Core:
    """zeit._core, with modules of the Rust core in place of the C++ ones where chosen."""

    def __getattr__(self, name):
        module = getattr(_rust if backend_of(name) == "rust" else _cpp, name)
        setattr(self, name, module)  # resolved once
        return module

    def __dir__(self):
        return dir(_cpp)


core = _Core()
