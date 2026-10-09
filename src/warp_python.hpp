// SPDX-License-Identifier: GPL-2.0-or-later
#pragma once

#include <pybind11/pybind11.h>

namespace zeit::warp {

// Adds warp, coords and lattice_checks to the module (zeit._core.warp).
void register_python(pybind11::module_& m);

}  // namespace zeit::warp
