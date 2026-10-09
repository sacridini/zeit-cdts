# Third-party notices

Zeit is licensed under the GNU General Public License, version 2 or (at your
option) any later version (`GPL-2.0-or-later`, see `LICENSE`).

Several algorithms in Zeit are ports of, or are derived from, the software
listed below. Their licenses are all compatible with distributing Zeit under
the GPL. Because the CCDC lasso solver is derived from GPL-2.0-only code, the
compiled `zeit._core` extension as a whole is distributed under the terms of
**GPL version 2**.

## Ported or derived code

| Zeit component | Derived from | License |
| :--- | :--- | :--- |
| `src/ccdc.cpp` (CCDC) | [GERSL/CCDC](https://github.com/GERSL/CCDC), Zhu & Woodcock (2014), MATLAB | MIT |
| `src/ccdc.cpp`, `glmnet_lasso()` | GLMnet Fortran (Friedman, Hastie & Tibshirani), as bundled with GERSL/CCDC | GPL-2.0-only |
| `src/bfast.cpp`, `src/bfast_monitor.cpp`, `src/bfast_lite.cpp` | R package [bfast](https://github.com/bfast2/bfast), Verbesselt et al. | GPL-2.0-or-later |
| `src/bfast*.cpp` (efp/MOSUM process, critical values, breakpoints) | R package [strucchangeRcpp](https://github.com/bfast2/strucchangeRcpp), Zeileis et al. | GPL-2.0 or GPL-3.0 |
| `src/stl_decompose.cpp` | R `stats::stl()` (`src/library/stats/src/stl.c`), R Core Team | GPL-2.0 or GPL-3.0 |
| `src/mann_kendall.cpp` | [pymannkendall](https://github.com/mmhs013/pymannkendall), Hussain & Mahmud (2019) | MIT |
| `src/phenology*.cpp` (methodology) | R package [phenofit](https://github.com/eco-hydro/phenofit), Kong et al. (2022) | GPL-2.0 |
| `zeit/ai/utae.py` (U-TAE, L-TAE) | [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps), Garnot & Landrieu (2021) | MIT |
| `zeit/ai/tempcnn.py` (architecture) | R package [sits](https://github.com/e-sensing/sits), `sits_tempcnn()` | GPL-2.0 |
| `zeit/segmentation.py` (seed grids) | R package [snic](https://github.com/rolfsimoes/snic), `snic_grid()` | GPL-2.0 |
| `src/warp.cpp` (resampling kernels of `load_raster(..., like=)`) | [GDAL](https://gdal.org) 3.12, `alg/gdalwarpkernel.cpp`, Warmerdam, Rouault et al. (via landschaft, same author as Zeit) | MIT |
| `src/landtrendr.cpp` (LandTrendr) | [KennedyResearch/LandTrendr-2012](https://github.com/KennedyResearch/LandTrendr-2012), Kennedy et al. (2010), IDL | no license published — see below |

SNIC (`src/snic.cpp`) is implemented from the paper (Achanta & Süsstrunk,
CVPR 2017). The authors' reference code is not included; the test suite only
stores outputs of it (`tests/data/snic_reference_parity.npz`) to check that
Zeit gives the same labels.

**LandTrendr-2012**: the original repository does not state a license. Zeit's
port is a derivative of that code; permission from the authors is being
sought. Until then, the LandTrendr port's redistribution terms are unresolved.

## Bundled or linked libraries

| Library | Use | License |
| :--- | :--- | :--- |
| [Eigen](https://eigen.tuxfamily.org) (`third_party/eigen`) | linear algebra, compiled in | MPL-2.0 (see `third_party/eigen/COPYING.*`) |
| Eigen `unsupported/NonLinearOptimization` | phenology curve fitting, compiled in | MINPACK license (see below) |
| [pybind11](https://github.com/pybind/pybind11) | Python bindings, compiled in | BSD-3-Clause |

This product includes software developed by the University of Chicago, as
Operator of Argonne National Laboratory (MINPACK, via Eigen's
NonLinearOptimization module; full notice in
`third_party/eigen/COPYING.MINPACK`).

## MIT License notices

The following copyright notices apply to the MIT-licensed code listed above:

- Copyright (c) 2003, Frank Warmerdam; Copyright (c) 2008-2013, Even Rouault (GDAL, `gdalwarpkernel.cpp`)
- Copyright (c) 2018 Global Environmental Remote Sensing Lab (GERSL/CCDC)
- Copyright (c) 2019 Md. Manjurul Hussain Shourov (pymannkendall)
- Copyright (c) 2021 VSainteuf (utae-paps)

> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

The full texts of the GPL versions referred to above are available at
<https://www.gnu.org/licenses/>.
