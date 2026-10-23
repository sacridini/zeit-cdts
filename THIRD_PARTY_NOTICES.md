# Third-party notices

Zeit is licensed under the GNU General Public License, version 2 or (at your
option) any later version (`GPL-2.0-or-later`, see `LICENSE`).

Several algorithms in Zeit are ports of, or are derived from, the software
listed below. Their licenses are all compatible with distributing Zeit under
the GPL. Because the CCDC lasso solver is derived from GPL-2.0-only code, the
compiled `zeit._core` extension (the Rust core, `rust/`) as a whole is
distributed under the terms of **GPL version 2**.

## Ported or derived code

| Zeit component | Derived from | License |
| :--- | :--- | :--- |
| `rust/zeit-core/src/ccdc.rs` (CCDC), `rust/zeit-core/src/robustfit.rs` | [GERSL/CCDC](https://github.com/GERSL/CCDC), Zhu & Woodcock (2014), MATLAB | MIT |
| `rust/zeit-core/src/ccdc.rs`, the lasso solver | GLMnet Fortran (Friedman, Hastie & Tibshirani), as bundled with GERSL/CCDC | GPL-2.0-only |
| `rust/zeit-core/src/bfast.rs`, `bfast_monitor.rs`, `bfast_lite.rs` | R package [bfast](https://github.com/bfast2/bfast), Verbesselt et al. | GPL-2.0-or-later |
| `rust/zeit-core/src/bfast*.rs` (efp/MOSUM process, critical values, breakpoints) | R package [strucchangeRcpp](https://github.com/bfast2/strucchangeRcpp), Zeileis et al. | GPL-2.0 or GPL-3.0 |
| `rust/zeit-core/src/stl.rs` | R `stats::stl()` (`src/library/stats/src/stl.c`), R Core Team | GPL-2.0 or GPL-3.0 |
| `rust/zeit-core/src/optimize.rs` (Brent's minimizer, `fmin` and `fmin_netlib`) | R `stats::optimize()` (`src/library/stats/src/optimize.c`, `Brent_fmin`, itself a translation of netlib's `fmin.f`, Forsythe, Malcolm & Moler 1977), R Core Team | GPL-2.0 or GPL-3.0 |
| `rust/zeit-core/src/mann_kendall.rs` | [pymannkendall](https://github.com/mmhs013/pymannkendall), Hussain & Mahmud (2019) | MIT |
| `rust/zeit-core/src/phenology.rs`, `phenology/` (methodology) | R package [phenofit](https://github.com/eco-hydro/phenofit), Kong et al. (2022) | GPL-2.0 |
| `rust/zeit-core/src/eigen.rs`, `eigen/` (QR, LDLT, matrix products, sparse Cholesky with AMD ordering, Levenberg-Marquardt), ported operation by operation so the results are those of zeit 1.x | [Eigen](https://eigen.tuxfamily.org) 3.4 | MPL-2.0 |
| `rust/zeit-core/src/eigen/lm.rs` (Levenberg-Marquardt, from Eigen's `unsupported/NonLinearOptimization`) | MINPACK, University of Chicago | MINPACK license (see below) |
| `zeit/ai/utae.py` (U-TAE, L-TAE) | [VSainteuf/utae-paps](https://github.com/VSainteuf/utae-paps), Garnot & Landrieu (2021) | MIT |
| `zeit/ai/presto.py` (Presto encoder; its pretrained weights are downloaded, not bundled) | [nasaharvest/presto](https://github.com/nasaharvest/presto), Tseng et al. (2023) | MIT |
| `zeit/ai/prithvi.py` (Prithvi-EO-2.0 encoder; its pretrained weights are downloaded, not bundled) | `prithvi_mae.py` of [ibm-nasa-geospatial/Prithvi-EO-2.0](https://huggingface.co/ibm-nasa-geospatial), Szwarcman et al. (2024), IBM Corp. | Apache-2.0 |
| `zeit/ai/tempcnn.py` (architecture) | R package [sits](https://github.com/e-sensing/sits), `sits_tempcnn()` | GPL-2.0 |
| `zeit/segmentation.py` (seed grids) | R package [snic](https://github.com/rolfsimoes/snic), `snic_grid()` | GPL-2.0 |
| `rust/zeit-core/src/warp.rs`, `warp/` (resampling kernels of `load_raster(..., like=)`) | [GDAL](https://gdal.org) 3.12, `alg/gdalwarpkernel.cpp`, Warmerdam, Rouault et al. (via landschaft, same author as Zeit) | MIT |
| `rust/zeit-core/src/landtrendr.rs` (LandTrendr) | [KennedyResearch/LandTrendr-2012](https://github.com/KennedyResearch/LandTrendr-2012), Kennedy et al. (2010), IDL | no license published — see below |

SNIC (`rust/zeit-core/src/snic.rs`) is implemented from the paper (Achanta & Süsstrunk,
CVPR 2017). The authors' reference code is not included; the test suite only
stores outputs of it (`tests/data/snic_reference_parity.npz`) to check that
Zeit gives the same labels.

The remotePARTS methods (`rust/zeit-core/src/ar_reml.rs`, `gls.rs`,
`spatial.rs`, `parts.rs`) are implemented from the paper (Ives et al., Remote
Sensing of Environment 266, 112678, 2021) and the statistics it defines; the
AR(1) fit, the nugget of a fit alone and the statistics between partitions
are computed by other algorithms than the R package's. The p-value of the
correlated chi-square test integrates Imhof's (1961) formula. The package's code
(GPL-3.0-or-later, which could not be part of the GPL-2.0 `zeit._core`) is not
included; the test suite only stores outputs of it
(`rust/zeit-core/tests/data/remoteparts_*.json`, made by
`tests/data/make_remoteparts_fixtures.R`) to check that Zeit gives the same fits.

**LandTrendr-2012**: the original repository does not state a license. Zeit's
port is a derivative of that code; permission from the authors is being
sought. Until then, the LandTrendr port's redistribution terms are unresolved.

## Bundled or linked libraries

The Rust crates compiled into `zeit._core` (from `rust/Cargo.lock`; crates
used only to build it are left out):

| Library | Use | License |
| :--- | :--- | :--- |
| [PyO3](https://github.com/PyO3/pyo3) (`pyo3`, `pyo3-ffi`) | Python bindings | MIT OR Apache-2.0 |
| [rust-numpy](https://github.com/PyO3/rust-numpy) (`numpy`) | NumPy arrays in the bindings | BSD-2-Clause (see below) |
| [ndarray](https://github.com/rust-ndarray/ndarray), `matrixmultiply`, `rawpointer`, `num-traits`, `num-complex`, `num-integer` | array views (through rust-numpy) | MIT OR Apache-2.0 |
| [rayon](https://github.com/rayon-rs/rayon) (`rayon`, `rayon-core`, `crossbeam-*`, `either`) | threads | MIT OR Apache-2.0 |
| [mimalloc](https://github.com/purpleprotocol/mimalloc_rust) (`mimalloc`, `libmimalloc-sys`, with Microsoft's [mimalloc](https://github.com/microsoft/mimalloc) C library) | memory allocator | MIT |
| [faer](https://codeberg.org/sarah-quinones/faer) (`faer`, `faer-traits`, `gemm`, `gemm-*`, `nano-gemm`, `nano-gemm-*`, `private-gemm-x86`, `pulp`, `dyn-stack`, `reborrow`, `equator`, `qd`, `spindle`) | dense linear algebra of the GLS (Cholesky, tridiagonalization) | MIT; parts derived from Eigen (MPL-2.0), LAPACK and SuiteSparse AMD/COLAMD (BSD-3-Clause, see below) |
| `raw-cpuid`, `libm` | support (through faer) | MIT |
| `atomic-wait` | support (through faer) | BSD-2-Clause (see below) |
| `libc`, `once_cell`, `portable-atomic`, `portable-atomic-util`, `rustc-hash`, `bitflags`, `bytemuck`, `cfg-if`, `generativity`, `num_cpus`, `windows-sys` | support | MIT OR Apache-2.0 (`bytemuck`: or Zlib) |

The parts of Eigen ported to Rust (above), and those faer derives from Eigen,
are covered by the Mozilla Public License 2.0 (<https://mozilla.org/MPL/2.0/>), which allows distributing them
as part of a GPL work.

This product includes software developed by the University of Chicago, as
Operator of Argonne National Laboratory (MINPACK, via Eigen's
NonLinearOptimization module, ported to Rust in
`rust/zeit-core/src/eigen/lm.rs`). The MINPACK copyright notice:

> Minpack Copyright Notice (1999) University of Chicago. All rights reserved
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> 1. Redistributions of source code must retain the above copyright notice,
>    this list of conditions and the following disclaimer.
> 2. Redistributions in binary form must reproduce the above copyright notice,
>    this list of conditions and the following disclaimer in the documentation
>    and/or other materials provided with the distribution.
> 3. The end-user documentation included with the redistribution, if any, must
>    include the following acknowledgment: "This product includes software
>    developed by the University of Chicago, as Operator of Argonne National
>    Laboratory." Alternately, this acknowledgment may appear in the software
>    itself, if and wherever such third-party acknowledgments normally appear.
> 4. WARRANTY DISCLAIMER. THE SOFTWARE IS SUPPLIED "AS IS" WITHOUT WARRANTY OF
>    ANY KIND. THE COPYRIGHT HOLDER, THE UNITED STATES, THE UNITED STATES
>    DEPARTMENT OF ENERGY, AND THEIR EMPLOYEES: (1) DISCLAIM ANY WARRANTIES,
>    EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY IMPLIED WARRANTIES OF
>    MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, TITLE OR
>    NON-INFRINGEMENT, (2) DO NOT ASSUME ANY LEGAL LIABILITY OR RESPONSIBILITY
>    FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF THE SOFTWARE, (3) DO NOT
>    REPRESENT THAT USE OF THE SOFTWARE WOULD NOT INFRINGE PRIVATELY OWNED
>    RIGHTS, (4) DO NOT WARRANT THAT THE SOFTWARE WILL FUNCTION UNINTERRUPTED,
>    THAT IT IS ERROR-FREE OR THAT ANY ERRORS WILL BE CORRECTED.
> 5. LIMITATION OF LIABILITY. IN NO EVENT WILL THE COPYRIGHT HOLDER, THE UNITED
>    STATES, THE UNITED STATES DEPARTMENT OF ENERGY, OR THEIR EMPLOYEES: BE
>    LIABLE FOR ANY INDIRECT, INCIDENTAL, CONSEQUENTIAL, SPECIAL OR PUNITIVE
>    DAMAGES OF ANY KIND OR NATURE, INCLUDING BUT NOT LIMITED TO LOSS OF
>    PROFITS OR LOSS OF DATA, FOR ANY REASON WHATSOEVER, WHETHER SUCH LIABILITY
>    IS ASSERTED ON THE BASIS OF CONTRACT, TORT (INCLUDING NEGLIGENCE OR STRICT
>    LIABILITY), OR OTHERWISE, EVEN IF ANY OF SAID PARTIES HAS BEEN WARNED OF
>    THE POSSIBILITY OF SUCH LOSS OR DAMAGES.

## BSD-2-Clause notices (rust-numpy, atomic-wait)

> Copyright (c) 2017, Toshiki Teramura
> All rights reserved.
>
> Copyright (c) 2022, Mara Bos <m-ou.se@m-ou.se>
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## BSD-3-Clause notices (LAPACK, SuiteSparse AMD and COLAMD, through faer)

> Copyright (c) 1992-2025 The University of Tennessee and The University of
> Tennessee Research Foundation. All rights reserved.
> Copyright (c) 2000-2025 The University of California Berkeley. All rights
> reserved.
> Copyright (c) 2006-2025 The University of Colorado Denver. All rights
> reserved. (LAPACK)
>
> AMD, Copyright (c), 1996-2022, Timothy A. Davis, Patrick R. Amestoy, and
> Iain S. Duff. All Rights Reserved.
>
> COLAMD, Copyright 1998-2022, Timothy A. Davis.
>
> Redistribution and use in source and binary forms, with or without
> modification, are permitted provided that the following conditions are met:
>
> * Redistributions of source code must retain the above copyright notice, this
>   list of conditions and the following disclaimer.
> * Redistributions in binary form must reproduce the above copyright notice,
>   this list of conditions and the following disclaimer in the documentation
>   and/or other materials provided with the distribution.
> * Neither the name of the copyright holders, nor of the organizations to which
>   the authors are affiliated, nor the names of its contributors may be used to
>   endorse or promote products derived from this software without specific
>   prior written permission.
>
> The copyright holders provide no reassurances that the source code provided
> does not infringe any patent, copyright, or any other intellectual property
> rights of third parties. The copyright holders disclaim any liability to any
> recipient for claims brought against recipient by any third party for
> infringement of that parties intellectual property rights. (LAPACK)
>
> THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
> AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
> IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
> DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
> FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
> DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
> SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
> CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
> OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
> OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

## MIT License notices

The following copyright notices apply to the MIT-licensed code listed above:

- Copyright (c) 2003, Frank Warmerdam; Copyright (c) 2008-2013, Even Rouault (GDAL, `gdalwarpkernel.cpp`)
- Copyright (c) 2018-2025 Microsoft Corporation, Daan Leijen (mimalloc); Copyright 2019 Octavian Oncescu (mimalloc crates)
- Copyright (c) 2018 Global Environmental Remote Sensing Lab (GERSL/CCDC)
- Copyright (c) 2021-2025 sarah quiñones el kazdadi (faer, faer-traits, gemm, nano-gemm, private-gemm-x86, pulp, dyn-stack, reborrow, equator, qd, spindle)
- Copyright (c) 2015 Gerd Zellweger (raw-cpuid)
- The Rust Project Developers (libm)
- Copyright (c) 2019 Md. Manjurul Hussain Shourov (pymannkendall)
- Copyright (c) 2021 VSainteuf (utae-paps)
- Copyright (c) 2024 Presto Authors (nasaharvest/presto)

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
