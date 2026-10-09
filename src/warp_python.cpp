// SPDX-License-Identifier: GPL-2.0-or-later
//
// Python bindings of the warp kernel (zeit._core.warp), from landschaft 1.35.0
// (src/cpp/io/python.cpp: io_warp, io_warp_coords, io_lattice_checks), with
// per_band and the thread count of each call.
#include "warp_python.hpp"

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <array>
#include <climits>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "warp.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;

namespace {

using zeit::warp::i64;
using zeit::warp::WarpCoords;
using zeit::warp::WarpSpec;

// Whether two types are the same by kind and size, in the machine's byte order: NumPy has
// two dtypes for one C type (int and long are both int32 on Windows, long and long long
// int64 elsewhere), so an identity check rejects e.g. np.result_type(np.uint16, np.int16).
bool same_type(const py::dtype& a, const py::dtype& b) {
    return a.kind() == b.kind() && a.itemsize() == b.itemsize() && a.attr("isnative").cast<bool>() &&
           b.attr("isnative").cast<bool>();
}

template <class T>
bool is_type(const py::dtype& dt) {
    return same_type(dt, py::dtype::of<T>());
}

// Calls f with the typed data of a C-contiguous array of one of the raster types.
template <class F>
auto with_typed(const py::array& a, F&& f) {
    if (!(a.flags() & py::array::c_style)) throw py::value_error("expected a C-contiguous array");
    const py::dtype dt = a.dtype();
    if (is_type<std::uint8_t>(dt)) return f(static_cast<const std::uint8_t*>(a.data()));
    if (is_type<std::int8_t>(dt)) return f(static_cast<const std::int8_t*>(a.data()));
    if (is_type<std::uint16_t>(dt)) return f(static_cast<const std::uint16_t*>(a.data()));
    if (is_type<std::int16_t>(dt)) return f(static_cast<const std::int16_t*>(a.data()));
    if (is_type<std::uint32_t>(dt)) return f(static_cast<const std::uint32_t*>(a.data()));
    if (is_type<std::int32_t>(dt)) return f(static_cast<const std::int32_t*>(a.data()));
    if (is_type<std::int64_t>(dt)) return f(static_cast<const std::int64_t*>(a.data()));
    if (is_type<float>(dt)) return f(static_cast<const float*>(a.data()));
    if (is_type<double>(dt)) return f(static_cast<const double*>(a.data()));
    throw py::type_error("unsupported raster type: use (u)int8/16/32, int64, float32 or float64");
}

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using Int8Array = py::array_t<std::int8_t, py::array::c_style | py::array::forcecast>;
using I64Array = py::array_t<i64, py::array::c_style | py::array::forcecast>;
using PointerArray = py::array_t<std::uintptr_t, py::array::c_style | py::array::forcecast>;

// The OpenMP thread count of the calling thread for one call (n < 1: unchanged), set back
// after it, so that the other engines' defaults (omp_get_max_threads) stay as they were.
class Threads {
public:
    explicit Threads(int n) {
#ifdef _OPENMP
        previous_ = omp_get_max_threads();
        if (n > 0) omp_set_num_threads(n);
#else
        (void)n;
#endif
    }
    ~Threads() {
#ifdef _OPENMP
        omp_set_num_threads(previous_);
#endif
    }
    Threads(const Threads&) = delete;
    Threads& operator=(const Threads&) = delete;

private:
    int previous_ = 1;
};

int max_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

// The source coordinates of the warp kernel's samples (checked): samples rows
// r0 .. last_row - 1, columns c0 .. last_col - 1 must be covered.
WarpCoords make_coords(int kind, double offset, const std::array<double, 6>& gt, const std::array<double, 6>& inv,
                       const Int8Array& level, const DoubleArray& grid0, const I64Array& node_off,
                       const DoubleArray& nodes, const std::array<i64, 3>& cells, const DoubleArray& dense,
                       i64 dense_row0, i64 dense_col0, std::uintptr_t transformer,
                       const PointerArray& transformer_args, i64 r0, i64 c0, i64 last_row, i64 last_col) {
    using zeit::warp::kWarpCell;
    WarpCoords c;
    c.kind = kind;
    c.offset = offset;
    std::copy(gt.begin(), gt.end(), c.gt);
    std::copy(inv.begin(), inv.end(), c.inv);
    if (kind == WarpCoords::LATTICE) {
        const i64 cells_x = cells[0], cell_x0 = cells[1], cell_y0 = cells[2];
        if (cells_x < 1 || cell_x0 < 0 || cell_y0 < 0 || level.size() % cells_x != 0) {
            throw py::value_error("bad lattice");
        }
        const i64 cells_y = static_cast<i64>(level.size()) / cells_x;
        if (cell_x0 * kWarpCell > c0 || cell_y0 * kWarpCell > r0 || (cell_x0 + cells_x) * kWarpCell < last_col ||
            (cell_y0 + cells_y) * kWarpCell < last_row) {
            throw py::value_error("the lattice does not cover the destination region");
        }
        if (node_off.size() != level.size()) throw py::value_error("one node offset per lattice cell");
        if (grid0.size() != (cells_x + 1) * (cells_y + 1) * 2) {
            throw py::value_error("grid0 must hold the corners of every lattice cell");
        }
        const i64 n_nodes = static_cast<i64>(nodes.size()) / 2;
        for (i64 k = 0; k < static_cast<i64>(level.size()); ++k) {
            const int lev = level.data()[k];
            if (lev < 0 || lev > 4) throw py::value_error("lattice levels go from 0 to 4");
            if (lev == 0) continue;
            const i64 side = kWarpCell / (kWarpCell >> lev) + 1;
            if (node_off.data()[k] < 0 || node_off.data()[k] + side * side > n_nodes) {
                throw py::value_error("lattice nodes out of range");
            }
        }
        c.cells_x = cells_x;
        c.cells_y = cells_y;
        c.cell_x0 = cell_x0;
        c.cell_y0 = cell_y0;
        c.level = level.data();
        c.grid0 = grid0.data();
        c.node_off = node_off.data();
        c.nodes = nodes.data();
    } else if (kind == WarpCoords::DENSE) {
        if (dense.ndim() != 3 || dense.shape(2) != 2) {
            throw py::value_error("dense coordinates must be (rows, columns, 2)");
        }
        if (dense_row0 > r0 || dense_col0 > c0 || dense_row0 + dense.shape(0) < last_row ||
            dense_col0 + dense.shape(1) < last_col) {
            throw py::value_error("the dense coordinates do not cover the destination region");
        }
        c.dense = dense.data();
        c.dense_row0 = dense_row0;
        c.dense_col0 = dense_col0;
        c.dense_stride = dense.shape(1);
    } else if (kind != WarpCoords::AFFINE && kind != WarpCoords::EXACT) {
        throw py::value_error("unknown coordinate kind");
    }
    if (transformer != 0) {
        if (transformer_args.size() < 1) throw py::value_error("one transformer argument per thread");
        c.transformer = reinterpret_cast<zeit::warp::TransformerFunc>(transformer);
        c.transformer_args = reinterpret_cast<void* const*>(transformer_args.data());
        c.n_transformer_args = static_cast<int>(transformer_args.size());
    } else if (kind == WarpCoords::EXACT) {
        throw py::value_error("exact coordinates need the transformer");
    }
    return c;
}

}  // namespace

namespace zeit::warp {

void register_python(py::module_& m) {
    m.def(
        "warp",
        [](const py::array& src, int method, const std::array<int, 2>& src_shape, const std::array<int, 2>& window,
           const std::array<double, 2>& scales, bool has_src_nodata, double src_nodata, double dst_nodata,
           const std::array<int, 4>& region, int kind, double offset, const std::array<double, 6>& gt,
           const std::array<double, 6>& inv, const Int8Array& level, const DoubleArray& grid0,
           const I64Array& node_off, const DoubleArray& nodes, const std::array<i64, 3>& cells,
           const DoubleArray& dense, i64 dense_row0, i64 dense_col0, std::uintptr_t transformer,
           const PointerArray& transformer_args, bool per_band, int n_threads) {
            if (src.ndim() != 3) throw py::value_error("src must be (bands, rows, columns)");
            if (!(src.flags() & py::array::c_style)) throw py::value_error("src must be C-contiguous");
            WarpSpec sp;
            sp.method = method;
            sp.bands = static_cast<int>(src.shape(0));
            sp.src_h = src_shape[0];
            sp.src_w = src_shape[1];
            sp.win_r0 = window[0];
            sp.win_c0 = window[1];
            sp.win_h = static_cast<int>(src.shape(1));
            sp.win_w = static_cast<int>(src.shape(2));
            sp.xscale = scales[0];
            sp.yscale = scales[1];
            sp.has_src_nodata = has_src_nodata;
            sp.src_nodata = src_nodata;
            sp.dst_nodata = dst_nodata;
            sp.per_band = per_band;
            sp.dst_r0 = region[0];
            sp.dst_c0 = region[1];
            sp.dst_h = region[2];
            sp.dst_w = region[3];
            const bool valid_method = (method >= WARP_NEAREST && method <= WARP_MODE) ||
                                      (method >= WARP_MAX && method <= WARP_Q3) || method == WARP_RMS;
            if (!valid_method) throw py::value_error("unknown warp method");
            if (sp.bands < 1 || sp.dst_h < 0 || sp.dst_w < 0 || sp.dst_r0 < 0 || sp.dst_c0 < 0) {
                throw py::value_error("bad destination region");
            }
            if (sp.win_r0 < 0 || sp.win_c0 < 0 || sp.win_r0 + sp.win_h > sp.src_h || sp.win_c0 + sp.win_w > sp.src_w) {
                throw py::value_error("the source window must lie in the source");
            }
            if (!(scales[0] > 0) || !(scales[1] > 0)) throw py::value_error("scales must be positive");
            // Samples needed: rows r0 .. r0 + h (+ 1 for the corners of the area methods).
            const i64 extra = warp_by_area(method) ? 1 : 0;
            const i64 last_row = static_cast<i64>(sp.dst_r0) + sp.dst_h + extra;  // exclusive
            const i64 last_col = static_cast<i64>(sp.dst_c0) + sp.dst_w + extra;
            const WarpCoords c = make_coords(kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense,
                                             dense_row0, dense_col0, transformer, transformer_args, sp.dst_r0,
                                             sp.dst_c0, last_row, last_col);
            Threads threads(n_threads);
            if (c.transformer != nullptr && c.n_transformer_args < max_threads()) {
                throw py::value_error("one transformer argument per thread");
            }
            py::array out(src.dtype(), std::vector<py::ssize_t>{sp.bands, sp.dst_h, sp.dst_w});
            void* o = out.mutable_data();
            auto run = [&](const auto* p) {
                using T = std::remove_const_t<std::remove_pointer_t<decltype(p)>>;
                py::gil_scoped_release release;
                warp<T>(p, static_cast<T*>(o), c, sp);
                return 0;
            };
            if (is_type<std::uint64_t>(src.dtype())) run(static_cast<const std::uint64_t*>(src.data()));
            else with_typed(src, run);
            return out;
        },
        py::arg("src"), py::arg("method"), py::arg("src_shape"), py::arg("window"), py::arg("scales"),
        py::arg("has_src_nodata"), py::arg("src_nodata"), py::arg("dst_nodata"), py::arg("region"), py::arg("kind"),
        py::arg("offset"), py::arg("gt"), py::arg("inv"), py::arg("level"), py::arg("grid0"), py::arg("node_off"),
        py::arg("nodes"), py::arg("cells"), py::arg("dense"), py::arg("dense_row0"), py::arg("dense_col0"),
        py::arg("transformer"), py::arg("transformer_args"), py::arg("per_band") = true, py::arg("n_threads") = 0,
        "GDAL's warp kernel on a region of the destination: (bands, rows, columns).");

    m.def(
        "coords",
        [](int kind, double offset, const std::array<double, 6>& gt, const std::array<double, 6>& inv,
           const Int8Array& level, const DoubleArray& grid0, const I64Array& node_off, const DoubleArray& nodes,
           const std::array<i64, 3>& cells, const DoubleArray& dense, i64 dense_row0, i64 dense_col0,
           std::uintptr_t transformer, const PointerArray& transformer_args, const std::array<i64, 4>& region) {
            const i64 r0 = region[0], c0 = region[1], nr = region[2], nc = region[3];
            if (r0 < 0 || c0 < 0 || nr < 0 || nc < 0 || nc > INT32_MAX) throw py::value_error("bad region");
            const WarpCoords c = make_coords(kind, offset, gt, inv, level, grid0, node_off, nodes, cells, dense,
                                             dense_row0, dense_col0, transformer, transformer_args, r0, c0, r0 + nr,
                                             c0 + nc);
            py::array_t<double> out({nr, nc, static_cast<i64>(2)});
            double* o = out.mutable_data();
            {
                py::gil_scoped_release release;
                std::vector<double> x(static_cast<std::size_t>(nc)), y(static_cast<std::size_t>(nc));
                for (i64 j = 0; j < nr; ++j) {
                    c.row(r0 + j, c0, static_cast<int>(nc), x.data(), y.data());
                    for (i64 i = 0; i < nc; ++i) {
                        o[2 * (j * nc + i)] = x[static_cast<std::size_t>(i)];
                        o[2 * (j * nc + i) + 1] = y[static_cast<std::size_t>(i)];
                    }
                }
            }
            return out;
        },
        py::arg("kind"), py::arg("offset"), py::arg("gt"), py::arg("inv"), py::arg("level"), py::arg("grid0"),
        py::arg("node_off"), py::arg("nodes"), py::arg("cells"), py::arg("dense"), py::arg("dense_row0"),
        py::arg("dense_col0"), py::arg("transformer"), py::arg("transformer_args"), py::arg("region"),
        "The source coordinates the warp kernel takes for the samples of a region: (rows, columns, 2).");

    m.def(
        "lattice_checks",
        [](const py::array_t<double, py::array::c_style | py::array::forcecast>& g, int src_h, int src_w,
           int n_threads) {
            if (g.ndim() != 4 || g.shape(3) != 2 || g.shape(1) % 2 != 1 || g.shape(2) % 2 != 1 || g.shape(1) < 3 ||
                g.shape(2) < 3) {
                throw py::value_error("g must be (n, 2 my + 1, 2 mx + 1, 2)");
            }
            const i64 n = g.shape(0);
            const int my = static_cast<int>((g.shape(1) - 1) / 2), mx = static_cast<int>((g.shape(2) - 1) / 2);
            py::array_t<double> err({n, static_cast<i64>(my), static_cast<i64>(mx)});
            py::array_t<bool> bad({n, static_cast<i64>(my), static_cast<i64>(mx)});
            py::array_t<bool> outside({n, static_cast<i64>(my), static_cast<i64>(mx)});
            py::array_t<double> bbox({n, static_cast<i64>(my), static_cast<i64>(mx), static_cast<i64>(4)});
            {
                Threads threads(n_threads);
                py::gil_scoped_release release;
                lattice_checks(g.data(), n, my, mx, src_h, src_w, err.mutable_data(),
                               reinterpret_cast<std::uint8_t*>(bad.mutable_data()),
                               reinterpret_cast<std::uint8_t*>(outside.mutable_data()), bbox.mutable_data());
            }
            return py::make_tuple(err, bad, outside, bbox);
        },
        py::arg("g"), py::arg("src_h"), py::arg("src_w"), py::arg("n_threads") = 0,
        "Checks of lattice cells: (err, bad, outside, bbox), one per cell of every grid.");
}

}  // namespace zeit::warp
