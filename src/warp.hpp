// SPDX-License-Identifier: GPL-2.0-or-later
//
// The warp kernel of load_raster(..., like=): GDAL's resampling
// (gdalwarpkernel.cpp, GDAL 3.12) as GDAL runs it for a destination with a
// NoData value, on the source coordinates of a fixed set of destination
// points (exact ones, or interpolated in a lattice anchored at the
// destination's origin), so that any region of the destination gets the same
// values whoever computes it.
//
// From landschaft 1.35.0 (src/cpp/io/warp.hpp, by the same author), with one
// addition: per_band, a validity mask of each band's own (the dates of a time
// series have clouds of their own), instead of GDAL's unified one.
#pragma once

#include <cstdint>

namespace zeit::warp {

using i64 = std::int64_t;

// GDAL's GDALResampleAlg numbers.
enum WarpMethod : int {
    WARP_NEAREST = 0,
    WARP_BILINEAR = 1,
    WARP_CUBIC = 2,
    WARP_CUBICSPLINE = 3,
    WARP_LANCZOS = 4,
    WARP_AVERAGE = 5,
    WARP_MODE = 6,
    WARP_MAX = 8,
    WARP_MIN = 9,
    WARP_MED = 10,
    WARP_Q1 = 11,
    WARP_Q3 = 12,
    WARP_RMS = 14,
};

// True for the methods that take the source cells a destination cell covers
// (their samples are the cells' corners, not their centres).
inline bool warp_by_area(int method) { return method >= WARP_AVERAGE; }

// Samples a side of a lattice cell.
constexpr int kWarpCell = 16;

// GDAL's GDALTransformerFunc (GDALGenImgProjTransform): pixel coordinates of
// the destination to the source's (dst_to_src) or back, in place; success[k]
// false where the transformation fails.
using TransformerFunc = int (*)(void* arg, int dst_to_src, int n, double* x, double* y, double* z, int* success);

// The lattice's interpolation between the nodes around a sample, at fractions
// fa (along x) and fb (along y) of the way: along y on the left and right
// edges, then along x.
inline double lattice_interp(double p00, double p10, double p01, double p11, double fa, double fb) {
    const double left = p00 * (1.0 - fb) + p01 * fb;
    const double right = p10 * (1.0 - fb) + p11 * fb;
    return left * (1.0 - fa) + right * fa;
}

// The source pixel coordinates (column, row; 0 at the source's top-left
// corner) of the destination samples. Sample (i, j) is the destination point
// (i + offset, j + offset) in destination pixel coordinates: the cell centres
// (offset 0.5) or the cell corners (offset 0, one more sample a side).
struct WarpCoords {
    enum Kind : int { AFFINE = 0, LATTICE = 1, DENSE = 2, EXACT = 3 };
    int kind = AFFINE;
    double offset = 0.5;
    // AFFINE (the same CRS): the destination geotransform and the inverse of
    // the source's, applied as GDAL's GDALGenImgProjTransform does.
    double gt[6]{};
    double inv[6]{};
    // LATTICE: cells of kWarpCell x kWarpCell samples, the cells_x x cells_y
    // from cell (cell_x0, cell_y0) on. Cell (cx, cy), number
    // c = (cy - cell_y0) * cells_x + cx - cell_x0, has its nodes every
    // kWarpCell >> level[c] samples: at level 0 the corners, in grid0
    // ((cells_y + 1) x (cells_x + 1) x, y pairs, every kWarpCell samples),
    // else an (n + 1) x (n + 1) block of x, y pairs at nodes + 2 * node_off[c].
    // The samples between nodes are interpolated (lattice_interp).
    i64 cells_x = 0, cells_y = 0, cell_x0 = 0, cell_y0 = 0;
    const std::int8_t* level = nullptr;
    const double* grid0 = nullptr;
    const i64* node_off = nullptr;
    const double* nodes = nullptr;
    // DENSE: x, y pairs of every sample of rows dense_row0.., columns
    // dense_col0.. (dense_stride samples a row).
    const double* dense = nullptr;
    i64 dense_row0 = 0, dense_col0 = 0, dense_stride = 0;
    // The exact transformation (GDAL's transformer, one argument per OpenMP
    // thread): the samples of EXACT coordinates, and the points GDAL
    // transforms on demand (near the edge of a projection's domain, across
    // the antimeridian); without it those checks are left out.
    TransformerFunc transformer = nullptr;
    void* const* transformer_args = nullptr;
    int n_transformer_args = 0;

    // x, y of the samples (i0 .. i0 + n - 1, j).
    void row(i64 j, i64 i0, int n, double* x, double* y) const;
    // Transforms n points in place; false where it fails (or no transformer).
    void transform(bool dst_to_src, int n, double* x, double* y, int* ok) const;
};

struct WarpSpec {
    int method = WARP_NEAREST;
    int bands = 1;
    // The whole source, and the window of it that `src` holds.
    int src_h = 0, src_w = 0;
    int win_r0 = 0, win_c0 = 0, win_h = 0, win_w = 0;
    // GDAL's dfXScale / dfYScale (destination cells per source cell).
    double xscale = 1.0, yscale = 1.0;
    // Source NoData (GDAL's validity mask: a cell is valid when any band is
    // not NoData; no mask without a NoData value).
    bool has_src_nodata = false;
    // Every band on its own (its own validity mask and NoData avoidance), as
    // GDAL warping each band alone; the coordinates are still computed once.
    bool per_band = false;
    double src_nodata = 0.0;
    double dst_nodata = 0.0;
    // The destination region computed: rows dst_r0.., columns dst_c0.. of
    // the whole destination.
    int dst_r0 = 0, dst_c0 = 0, dst_h = 0, dst_w = 0;
};

// Warps src ((bands, win_h, win_w), row-major) into dst ((bands, dst_h,
// dst_w)), the cells without a value set to dst_nodata. Rows in parallel.
template <class T>
void warp(const T* src, T* dst, const WarpCoords& coords, const WarpSpec& spec);

// The checks of lattice cells: g holds n grids of (2 my + 1) x (2 mx + 1) x, y
// pairs, exact coordinates every half step of a cell (its nodes at even
// indices). For every cell (n x my x mx): err, the largest |dx| + |dy| of
// lattice_interp at the cell's centre and the middles of its top and left
// edges (the other edges are its neighbours'); bad, a node or check not
// finite; outside, the cell well outside the source (src_h x src_w), its
// errors included; bbox, xmin, xmax, ymin, ymax of its finite corner nodes
// (NaN when none).
void lattice_checks(const double* g, i64 n, int my, int mx, int src_h, int src_w, double* err, std::uint8_t* bad,
                    std::uint8_t* outside, double* bbox);

}  // namespace zeit::warp
