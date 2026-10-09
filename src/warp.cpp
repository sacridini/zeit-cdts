// SPDX-License-Identifier: GPL-2.0-or-later
//
// From landschaft 1.35.0 (src/cpp/io/warp.cpp, by the same author); the only
// change is WarpSpec::per_band (each band with a validity mask of its own).
//
// The warp kernel of load_raster(..., like=), ported from GDAL 3.12's
// alg/gdalwarpkernel.cpp (MIT, Frank Warmerdam, Even Rouault and others):
// the code paths GDAL takes for a destination with a NoData value and
// UNIFIED_SRC_NODATA (as rasterio's reproject sets them up), with the same
// operations in the same order. GDAL's kernels see their source pixels from
// the origin of the window they read; here coordinates are the whole
// source's (the window is only used to index the cells it holds), which gives
// the same values (the subtraction of a whole window offset is exact) and
// makes them independent of the window.
//
// The points GDAL transforms on demand (the edge of a projection's domain,
// the antimeridian) go through GDAL's own transformer when the caller gives
// it. Left out: sum (GDAL's own warp, see _reproject.py) and the retries of
// GDAL's approximate transformation (its own; the coordinates here are
// fixed beforehand).
#include "warp.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace zeit::warp {

void WarpCoords::transform(bool dst_to_src, int n, double* x, double* y, int* ok) const {
    if (transformer == nullptr || n_transformer_args < 1) {
        std::fill_n(ok, n, 0);
        return;
    }
#ifdef _OPENMP
    const int t = omp_get_thread_num();
#else
    const int t = 0;
#endif
    thread_local std::vector<double> z;
    z.assign(static_cast<std::size_t>(n), 0.0);
    transformer(transformer_args[t % n_transformer_args], dst_to_src ? 1 : 0, n, x, y, z.data(), ok);
}

void WarpCoords::row(i64 j, i64 i0, int n, double* x, double* y) const {
    if (kind == AFFINE) {
        // GDALGenImgProjTransform without reprojection.
        const double v = static_cast<double>(j) + offset;
        for (int k = 0; k < n; ++k) {
            const double u = static_cast<double>(i0 + k) + offset;
            const double gx = gt[0] + u * gt[1] + v * gt[2];
            const double gy = gt[3] + u * gt[4] + v * gt[5];
            x[k] = inv[0] + gx * inv[1] + gy * inv[2];
            y[k] = inv[3] + gx * inv[4] + gy * inv[5];
        }
        return;
    }
    if (kind == EXACT) {
        // GWKRealCase / GWKAverageOrModeComputeLineCoords: the samples of a row at once.
        const double v = static_cast<double>(j) + offset;
        for (int k = 0; k < n; ++k) {
            x[k] = static_cast<double>(i0 + k) + offset;
            y[k] = v;
        }
        thread_local std::vector<int> ok;
        ok.resize(static_cast<std::size_t>(n));
        transform(true, n, x, y, ok.data());
        for (int k = 0; k < n; ++k) {
            if (!ok[static_cast<std::size_t>(k)]) x[k] = y[k] = std::numeric_limits<double>::infinity();
        }
        return;
    }
    if (kind == DENSE) {
        const double* p = dense + 2 * ((j - dense_row0) * dense_stride + (i0 - dense_col0));
        for (int k = 0; k < n; ++k) {
            x[k] = p[2 * k];
            y[k] = p[2 * k + 1];
        }
        return;
    }
    // LATTICE: run over the samples of the row a sub-cell at a time; every
    // sample's value depends only on its sub-cell's nodes and its fractions.
    const i64 cy = j / kWarpCell;
    const int lj = static_cast<int>(j - cy * kWarpCell);
    int k = 0;
    while (k < n) {
        const i64 i = i0 + k;
        const i64 cx = i / kWarpCell;
        int li = static_cast<int>(i - cx * kWarpCell);
        const i64 cell = (cy - cell_y0) * cells_x + (cx - cell_x0);
        const int lev = level[cell];
        const int step = kWarpCell >> lev;
        const double* base;
        i64 stride;
        if (lev == 0) {
            base = grid0 + 2 * ((cy - cell_y0) * (cells_x + 1) + (cx - cell_x0));
            stride = cells_x + 1;
        } else {
            base = nodes + 2 * node_off[cell];
            stride = kWarpCell / step + 1;
        }
        const double inv_step = 1.0 / step;  // exact: steps are powers of 2
        const int b = lj / step;
        const int fbn = lj - b * step;
        const double fb = fbn * inv_step;
        const int end = std::min(n, k + (kWarpCell - li));
        while (k < end) {
            const int a = li / step;
            const double* p00 = base + 2 * (static_cast<i64>(b) * stride + a);
            const double* p10 = p00 + 2;
            const double* p01 = p00 + 2 * stride;
            const double* p11 = p01 + 2;
            const int sub_end = std::min(end, k + (step - (li - a * step)));
            const double lx = p00[0] * (1.0 - fb) + p01[0] * fb, rx = p10[0] * (1.0 - fb) + p11[0] * fb;
            const double ly = p00[1] * (1.0 - fb) + p01[1] * fb, ry = p10[1] * (1.0 - fb) + p11[1] * fb;
            for (; k < sub_end; ++k, ++li) {
                const int fan = li - a * step;
                if (fan == 0 && fbn == 0) {  // a node: exact, whatever its neighbours
                    x[k] = p00[0];
                    y[k] = p00[1];
                    continue;
                }
                const double fa = fan * inv_step;
                x[k] = lx * (1.0 - fa) + rx * fa;
                y[k] = ly * (1.0 - fa) + ry * fa;
            }
        }
    }
}

void lattice_checks(const double* g, i64 n, int my, int mx, int src_h, int src_w, double* err, std::uint8_t* bad,
                    std::uint8_t* outside, double* bbox) {
    const i64 sx = 2 * static_cast<i64>(mx) + 1, sy = 2 * static_cast<i64>(my) + 1;
    const i64 cells = n * my * mx;
    const double nan = std::numeric_limits<double>::quiet_NaN();
#pragma omp parallel for schedule(static)
    for (i64 c = 0; c < cells; ++c) {
        const i64 q = c / (static_cast<i64>(my) * mx);
        const int b = static_cast<int>((c / mx) % my), a = static_cast<int>(c % mx);
        const double* grid = g + q * sy * sx * 2;
        auto pt = [&](int r, int s) { return grid + 2 * ((2 * static_cast<i64>(b) + r) * sx + 2 * a + s); };
        const double *p00 = pt(0, 0), *p10 = pt(0, 2), *p01 = pt(2, 0), *p11 = pt(2, 2);
        const double* checks[3] = {pt(1, 1), pt(0, 1), pt(1, 0)};
        const double fractions[3][2] = {{0.5, 0.5}, {0.5, 0.0}, {0.0, 0.5}};
        double e = 0.0;
        bool finite = true;
        for (int k = 0; k < 3; ++k) {
            const double fa = fractions[k][0], fb = fractions[k][1];
            const double ex = std::fabs(lattice_interp(p00[0], p10[0], p01[0], p11[0], fa, fb) - checks[k][0]);
            const double ey = std::fabs(lattice_interp(p00[1], p10[1], p01[1], p11[1], fa, fb) - checks[k][1]);
            const double ek = ex + ey;
            if (!std::isfinite(ek)) finite = false;
            else e = std::max(e, ek);
        }
        double lo_x = std::numeric_limits<double>::infinity(), hi_x = -lo_x, lo_y = lo_x, hi_y = -lo_x;
        for (int r = 0; r < 3; ++r) {
            for (int s = 0; s < 3; ++s) {
                const double* p = pt(r, s);
                if (!std::isfinite(p[0]) || !std::isfinite(p[1])) {
                    finite = false;
                    continue;
                }
                lo_x = std::min(lo_x, p[0]);
                hi_x = std::max(hi_x, p[0]);
                lo_y = std::min(lo_y, p[1]);
                hi_y = std::max(hi_y, p[1]);
            }
        }
        err[c] = finite ? e : nan;
        bad[c] = finite ? 0 : 1;
        const double margin = 2 * e + 4;
        outside[c] = finite && (hi_x < -margin || lo_x > src_w + margin || hi_y < -margin || lo_y > src_h + margin);
        double* box = bbox + 4 * c;
        box[0] = box[2] = std::numeric_limits<double>::infinity();
        box[1] = box[3] = -std::numeric_limits<double>::infinity();
        bool any = false;
        for (const double* p : {p00, p10, p01, p11}) {
            if (!std::isfinite(p[0]) || !std::isfinite(p[1])) continue;
            any = true;
            box[0] = std::min(box[0], p[0]);
            box[1] = std::max(box[1], p[0]);
            box[2] = std::min(box[2], p[1]);
            box[3] = std::max(box[3], p[1]);
        }
        if (!any) box[0] = box[1] = box[2] = box[3] = nan;
    }
}

namespace {

constexpr double kBandDensityThreshold = 0.0000000001;
constexpr double kSrcDensityThreshold = 0.000000001;
constexpr double kPi = 3.14159265358979323846;

int filter_radius(int method) {
    switch (method) {
        case WARP_BILINEAR: return 1;
        case WARP_CUBIC: return 2;
        case WARP_CUBICSPLINE: return 2;
        case WARP_LANCZOS: return 3;
        default: return 0;
    }
}

// GWKBilinear, GWKCubic (CubicKernel) and GWKBSpline.
double filter_bilinear(double dfX) {
    const double dfAbsX = std::fabs(dfX);
    if (dfAbsX <= 1.0) return 1 - dfAbsX;
    return 0.0;
}

double filter_cubic(double dfX) {
    const double dfAbsX = std::fabs(dfX);
    if (dfAbsX <= 1.0) {
        const double dfX2 = dfX * dfX;
        return dfX2 * (1.5 * dfAbsX - 2.5) + 1;
    } else if (dfAbsX <= 2.0) {
        const double dfX2 = dfX * dfX;
        return dfX2 * (-0.5 * dfAbsX + 2.5) - 4 * dfAbsX + 2;
    }
    return 0.0;
}

double filter_bspline(double x) {
    const double xp2 = x + 2.0;
    const double xp1 = x + 1.0;
    const double xm1 = x - 1.0;
    const double xp2c = xp2 * xp2 * xp2;
    // (GDAL's nested conditional, unrolled; the same sums in the same order)
    if (!(xp2 > 0.0)) return 0.0;
    if (!(xp1 > 0.0)) return 0.0 + xp2c;
    const double inner = (x > 0.0) ? ((xm1 > 0.0) ? -4.0 * xm1 * xm1 * xm1 : 0.0) + 6.0 * x * x * x : 0.0;
    return (inner + -4.0 * xp1 * xp1 * xp1) + xp2c;
}

using FilterFn = double (*)(double);

FilterFn filter_fn(int method) {
    switch (method) {
        case WARP_BILINEAR: return filter_bilinear;
        case WARP_CUBIC: return filter_cubic;
        case WARP_CUBICSPLINE: return filter_bspline;
        default: return nullptr;
    }
}

// floor and ceil of values in the range of int, as ints (MSVC calls the CRT
// for std::floor and std::ceil without SSE4.1; the same results).
inline int ifloor(double v) {
    const int i = static_cast<int>(v);
    return static_cast<double>(i) > v ? i - 1 : i;
}

inline int iceil(double v) {
    const int i = static_cast<int>(v);
    return static_cast<double>(i) < v ? i + 1 : i;
}

// ClampRoundAndAvoidNoData's clamping and rounding.
template <class T>
T clamp_round(double dfReal) {
    if constexpr (std::is_integral_v<T>) {
        if (dfReal < static_cast<double>(std::numeric_limits<T>::lowest())) return std::numeric_limits<T>::lowest();
        if (dfReal > static_cast<double>(std::numeric_limits<T>::max())) return std::numeric_limits<T>::max();
        if constexpr (std::is_signed_v<T> && sizeof(T) <= 4) return static_cast<T>(ifloor(dfReal + 0.5));
        else if constexpr (std::is_signed_v<T>) return static_cast<T>(std::floor(dfReal + 0.5));
        else return static_cast<T>(dfReal + 0.5);
    } else {
        return static_cast<T>(dfReal);
    }
}

// AvoidNoData: a value equal to the destination NoData moves one step away.
template <class T>
void avoid_nodata(T& v) {
    if constexpr (std::is_integral_v<T>) {
        if (v == std::numeric_limits<T>::lowest()) v = static_cast<T>(std::numeric_limits<T>::lowest() + 1);
        else v--;
    } else {
        if (v == std::numeric_limits<T>::max()) v = std::nextafter(v, static_cast<T>(0));
        else v = std::nextafter(v, std::numeric_limits<T>::max());
    }
}

// GDAL's ARE_REAL_EQUAL (float epsilon for doubles too).
inline bool real_equal(float a, float b) {
    return a == b || std::abs(a - b) < std::numeric_limits<float>::epsilon() * std::abs(a + b) * 2;
}
inline bool real_equal(double a, double b) {
    return a == b ||
           std::abs(a - b) < static_cast<double>(std::numeric_limits<float>::epsilon()) * std::abs(a + b) * 2;
}

// GDALWarpNoDataMasker: is v the source NoData?
template <class T>
struct NoDataTest {
    bool active = true;
    int nint = 0;
    float fnd = 0.0f;
    double dnd = 0.0;
    bool nan = false;

    explicit NoDataTest(double nodata) {
        if constexpr (std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t> ||
                      std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t>) {
            if (!(nodata >= static_cast<double>(std::numeric_limits<T>::min())) ||
                nodata > static_cast<double>(std::numeric_limits<T>::max()) + 0.000001) {
                active = false;
            } else {
                nint = static_cast<int>(std::floor(nodata + 0.000001));
            }
        } else if constexpr (std::is_same_v<T, float>) {
            fnd = static_cast<float>(nodata);
            nan = std::isnan(fnd);
        } else {
            dnd = nodata;
            nan = std::isnan(dnd);
        }
    }

    bool operator()(T v) const {
        if constexpr (std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t> ||
                      std::is_same_v<T, std::int16_t> || std::is_same_v<T, std::uint16_t>) {
            return static_cast<int>(v) == nint;
        } else if constexpr (std::is_same_v<T, float>) {
            return nan ? std::isnan(v) : real_equal(v, fnd);
        } else {
            const double dv = static_cast<double>(v);
            return nan ? std::isnan(dv) : real_equal(dv, dnd);
        }
    }
};

// The source cells given: a window of the whole source, and GDAL's unified
// validity mask (a cell is valid when any band is not NoData): tested on the
// value itself for one band, a mask computed beforehand for several.
template <class T>
struct Src {
    const T* data = nullptr;
    i64 plane = 0;
    int h = 0, w = 0;
    int r0 = 0, c0 = 0, wh = 0, ww = 0;
    bool masked = false;                   // GDAL has a validity mask
    const std::uint8_t* valid = nullptr;  // several bands
    NoDataTest<T> nodata{0.0};

    i64 at(int r, int c) const { return static_cast<i64>(r - r0) * ww + (c - c0); }
    bool held(int r, int c) const { return r >= r0 && c >= c0 && r < r0 + wh && c < c0 + ww; }
    bool valid_at(i64 k) const { return !masked || (valid != nullptr ? valid[k] != 0 : !nodata(data[k])); }
    // GDAL's density of a cell: 1 when valid, else 0.
    bool ok(int r, int c) const { return held(r, c) && valid_at(at(r, c)); }
    double val(int b, int r, int c) const { return static_cast<double>(data[b * plane + at(r, c)]); }
    T raw(int b, int r, int c) const { return data[b * plane + at(r, c)]; }
};

struct Kernel {
    int method = 0;
    double xscale = 1.0, yscale = 1.0;
    bool use4 = true;
    int nXRadius = 0, nYRadius = 0, nFiltInitX = 0, nFiltInitY = 0;
    FilterFn filter = nullptr;
    // GWKResampleCreateWrkStruct's Lanczos constants.
    double cosPiXScaleOver3 = 0, sinPiXScaleOver3 = 0, cosPiXScale = 0, sinPiXScale = 0;
    double cosPiYScaleOver3 = 0, sinPiYScaleOver3 = 0, cosPiYScale = 0, sinPiYScale = 0;

    Kernel(int method_, double xs, double ys) : method(method_), xscale(xs), yscale(ys) {
        use4 = xscale >= 0.95 && yscale >= 0.95;
        const int radius = filter_radius(method);
        const double dfXFilter = radius, dfYFilter = radius;
        nXRadius = xscale < 1.0 ? static_cast<int>(std::ceil(dfXFilter / xscale)) : static_cast<int>(dfXFilter);
        nYRadius = yscale < 1.0 ? static_cast<int>(std::ceil(dfYFilter / yscale)) : static_cast<int>(dfYFilter);
        nFiltInitX = ((radius + 1) % 2) - nXRadius;
        nFiltInitY = ((radius + 1) % 2) - nYRadius;
        filter = filter_fn(method);
        if (method == WARP_LANCZOS) {
            if (xscale < 1) {
                cosPiXScaleOver3 = std::cos(kPi / 3 * xscale);
                sinPiXScaleOver3 = std::sqrt(1 - cosPiXScaleOver3 * cosPiXScaleOver3);
                cosPiXScale = (4 * cosPiXScaleOver3 * cosPiXScaleOver3 - 3) * cosPiXScaleOver3;
                sinPiXScale = std::sqrt(1 - cosPiXScale * cosPiXScale);
            }
            if (yscale < 1) {
                cosPiYScaleOver3 = std::cos(kPi / 3 * yscale);
                sinPiYScaleOver3 = std::sqrt(1 - cosPiYScaleOver3 * cosPiYScaleOver3);
                cosPiYScale = (4 * cosPiYScaleOver3 * cosPiYScaleOver3 - 3) * cosPiYScaleOver3;
                sinPiYScale = std::sqrt(1 - cosPiYScale * cosPiYScale);
            }
        }
    }
};

// A thread's work space (GWKResampleWrkStruct).
struct Work {
    std::vector<double> weightsX, weightsY;
    int lastSrcX = -10, lastSrcY = -10;
    double lastDeltaX = -10, lastDeltaY = -10;

    explicit Work(const Kernel& k)
        : weightsX(static_cast<std::size_t>((k.nXRadius + 1) * 2), 0.0),
          weightsY(static_cast<std::size_t>((k.nYRadius + 1) * 2), 0.0) {}
};

// GWKBilinearResample4Sample.
template <class T>
bool bilinear4(const Src<T>& s, int band, double dfSrcX, double dfSrcY, double* pdfDensity, double* pdfReal) {
    int iSrcX = ifloor(dfSrcX - 0.5);
    int iSrcY = ifloor(dfSrcY - 0.5);
    double dfRatioX = 1.5 - (dfSrcX - iSrcX);
    double dfRatioY = 1.5 - (dfSrcY - iSrcY);
    if (iSrcX == -1) {
        iSrcX = 0;
        dfRatioX = 1;
    }
    if (iSrcY == -1) {
        iSrcY = 0;
        dfRatioY = 1;
    }
    double dfAccumulatorReal = 0.0, dfAccumulatorDensity = 0.0, dfAccumulatorDivisor = 0.0;
    // The four cells in GDAL's order (upper left, upper right, lower left,
    // lower right; iSrcX, iSrcY >= 0 here), those in the source and valid.
    const double m[4] = {dfRatioX * dfRatioY, (1.0 - dfRatioX) * dfRatioY, dfRatioX * (1.0 - dfRatioY),
                         (1.0 - dfRatioX) * (1.0 - dfRatioY)};
    if (iSrcY + 1 < s.h && iSrcX + 1 < s.w && s.held(iSrcY, iSrcX) && s.held(iSrcY + 1, iSrcX + 1)) {
        const i64 k00 = s.at(iSrcY, iSrcX), k10 = k00 + s.ww;
        const T* d = s.data + band * s.plane;
        const i64 ks[4] = {k00, k00 + 1, k10, k10 + 1};
        for (int q = 0; q < 4; ++q) {
            if (!s.valid_at(ks[q])) continue;
            dfAccumulatorDivisor += m[q];
            dfAccumulatorReal += static_cast<double>(d[ks[q]]) * m[q];
            dfAccumulatorDensity += 1.0 * m[q];
        }
    } else {
        for (int q = 0; q < 4; ++q) {
            const int r = iSrcY + q / 2, c = iSrcX + q % 2;
            if (r >= s.h || c >= s.w || !s.ok(r, c)) continue;
            dfAccumulatorDivisor += m[q];
            dfAccumulatorReal += s.val(band, r, c) * m[q];
            dfAccumulatorDensity += 1.0 * m[q];
        }
    }

    if (dfAccumulatorDivisor == 1.0) {
        *pdfReal = dfAccumulatorReal;
        *pdfDensity = dfAccumulatorDensity;
        return false;
    } else if (dfAccumulatorDivisor < 0.00001) {
        *pdfReal = 0.0;
        *pdfDensity = 0.0;
        return false;
    }
    *pdfReal = dfAccumulatorReal / dfAccumulatorDivisor;
    *pdfDensity = dfAccumulatorDensity / dfAccumulatorDivisor;
    return true;
}

// GWKCubicComputeWeights and CONVOL4.
inline void cubic_weights(double x, double coeffs[4]) {
    const double halfX = 0.5 * x;
    const double threeX = 3.0 * x;
    const double halfX2 = halfX * x;
    coeffs[0] = halfX * (-1 + x * (2 - x));
    coeffs[1] = 1 + halfX2 * (-5 + threeX);
    coeffs[2] = halfX * (1 + x * (4 - threeX));
    coeffs[3] = halfX2 * (-1 + x);
}

inline double convol4(const double v1[4], const double v2[4]) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

// GWKCubicResample4Sample.
template <class T>
bool cubic4(const Src<T>& s, int band, double dfSrcX, double dfSrcY, double* pdfDensity, double* pdfReal) {
    const int iSrcX = static_cast<int>(dfSrcX - 0.5);
    const int iSrcY = static_cast<int>(dfSrcY - 0.5);
    const double dfDeltaX = dfSrcX - 0.5 - iSrcX;
    const double dfDeltaY = dfSrcY - 0.5 - iSrcY;
    if (iSrcX - 1 < 0 || iSrcX + 2 >= s.w || iSrcY - 1 < 0 || iSrcY + 2 >= s.h) {
        return bilinear4(s, band, dfSrcX, dfSrcY, pdfDensity, pdfReal);
    }
    double adfValueDens[4] = {}, adfValueReal[4] = {}, adfCoeffsX[4] = {};
    cubic_weights(dfDeltaX, adfCoeffsX);
    // Any of the 4 x 4 cells invalid: bilinear (GDAL checks them row by row,
    // the same outcome). Every row's density is the same sum of the weights.
    const double ones[4] = {1.0, 1.0, 1.0, 1.0};
    const double row_density = convol4(adfCoeffsX, ones);
    if (s.held(iSrcY - 1, iSrcX - 1) && s.held(iSrcY + 2, iSrcX + 2)) {
        i64 k0[4];
        for (int i = 0; i < 4; ++i) k0[i] = s.at(iSrcY - 1 + i, iSrcX - 1);
        if (s.masked) {
            for (int i = 0; i < 4; ++i) {
                for (int k = 0; k < 4; ++k) {
                    if (!s.valid_at(k0[i] + k)) return bilinear4(s, band, dfSrcX, dfSrcY, pdfDensity, pdfReal);
                }
            }
        }
        const T* d = s.data + band * s.plane;
        for (int i = 0; i < 4; ++i) {
            const T* p = d + k0[i];
            const double adfReal[4] = {static_cast<double>(p[0]), static_cast<double>(p[1]),
                                       static_cast<double>(p[2]), static_cast<double>(p[3])};
            adfValueDens[i] = row_density;
            adfValueReal[i] = convol4(adfCoeffsX, adfReal);
        }
    } else {
        for (int i = -1; i < 3; i++) {
            double adfReal[4];
            for (int k = 0; k < 4; ++k) {
                if (!s.ok(iSrcY + i, iSrcX - 1 + k)) return bilinear4(s, band, dfSrcX, dfSrcY, pdfDensity, pdfReal);
                adfReal[k] = s.val(band, iSrcY + i, iSrcX - 1 + k);
            }
            adfValueDens[i + 1] = row_density;
            adfValueReal[i + 1] = convol4(adfCoeffsX, adfReal);
        }
    }
    double adfCoeffsY[4] = {};
    cubic_weights(dfDeltaY, adfCoeffsY);
    *pdfDensity = convol4(adfCoeffsY, adfValueDens);
    *pdfReal = convol4(adfCoeffsY, adfValueReal);
    return true;
}

// GWKResample (bilinear and cubic when downsampling, cubic spline).
template <class T>
bool resample(const Src<T>& s, const Kernel& k, Work& wk, int band, double dfSrcX, double dfSrcY,
              double* pdfDensity, double* pdfReal) {
    const bool masked = s.masked;
    double dfAccumulatorReal = 0.0, dfAccumulatorDensity = 0.0, dfAccumulatorWeight = 0.0;
    const int iSrcX = ifloor(dfSrcX - 0.5);
    const int iSrcY = ifloor(dfSrcY - 0.5);
    const double dfDeltaX = dfSrcX - 0.5 - iSrcX;
    const double dfDeltaY = dfSrcY - 0.5 - iSrcY;
    const double dfXScale = k.xscale, dfYScale = k.yscale;

    int j = k.nFiltInitY;
    int jMax = k.nYRadius;
    if (iSrcY + j < 0) j = -iSrcY;
    if (iSrcY + jMax >= s.h) jMax = s.h - iSrcY - 1;
    int iMin = k.nFiltInitX;
    int iMax = k.nXRadius;
    if (iSrcX + iMin < 0) iMin = -iSrcX;
    if (iSrcX + iMax >= s.w) iMax = s.w - iSrcX - 1;

    const bool bXScaleBelow1 = dfXScale < 1.0;
    const bool bYScaleBelow1 = dfYScale < 1.0;
    double* padfWeightsX = wk.weightsX.data();
    for (int i = iMin; i <= iMax; ++i) {
        padfWeightsX[i - iMin] = bXScaleBelow1 ? k.filter((i - dfDeltaX) * dfXScale) : k.filter(i - dfDeltaX);
    }
    for (; j <= jMax; ++j) {
        const int r = iSrcY + j;
        const double dfWeight1 = bYScaleBelow1 ? k.filter((j - dfDeltaY) * dfYScale) : k.filter(j - dfDeltaY);
        double dfAccumulatorRealLocal = 0.0, dfAccumulatorDensityLocal = 0.0, dfAccumulatorWeightLocal = 0.0;
        for (int i = iMin; i <= iMax; ++i) {
            const int c = iSrcX + i;
            if (masked && !s.ok(r, c)) continue;
            const double dfWeight2 = padfWeightsX[i - iMin];
            dfAccumulatorRealLocal += s.val(band, r, c) * dfWeight2;
            if (masked) dfAccumulatorDensityLocal += 1.0 * dfWeight2;
            dfAccumulatorWeightLocal += dfWeight2;
        }
        dfAccumulatorReal += dfAccumulatorRealLocal * dfWeight1;
        dfAccumulatorDensity += dfAccumulatorDensityLocal * dfWeight1;
        dfAccumulatorWeight += dfAccumulatorWeightLocal * dfWeight1;
    }

    if (dfAccumulatorWeight < 0.000001 || (masked && dfAccumulatorDensity < 0.000001)) {
        *pdfDensity = 0.0;
        return false;
    }
    if (dfAccumulatorWeight < 0.99999 || dfAccumulatorWeight > 1.00001) {
        *pdfReal = dfAccumulatorReal / dfAccumulatorWeight;
        *pdfDensity = masked ? dfAccumulatorDensity / dfAccumulatorWeight : 1.0;
    } else {
        *pdfReal = dfAccumulatorReal;
        *pdfDensity = masked ? dfAccumulatorDensity : 1.0;
    }
    return true;
}

// GWKResampleOptimizedLanczos.
template <class T>
bool lanczos(const Src<T>& s, const Kernel& k, Work& wk, int band, double dfSrcX, double dfSrcY, double* pdfDensity,
             double* pdfReal) {
    const bool masked = s.masked;
    double dfAccumulatorReal = 0.0, dfAccumulatorDensity = 0.0, dfAccumulatorWeight = 0.0;
    const int iSrcX = ifloor(dfSrcX - 0.5);
    const int iSrcY = ifloor(dfSrcY - 0.5);
    const double dfDeltaX = dfSrcX - 0.5 - iSrcX;
    const double dfDeltaY = dfSrcY - 0.5 - iSrcY;
    const double dfXScale = k.xscale, dfYScale = k.yscale;

    double* const padfWeightsXShifted = wk.weightsX.data() - k.nFiltInitX;
    double* const padfWeightsYShifted = wk.weightsY.data() - k.nFiltInitY;

    int jMin = k.nFiltInitY;
    int jMax = k.nYRadius;
    if (iSrcY + jMin < 0) jMin = -iSrcY;
    if (iSrcY + jMax >= s.h) jMax = s.h - iSrcY - 1;
    int iMin = k.nFiltInitX;
    int iMax = k.nXRadius;
    if (iSrcX + iMin < 0) iMin = -iSrcX;
    if (iSrcX + iMax >= s.w) iMax = s.w - iSrcX - 1;

    constexpr double THREE_PI_PI = 3 * kPi * kPi;
    if (dfXScale < 1.0) {
        while ((iMin - dfDeltaX) * dfXScale < -3.0) iMin++;
        while ((iMax - dfDeltaX) * dfXScale > 3.0) iMax--;
        if (iSrcX != wk.lastSrcX || dfDeltaX != wk.lastDeltaX) {
            double dfX = (iMin - dfDeltaX) * dfXScale;
            const double dfPIXover3 = kPi / 3 * dfX;
            double dfCosOver3 = std::cos(dfPIXover3);
            double dfSinOver3 = std::sin(dfPIXover3);
            double dfSin = (3 - 4 * dfSinOver3 * dfSinOver3) * dfSinOver3;
            double dfCos = (4 * dfCosOver3 * dfCosOver3 - 3) * dfCosOver3;
            padfWeightsXShifted[iMin] = dfX == 0 ? 1.0 : THREE_PI_PI * dfSin * dfSinOver3 / (dfX * dfX);
            for (int i = iMin + 1; i <= iMax; ++i) {
                dfX += dfXScale;
                const double dfNewSin = dfSin * k.cosPiXScale + dfCos * k.sinPiXScale;
                const double dfNewSinOver3 = dfSinOver3 * k.cosPiXScaleOver3 + dfCosOver3 * k.sinPiXScaleOver3;
                padfWeightsXShifted[i] = dfX == 0 ? 1.0 : THREE_PI_PI * dfNewSin * dfNewSinOver3 / (dfX * dfX);
                const double dfNewCos = dfCos * k.cosPiXScale - dfSin * k.sinPiXScale;
                const double dfNewCosOver3 = dfCosOver3 * k.cosPiXScaleOver3 - dfSinOver3 * k.sinPiXScaleOver3;
                dfSin = dfNewSin;
                dfCos = dfNewCos;
                dfSinOver3 = dfNewSinOver3;
                dfCosOver3 = dfNewCosOver3;
            }
            wk.lastSrcX = iSrcX;
            wk.lastDeltaX = dfDeltaX;
        }
    } else {
        while (iMin - dfDeltaX < -3.0) iMin++;
        while (iMax - dfDeltaX > 3.0) iMax--;
        if (iSrcX != wk.lastSrcX || dfDeltaX != wk.lastDeltaX) {
            const double dfSinPIDeltaXOver3 = std::sin((-kPi / 3.0) * dfDeltaX);
            const double dfSin2PIDeltaXOver3 = dfSinPIDeltaXOver3 * dfSinPIDeltaXOver3;
            const double dfCosPIDeltaXOver3 = std::sqrt(1.0 - dfSin2PIDeltaXOver3);
            const double dfSinPIDeltaX = (3.0 - 4 * dfSin2PIDeltaXOver3) * dfSinPIDeltaXOver3;
            const double dfInvPI2Over3 = 3.0 / (kPi * kPi);
            const double dfInvPI2Over3xSinPIDeltaX = dfInvPI2Over3 * dfSinPIDeltaX;
            const double dfInvPI2Over3xSinPIDeltaXxm0d5SinPIDeltaXOver3 =
                -0.5 * dfInvPI2Over3xSinPIDeltaX * dfSinPIDeltaXOver3;
            const double dfSinPIOver3 = 0.8660254037844386;
            const double dfInvPI2Over3xSinPIDeltaXxSinPIOver3xCosPIDeltaXOver3 =
                dfSinPIOver3 * dfInvPI2Over3xSinPIDeltaX * dfCosPIDeltaXOver3;
            const double padfCst[] = {dfInvPI2Over3xSinPIDeltaX * dfSinPIDeltaXOver3,
                                      dfInvPI2Over3xSinPIDeltaXxm0d5SinPIDeltaXOver3 -
                                          dfInvPI2Over3xSinPIDeltaXxSinPIOver3xCosPIDeltaXOver3,
                                      dfInvPI2Over3xSinPIDeltaXxm0d5SinPIDeltaXOver3 +
                                          dfInvPI2Over3xSinPIDeltaXxSinPIOver3xCosPIDeltaXOver3};
            for (int i = iMin; i <= iMax; ++i) {
                const double dfX = i - dfDeltaX;
                padfWeightsXShifted[i] = dfX == 0.0 ? 1.0 : padfCst[(i + 3) % 3] / (dfX * dfX);
            }
            wk.lastSrcX = iSrcX;
            wk.lastDeltaX = dfDeltaX;
        }
    }

    if (dfYScale < 1.0) {
        while ((jMin - dfDeltaY) * dfYScale < -3.0) jMin++;
        while ((jMax - dfDeltaY) * dfYScale > 3.0) jMax--;
        if (iSrcY != wk.lastSrcY || dfDeltaY != wk.lastDeltaY) {
            double dfY = (jMin - dfDeltaY) * dfYScale;
            const double dfPIYover3 = kPi / 3 * dfY;
            double dfCosOver3 = std::cos(dfPIYover3);
            double dfSinOver3 = std::sin(dfPIYover3);
            double dfSin = (3 - 4 * dfSinOver3 * dfSinOver3) * dfSinOver3;
            double dfCos = (4 * dfCosOver3 * dfCosOver3 - 3) * dfCosOver3;
            padfWeightsYShifted[jMin] = dfY == 0 ? 1.0 : THREE_PI_PI * dfSin * dfSinOver3 / (dfY * dfY);
            for (int j = jMin + 1; j <= jMax; ++j) {
                dfY += dfYScale;
                const double dfNewSin = dfSin * k.cosPiYScale + dfCos * k.sinPiYScale;
                const double dfNewSinOver3 = dfSinOver3 * k.cosPiYScaleOver3 + dfCosOver3 * k.sinPiYScaleOver3;
                padfWeightsYShifted[j] = dfY == 0 ? 1.0 : THREE_PI_PI * dfNewSin * dfNewSinOver3 / (dfY * dfY);
                const double dfNewCos = dfCos * k.cosPiYScale - dfSin * k.sinPiYScale;
                const double dfNewCosOver3 = dfCosOver3 * k.cosPiYScaleOver3 - dfSinOver3 * k.sinPiYScaleOver3;
                dfSin = dfNewSin;
                dfCos = dfNewCos;
                dfSinOver3 = dfNewSinOver3;
                dfCosOver3 = dfNewCosOver3;
            }
            wk.lastSrcY = iSrcY;
            wk.lastDeltaY = dfDeltaY;
        }
    } else {
        while (jMin - dfDeltaY < -3.0) jMin++;
        while (jMax - dfDeltaY > 3.0) jMax--;
        if (iSrcY != wk.lastSrcY || dfDeltaY != wk.lastDeltaY) {
            const double dfSinPIDeltaYOver3 = std::sin((-kPi / 3.0) * dfDeltaY);
            const double dfSin2PIDeltaYOver3 = dfSinPIDeltaYOver3 * dfSinPIDeltaYOver3;
            const double dfCosPIDeltaYOver3 = std::sqrt(1.0 - dfSin2PIDeltaYOver3);
            const double dfSinPIDeltaY = (3.0 - 4.0 * dfSin2PIDeltaYOver3) * dfSinPIDeltaYOver3;
            const double dfInvPI2Over3 = 3.0 / (kPi * kPi);
            const double dfInvPI2Over3xSinPIDeltaY = dfInvPI2Over3 * dfSinPIDeltaY;
            const double dfInvPI2Over3xSinPIDeltaYxm0d5SinPIDeltaYOver3 =
                -0.5 * dfInvPI2Over3xSinPIDeltaY * dfSinPIDeltaYOver3;
            const double dfSinPIOver3 = 0.8660254037844386;
            const double dfInvPI2Over3xSinPIDeltaYxSinPIOver3xCosPIDeltaYOver3 =
                dfSinPIOver3 * dfInvPI2Over3xSinPIDeltaY * dfCosPIDeltaYOver3;
            const double padfCst[] = {dfInvPI2Over3xSinPIDeltaY * dfSinPIDeltaYOver3,
                                      dfInvPI2Over3xSinPIDeltaYxm0d5SinPIDeltaYOver3 -
                                          dfInvPI2Over3xSinPIDeltaYxSinPIOver3xCosPIDeltaYOver3,
                                      dfInvPI2Over3xSinPIDeltaYxm0d5SinPIDeltaYOver3 +
                                          dfInvPI2Over3xSinPIDeltaYxSinPIOver3xCosPIDeltaYOver3};
            for (int j = jMin; j <= jMax; ++j) {
                const double dfY = j - dfDeltaY;
                padfWeightsYShifted[j] = dfY == 0.0 ? 1.0 : padfCst[(j + 3) % 3] / (dfY * dfY);
            }
            wk.lastSrcY = iSrcY;
            wk.lastDeltaY = dfDeltaY;
        }
    }

    if (!masked) {
        double dfRowAccWeight = 0.0;
        for (int i = iMin; i <= iMax; ++i) dfRowAccWeight += padfWeightsXShifted[i];
        double dfColAccWeight = 0.0;
        for (int j = jMin; j <= jMax; ++j) dfColAccWeight += padfWeightsYShifted[j];
        dfAccumulatorWeight = dfRowAccWeight * dfColAccWeight;
    }

    if constexpr (std::is_same_v<T, std::uint8_t>) {
        if (!masked) {
            // The Byte case without masks (with GDAL's SSE2 sums of 6 columns).
            if (dfAccumulatorWeight < 0.000001) {
                *pdfDensity = 0.0;
                return false;
            }
            if (iMax - iMin + 1 == 6) {
                const double w0 = padfWeightsXShifted[iMin], w1 = padfWeightsXShifted[iMin + 1];
                const double w2 = padfWeightsXShifted[iMin + 2], w3 = padfWeightsXShifted[iMin + 3];
                const double w4 = padfWeightsXShifted[iMin + 4], w5 = padfWeightsXShifted[iMin + 5];
                auto row_acc = [&](int r) {
                    const int c = iSrcX + iMin;
                    const double p0 = s.raw(band, r, c) * w0, p1 = s.raw(band, r, c + 1) * w1;
                    const double p2 = s.raw(band, r, c + 2) * w2, p3 = s.raw(band, r, c + 3) * w3;
                    const double dfRowAcc = (p0 + p2) + (p1 + p3);
                    const double dfRowAccEnd = s.raw(band, r, c + 4) * w4 + s.raw(band, r, c + 5) * w5;
                    return dfRowAcc + dfRowAccEnd;
                };
                int j = jMin;
                for (; j < jMax; j += 2) {
                    dfAccumulatorReal += row_acc(iSrcY + j) * padfWeightsYShifted[j];
                    dfAccumulatorReal += row_acc(iSrcY + j + 1) * padfWeightsYShifted[j + 1];
                }
                if (j == jMax) dfAccumulatorReal += row_acc(iSrcY + j) * padfWeightsYShifted[j];
            } else {
                for (int j = jMin; j <= jMax; ++j) {
                    const int r = iSrcY + j;
                    int i = iMin;
                    double dfRowAcc1 = 0.0, dfRowAcc2 = 0.0;
                    for (; i < iMax; i += 2) {
                        dfRowAcc1 += s.raw(band, r, iSrcX + i) * padfWeightsXShifted[i];
                        dfRowAcc2 += s.raw(band, r, iSrcX + i + 1) * padfWeightsXShifted[i + 1];
                    }
                    if (i == iMax) dfRowAcc1 += s.raw(band, r, iSrcX + i) * padfWeightsXShifted[i];
                    dfAccumulatorReal += (dfRowAcc1 + dfRowAcc2) * padfWeightsYShifted[j];
                }
            }
            if (dfAccumulatorWeight < 0.99999 || dfAccumulatorWeight > 1.00001) {
                const double dfInvAcc = 1.0 / dfAccumulatorWeight;
                *pdfReal = dfAccumulatorReal * dfInvAcc;
            } else {
                *pdfReal = dfAccumulatorReal;
            }
            *pdfDensity = 1.0;
            return true;
        }
    }

    int nCountValid = 0;
    for (int j = jMin; j <= jMax; ++j) {
        const int r = iSrcY + j;
        const double dfWeight1 = padfWeightsYShifted[j];
        if (masked) {
            for (int i = iMin; i <= iMax; ++i) {
                if (!s.ok(r, iSrcX + i)) continue;
                nCountValid++;
                const double dfWeight2 = dfWeight1 * padfWeightsXShifted[i];
                dfAccumulatorReal += s.val(band, r, iSrcX + i) * dfWeight2;
                dfAccumulatorDensity += 1.0 * dfWeight2;
                dfAccumulatorWeight += dfWeight2;
            }
        } else {
            double dfRowAccReal = 0.0;
            for (int i = iMin; i <= iMax; ++i) dfRowAccReal += s.val(band, r, iSrcX + i) * padfWeightsXShifted[i];
            dfAccumulatorReal += dfRowAccReal * dfWeight1;
        }
    }

    if (dfAccumulatorWeight < 0.000001 ||
        (masked && (dfAccumulatorDensity < 0.000001 || nCountValid < (jMax - jMin + 1) * (iMax - iMin + 1) / 2))) {
        *pdfDensity = 0.0;
        return false;
    }
    if (dfAccumulatorWeight < 0.99999 || dfAccumulatorWeight > 1.00001) {
        const double dfInvAcc = 1.0 / dfAccumulatorWeight;
        *pdfReal = dfAccumulatorReal * dfInvAcc;
        *pdfDensity = masked ? dfAccumulatorDensity * dfInvAcc : 1.0;
    } else {
        *pdfReal = dfAccumulatorReal;
        *pdfDensity = masked ? dfAccumulatorDensity : 1.0;
    }
    return true;
}

// The destination being written: GWKSetPixelValueReal and the NoData
// avoidance of one or several bands.
template <class T>
struct Dst {
    T* data = nullptr;
    i64 plane = 0;
    int bands = 1;
    double nodata = 0.0;

    // The cells of row jr without a value yet: NoData.
    void clear_row(int jr, int w) const {
        const T fill = static_cast<T>(nodata);
        for (int b = 0; b < bands; ++b) std::fill_n(data + b * plane + static_cast<i64>(jr) * w, w, fill);
    }

    void set(int band, i64 at, double dfDensity, double dfReal) const {
        T& out = data[band * plane + at];
        if (dfDensity < 0.9999) {
            if (dfDensity < 0.0001) return;
            // The destination cell has no value yet (density 0).
            const double dfDstReal = static_cast<double>(out);
            const double dfDstInfluence = (1.0 - dfDensity) * 0.0;
            dfReal = (dfReal * dfDensity + dfDstReal * dfDstInfluence) / (dfDensity + dfDstInfluence);
        }
        out = clamp_round<T>(dfReal);
        if (bands == 1 && nodata == static_cast<double>(out)) avoid_nodata(out);
    }

    void set_value(int band, i64 at, T value) const {
        T& out = data[band * plane + at];
        out = value;
        if (bands == 1 && nodata == static_cast<double>(out)) avoid_nodata(out);
    }

    // GWKAvoidNoDataMultiBand.
    void avoid_multi(i64 at) const {
        if (bands == 1) return;
        for (int b = 0; b < bands; ++b) {
            if (nodata != static_cast<double>(data[b * plane + at])) return;
        }
        for (int b = 0; b < bands; ++b) avoid_nodata(data[b * plane + at]);
    }
};

// GWKOneSourceCornerFailsToReproject: a corner of the source does not go to
// the destination (with the transformer only).
template <class T>
bool corner_fails(const Src<T>& s, const WarpCoords& coords) {
    if (coords.transformer == nullptr) return false;
    for (int iY = 0; iY <= 1; ++iY) {
        for (int iX = 0; iX <= 1; ++iX) {
            double x = static_cast<double>(iX) * s.w, y = static_cast<double>(iY) * s.h;
            int ok = 0;
            coords.transform(false, 1, &x, &y, &ok);
            if (!ok) return true;
        }
    }
    return false;
}

// GWKAdjustSrcOffsetOnEdge: an invalid cell at the edge of the projection's
// domain (a corner of it does not go to the destination) takes a valid
// neighbour.
template <class T>
bool adjust_on_edge(const Src<T>& s, const WarpCoords& coords, int& isx, int& isy) {
    int ok = 0;
    double x = isx, y = isy;
    coords.transform(false, 1, &x, &y, &ok);
    if (ok) {
        x = isx;
        y = isy + 1;
        coords.transform(false, 1, &x, &y, &ok);
    }
    if (ok) {
        x = isx + 1;
        y = isy;
        coords.transform(false, 1, &x, &y, &ok);
    }
    if (ok) return false;
    if (isx + 1 < s.w && s.ok(isy, isx + 1)) {
        isx++;
        return true;
    }
    if (isy + 1 < s.h && s.ok(isy + 1, isx)) {
        isy++;
        return true;
    }
    if (isx > 0 && s.ok(isy, isx - 1)) {
        isx--;
        return true;
    }
    if (isy > 0 && s.ok(isy - 1, isx)) {
        isy--;
        return true;
    }
    return false;
}

// GWKNearest / GWKRealCase: the methods that sample at cell centres, one
// loop per method.
// With per_band (ps, pd: one source and destination of a band each), every
// band is tested and written on its own.
template <class T, int M>
void warp_points_loop(const Src<T>& s, const Dst<T>& d, const Src<T>* ps, const Dst<T>* pd, const WarpCoords& coords,
                      const WarpSpec& sp, const Kernel& k) {
    // GWKNearestThread<T> copies the cell (Byte, Int8, (U)Int16, Float32);
    // GWKRealCase goes through a double, which only changes 64-bit integers.
    constexpr bool kCopy = !(std::is_same_v<T, std::int64_t> || std::is_same_v<T, std::uint64_t>);
    const bool one_cell = s.w == 1 || s.h == 1;
    const bool on_edge = s.masked && corner_fails(s, coords);
#pragma omp parallel
    {
        std::vector<double> x(static_cast<std::size_t>(sp.dst_w)), y(static_cast<std::size_t>(sp.dst_w));
        Work wk(k);
#pragma omp for schedule(dynamic, 4)
        for (int jr = 0; jr < sp.dst_h; ++jr) {
            d.clear_row(jr, sp.dst_w);
            coords.row(sp.dst_r0 + jr, sp.dst_c0, sp.dst_w, x.data(), y.data());
            for (int ir = 0; ir < sp.dst_w; ++ir) {
                const double sx = x[ir], sy = y[ir];
                // GWKCheckAndComputeSrcOffsets
                if (!(sx >= 0) || !(sy >= 0)) continue;                      // also NaN
                if (!(sx + 1e-10 <= s.w) || !(sy + 1e-10 <= s.h)) continue;  // also infinite
                int isx = static_cast<int>(sx + 1.0e-10);
                int isy = static_cast<int>(sy + 1.0e-10);
                if (isx == s.w) isx--;
                if (isy == s.h) isy--;
                if (!s.held(isy, isx)) continue;  // never with a window computed right
                if (ps == nullptr && !s.valid_at(s.at(isy, isx)) &&
                    !(on_edge && adjust_on_edge(s, coords, isx, isy))) {
                    continue;
                }

                const i64 at = static_cast<i64>(jr) * sp.dst_w + ir;
                bool found = false;
                for (int b = 0; b < sp.bands; ++b) {
                    const Src<T>& sb = ps != nullptr ? ps[b] : s;
                    const Dst<T>& db = pd != nullptr ? pd[b] : d;
                    const int bi = ps != nullptr ? 0 : b;
                    int bx = isx, by = isy;
                    if (ps != nullptr && !sb.valid_at(sb.at(by, bx)) &&
                        !(on_edge && adjust_on_edge(sb, coords, bx, by))) {
                        continue;
                    }
                    double dens = 0.0, value = 0.0;
                    if (M == WARP_NEAREST || one_cell) {
                        if constexpr (kCopy && M == WARP_NEAREST) {
                            db.set_value(bi, at, sb.raw(bi, by, bx));
                            found = true;
                            continue;
                        }
                        value = sb.val(bi, by, bx);
                        dens = 1.0;
                    } else if constexpr (M == WARP_BILINEAR) {
                        if (k.use4) bilinear4(sb, bi, sx, sy, &dens, &value);
                        else resample(sb, k, wk, bi, sx, sy, &dens, &value);
                    } else if constexpr (M == WARP_CUBIC) {
                        if (k.use4) cubic4(sb, bi, sx, sy, &dens, &value);
                        else resample(sb, k, wk, bi, sx, sy, &dens, &value);
                    } else if constexpr (M == WARP_LANCZOS) {
                        lanczos(sb, k, wk, bi, sx, sy, &dens, &value);
                    } else {
                        resample(sb, k, wk, bi, sx, sy, &dens, &value);
                    }
                    if (dens < kBandDensityThreshold) continue;
                    found = true;
                    db.set(bi, at, dens, value);
                }
                if (found && ps == nullptr) d.avoid_multi(at);
            }
        }
    }
}

template <class T>
void warp_points(const Src<T>& s, const Dst<T>& d, const Src<T>* ps, const Dst<T>* pd, const WarpCoords& coords,
                 const WarpSpec& sp) {
    const Kernel k(sp.method, sp.xscale, sp.yscale);
    switch (sp.method) {
        case WARP_NEAREST: return warp_points_loop<T, WARP_NEAREST>(s, d, ps, pd, coords, sp, k);
        case WARP_BILINEAR: return warp_points_loop<T, WARP_BILINEAR>(s, d, ps, pd, coords, sp, k);
        case WARP_CUBIC: return warp_points_loop<T, WARP_CUBIC>(s, d, ps, pd, coords, sp, k);
        case WARP_CUBICSPLINE: return warp_points_loop<T, WARP_CUBICSPLINE>(s, d, ps, pd, coords, sp, k);
        default: return warp_points_loop<T, WARP_LANCZOS>(s, d, ps, pd, coords, sp, k);
    }
}

// GWKAverageOrModeComputeSourceCoords: the source cells under destination
// cell (i, j), from its corners; across the antimeridian (wrap) the columns
// go on past the source's last one, back to its first.
struct Footprint {
    double dfXMin, dfYMin, dfXMax, dfYMax;
    int iSrcXMin, iSrcYMin, iSrcXMax, iSrcYMax;
    bool wrap;
};

bool footprint(double x1, double y1, double x2, double y2, int w, int h, int nXMargin, int nYMargin, double xscale,
               const WarpCoords& coords, i64 i, i64 j, Footprint& f) {
    if (!(x1 >= -nXMargin && x2 >= -nXMargin && y1 >= -nYMargin && y2 >= -nYMargin && x1 - w <= nXMargin &&
          x2 - w <= nXMargin && y1 - h <= nYMargin && y2 - h <= nYMargin)) {
        return false;
    }
    if (x1 > x2) std::swap(x1, x2);
    f.wrap = false;
    const int threshold = std::min(2, w / 10);
    if (coords.transformer != nullptr && x1 * xscale < threshold && (w - x2) * xscale < threshold) {
        // The cell spans the whole source: across the antimeridian if its middle is left of it.
        double x = static_cast<double>(i) + 0.5, y = static_cast<double>(j);
        int ok = 0;
        coords.transform(true, 1, &x, &y, &ok);
        if (ok && x < x1) {
            f.wrap = true;
            std::swap(x1, x2);
            x2 += w;
        }
    }
    f.dfXMin = x1;
    f.dfXMax = x2;
    constexpr double EPSILON = 1e-10;
    if (!(f.dfXMax > -EPSILON && f.dfXMin < w + EPSILON)) return false;
    // (the margins keep the coordinates in the range of int)
    f.iSrcXMin = std::max(ifloor(f.dfXMin + EPSILON), 0);
    f.iSrcXMax = iceil(f.dfXMax - EPSILON);
    if (!f.wrap) f.iSrcXMax = std::min(f.iSrcXMax, w);
    if (f.iSrcXMin == f.iSrcXMax && f.iSrcXMax < w) f.iSrcXMax++;
    if (y1 > y2) std::swap(y1, y2);
    f.dfYMin = y1;
    f.dfYMax = y2;
    if (!(f.dfYMax > -EPSILON && f.dfYMin < h + EPSILON)) return false;
    f.iSrcYMin = std::max(ifloor(f.dfYMin + EPSILON), 0);
    f.iSrcYMax = std::min(iceil(f.dfYMax - EPSILON), h);
    if (f.iSrcYMin == f.iSrcYMax && f.iSrcYMax < h) f.iSrcYMax++;
    return true;
}

// COMPUTE_WEIGHT_Y and COMPUTE_WEIGHT.
inline double weight_y(const Footprint& f, int iSrcY) {
    return (iSrcY == f.iSrcYMin) ? ((f.iSrcYMin + 1 == f.iSrcYMax) ? 1.0 : 1 - (f.dfYMin - f.iSrcYMin))
           : (iSrcY + 1 == f.iSrcYMax) ? 1 - (f.iSrcYMax - f.dfYMax)
                                       : 1.0;
}

inline double weight(const Footprint& f, int iSrcX, double dfWeightY) {
    return (iSrcX == f.iSrcXMin) ? ((f.iSrcXMin + 1 == f.iSrcXMax) ? dfWeightY
                                                                    : dfWeightY * (1 - (f.dfXMin - f.iSrcXMin)))
           : (iSrcX + 1 == f.iSrcXMax) ? dfWeightY * (1 - (f.iSrcXMax - f.dfXMax))
                                       : dfWeightY;
}

template <class T>
constexpr bool kHistogramMode = std::is_same_v<T, std::uint8_t> || std::is_same_v<T, std::int8_t> ||
                                std::is_same_v<T, std::uint16_t> || std::is_same_v<T, std::int16_t>;

// GWKAverageOrModeThread and GWKModeRealType: the methods that take the
// source cells a destination cell covers (per_band: as warp_points_loop).
template <class T, int M>
void warp_areas_loop(const Src<T>& s, const Dst<T>& d, const Src<T>* ps, const Dst<T>* pd, const WarpCoords& coords,
                     const WarpSpec& sp) {
    constexpr int method = M;
    const int nXMargin = 2 * std::max(1, static_cast<int>(std::ceil(1. / sp.xscale)));
    const int nYMargin = 2 * std::max(1, static_cast<int>(std::ceil(1. / sp.yscale)));
    float quant = 0.0f;
    if (method == WARP_MED) quant = 0.5f;
    else if (method == WARP_Q1) quant = 0.25f;
    else if (method == WARP_Q3) quant = 0.75f;
    int nBins = 0, nBinsOffset = 0;
    if constexpr (kHistogramMode<T>) {
        nBins = sizeof(T) == 1 ? 256 : 65536;
        nBinsOffset = std::is_signed_v<T> ? nBins / 2 : 0;
    }

#pragma omp parallel
    {
        const std::size_t n = static_cast<std::size_t>(sp.dst_w) + 1;
        std::vector<double> x0(n), y0(n), x1(n), y1(n);
        std::vector<float> counts(method == WARP_MODE ? static_cast<std::size_t>(nBins) : 0);
        std::vector<int> touched;
        std::vector<T> modeVals;
        std::vector<float> modeCounts;
        std::vector<double> values;
        int last = -2;  // the row whose lower corners x1, y1 hold
#pragma omp for schedule(dynamic, 8)
        for (int jr = 0; jr < sp.dst_h; ++jr) {
            d.clear_row(jr, sp.dst_w);
            const i64 j = sp.dst_r0 + jr;
            if (jr == last + 1) {  // its upper corners: the lower ones of the row before
                std::swap(x0, x1);
                std::swap(y0, y1);
            } else {
                coords.row(j, sp.dst_c0, sp.dst_w + 1, x0.data(), y0.data());
            }
            coords.row(j + 1, sp.dst_c0, sp.dst_w + 1, x1.data(), y1.data());
            last = jr;
            for (int ir = 0; ir < sp.dst_w; ++ir) {
                Footprint f;
                if (!footprint(x0[ir], y0[ir], x1[ir + 1], y1[ir + 1], s.w, s.h, nXMargin, nYMargin, sp.xscale, coords,
                               sp.dst_c0 + ir, j, f)) {
                    continue;
                }
                const i64 at = static_cast<i64>(jr) * sp.dst_w + ir;
                bool found = false;
                const bool held = f.iSrcXMin < f.iSrcXMax && f.iSrcYMin < f.iSrcYMax &&
                                  s.held(f.iSrcYMin, f.iSrcXMin) && s.held(f.iSrcYMax - 1, f.iSrcXMax - 1);
                for (int b = 0; b < sp.bands; ++b) {
                    const Src<T>& sb = ps != nullptr ? ps[b] : s;
                    const Dst<T>& db = pd != nullptr ? pd[b] : d;
                    const int bi = ps != nullptr ? 0 : b;
                    auto valid = [&](int r, int c) { return held ? sb.valid_at(sb.at(r, c)) : sb.ok(r, c); };
                    // the source column of footprint column c
                    auto col = [&](int c) { return f.wrap ? c % s.w : c; };
                    if constexpr (method == WARP_AVERAGE) {
                        double dfTotalWeight = 0.0, dfValueReal = 0.0;
                        for (int r = f.iSrcYMin; r < f.iSrcYMax; r++) {
                            const double dfWeightY = weight_y(f, r);
                            for (int c = f.iSrcXMin; c < f.iSrcXMax; c++) {
                                const int cc = col(c);
                                if (!valid(r, cc)) continue;
                                const double dfWeight = weight(f, c, dfWeightY);
                                if (dfWeight > 0) {
                                    dfTotalWeight += dfWeight;
                                    dfValueReal += (dfWeight / dfTotalWeight) * (sb.val(bi, r, cc) - dfValueReal);
                                }
                            }
                        }
                        if (dfTotalWeight > 0) {
                            found = true;
                            db.set(bi, at, 1.0, dfValueReal);
                        }
                    } else if constexpr (method == WARP_RMS) {
                        double dfTotalReal = 0.0, dfTotalWeight = 0.0;
                        for (int r = f.iSrcYMin; r < f.iSrcYMax; r++) {
                            const double dfWeightY = weight_y(f, r);
                            for (int c = f.iSrcXMin; c < f.iSrcXMax; c++) {
                                const int cc = col(c);
                                if (!valid(r, cc)) continue;
                                const double v = sb.val(bi, r, cc);
                                const double dfWeight = weight(f, c, dfWeightY);
                                dfTotalWeight += dfWeight;
                                dfTotalReal += v * v * dfWeight;
                            }
                        }
                        if (dfTotalWeight > 0) {
                            found = true;
                            db.set(bi, at, 1.0, std::sqrt(dfTotalReal / dfTotalWeight));
                        }
                    } else if constexpr (method == WARP_MODE) {
                        if constexpr (kHistogramMode<T>) {
                            float fMaxCount = 0.0f;
                            int nMode = -1;
                            bool bHasSourceValues = false;
                            for (int r = f.iSrcYMin; r < f.iSrcYMax; r++) {
                                const double dfWeightY = weight_y(f, r);
                                for (int c = f.iSrcXMin; c < f.iSrcXMax; c++) {
                                    const int cc = col(c);
                                    if (!valid(r, cc)) continue;
                                    bHasSourceValues = true;
                                    const int nVal = static_cast<int>(sb.val(bi, r, cc));
                                    const int iBin = nVal + nBinsOffset;
                                    const double dfWeight = weight(f, c, dfWeightY);
                                    float& count = counts[static_cast<std::size_t>(iBin)];
                                    if (count == 0.0f) touched.push_back(iBin);
                                    count += static_cast<float>(dfWeight);
                                    if (count > fMaxCount) {  // ties: the first (GWKTS_First)
                                        nMode = nVal;
                                        fMaxCount = count;
                                    }
                                }
                            }
                            for (int t : touched) counts[static_cast<std::size_t>(t)] = 0.0f;
                            touched.clear();
                            if (bHasSourceValues) {
                                found = true;
                                db.set(bi, at, 1.0, static_cast<double>(nMode));
                            }
                        } else {
                            // GWKModeRealType: values in order of appearance.
                            modeVals.clear();
                            modeCounts.clear();
                            int iModeIndex = -1;
                            for (int r = f.iSrcYMin; r < f.iSrcYMax; r++) {
                                const double dfWeightY = weight_y(f, r);
                                for (int c = f.iSrcXMin; c < f.iSrcXMax; c++) {
                                    const int cc = col(c);
                                    if (!valid(r, cc)) continue;
                                    const T nVal = sb.raw(bi, r, cc);
                                    const double dfWeight = weight(f, c, dfWeightY);
                                    std::size_t i = 0;
                                    for (; i < modeVals.size(); ++i) {
                                        bool same = modeVals[i] == nVal;  // IsSame: NaN is NaN
                                        if constexpr (std::is_floating_point_v<T>) {
                                            same = same || (std::isnan(modeVals[i]) && std::isnan(nVal));
                                        }
                                        if (same) {
                                            modeCounts[i] += static_cast<float>(dfWeight);
                                            if (modeCounts[i] > modeCounts[static_cast<std::size_t>(iModeIndex)]) {
                                                iModeIndex = static_cast<int>(i);
                                            }
                                            break;
                                        }
                                    }
                                    if (i == modeVals.size()) {
                                        modeVals.push_back(nVal);
                                        modeCounts.push_back(static_cast<float>(dfWeight));
                                        if (iModeIndex < 0) iModeIndex = static_cast<int>(i);
                                    }
                                }
                            }
                            if (iModeIndex != -1) {
                                found = true;
                                db.set_value(bi, at, modeVals[static_cast<std::size_t>(iModeIndex)]);
                            }
                        }
                    } else if constexpr (method == WARP_MAX || method == WARP_MIN) {
                        bool bFoundValid = false;
                        double dfTotalReal = method == WARP_MAX ? std::numeric_limits<double>::lowest()
                                                                : std::numeric_limits<double>::max();
                        for (int r = f.iSrcYMin; r < f.iSrcYMax; r++) {
                            for (int c = f.iSrcXMin; c < f.iSrcXMax; c++) {
                                const int cc = col(c);
                                if (!valid(r, cc)) continue;
                                const double v = sb.val(bi, r, cc);
                                bFoundValid = true;
                                if (method == WARP_MAX ? dfTotalReal < v : dfTotalReal > v) dfTotalReal = v;
                            }
                        }
                        if (bFoundValid) {
                            found = true;
                            db.set(bi, at, 1.0, dfTotalReal);
                        }
                    } else {  // med, q1, q3
                        values.clear();
                        for (int r = f.iSrcYMin; r < f.iSrcYMax; r++) {
                            for (int c = f.iSrcXMin; c < f.iSrcXMax; c++) {
                                const int cc = col(c);
                                if (valid(r, cc)) values.push_back(sb.val(bi, r, cc));
                            }
                        }
                        if (!values.empty()) {
                            std::sort(values.begin(), values.end());
                            const float pos = quant * static_cast<float>(values.size()) - 1;
                            const int quantIdx = static_cast<int>(std::ceil(pos));
                            found = true;
                            db.set(bi, at, 1.0, values[static_cast<std::size_t>(quantIdx)]);
                        }
                    }
                }
                if (found && ps == nullptr) d.avoid_multi(at);
            }
        }
    }
}

template <class T>
void warp_areas(const Src<T>& s, const Dst<T>& d, const Src<T>* ps, const Dst<T>* pd, const WarpCoords& coords,
                const WarpSpec& sp) {
    switch (sp.method) {
        case WARP_AVERAGE: return warp_areas_loop<T, WARP_AVERAGE>(s, d, ps, pd, coords, sp);
        case WARP_RMS: return warp_areas_loop<T, WARP_RMS>(s, d, ps, pd, coords, sp);
        case WARP_MODE: return warp_areas_loop<T, WARP_MODE>(s, d, ps, pd, coords, sp);
        case WARP_MAX: return warp_areas_loop<T, WARP_MAX>(s, d, ps, pd, coords, sp);
        case WARP_MIN: return warp_areas_loop<T, WARP_MIN>(s, d, ps, pd, coords, sp);
        case WARP_MED: return warp_areas_loop<T, WARP_MED>(s, d, ps, pd, coords, sp);
        case WARP_Q1: return warp_areas_loop<T, WARP_Q1>(s, d, ps, pd, coords, sp);
        default: return warp_areas_loop<T, WARP_Q3>(s, d, ps, pd, coords, sp);
    }
}

}  // namespace

template <class T>
void warp(const T* src, T* dst, const WarpCoords& coords, const WarpSpec& sp) {
    const i64 dst_plane = static_cast<i64>(sp.dst_h) * sp.dst_w;

    Src<T> s;
    s.data = src;
    s.plane = static_cast<i64>(sp.win_h) * sp.win_w;
    s.h = sp.src_h;
    s.w = sp.src_w;
    s.r0 = sp.win_r0;
    s.c0 = sp.win_c0;
    s.wh = sp.win_h;
    s.ww = sp.win_w;
    // GDAL's unified validity mask (a cell is valid when any band is not
    // NoData). GDAL drops it when its window happens to hold no NoData; here
    // it is kept whenever there is a NoData value, so that every window gives
    // the same values (only the Lanczos sums differ, in the last bits).
    std::vector<std::uint8_t> valid;
    s.nodata = NoDataTest<T>(sp.src_nodata);
    s.masked = sp.has_src_nodata && s.nodata.active;
    if (s.masked && sp.bands > 1 && !sp.per_band) {
        valid.assign(static_cast<std::size_t>(s.plane), 1);
        const i64 plane = s.plane;
        const int bands = sp.bands;
        const NoDataTest<T> is_nodata = s.nodata;
#pragma omp parallel for schedule(static)
        for (i64 i = 0; i < plane; ++i) {
            bool any = false;
            for (int b = 0; b < bands && !any; ++b) any = !is_nodata(src[b * plane + i]);
            valid[static_cast<std::size_t>(i)] = any ? 1 : 0;
        }
        s.valid = valid.data();
    }

    Dst<T> d;
    d.data = dst;
    d.plane = dst_plane;
    d.bands = sp.bands;
    d.nodata = sp.dst_nodata;
    // per_band: a source and a destination of one band each, the band's
    // values testing its own validity (GDAL warping the band alone).
    std::vector<Src<T>> ps;
    std::vector<Dst<T>> pd;
    if (sp.per_band && sp.bands > 1) {
        ps.assign(static_cast<std::size_t>(sp.bands), s);
        pd.assign(static_cast<std::size_t>(sp.bands), d);
        for (int b = 0; b < sp.bands; ++b) {
            ps[static_cast<std::size_t>(b)].data = src + b * s.plane;
            pd[static_cast<std::size_t>(b)].data = dst + b * dst_plane;
            pd[static_cast<std::size_t>(b)].bands = 1;
        }
    }
    const Src<T>* pps = ps.empty() ? nullptr : ps.data();
    const Dst<T>* ppd = pd.empty() ? nullptr : pd.data();
    if (warp_by_area(sp.method)) warp_areas(s, d, pps, ppd, coords, sp);
    else warp_points(s, d, pps, ppd, coords, sp);
}

template void warp<std::uint8_t>(const std::uint8_t*, std::uint8_t*, const WarpCoords&, const WarpSpec&);
template void warp<std::int8_t>(const std::int8_t*, std::int8_t*, const WarpCoords&, const WarpSpec&);
template void warp<std::uint16_t>(const std::uint16_t*, std::uint16_t*, const WarpCoords&, const WarpSpec&);
template void warp<std::int16_t>(const std::int16_t*, std::int16_t*, const WarpCoords&, const WarpSpec&);
template void warp<std::uint32_t>(const std::uint32_t*, std::uint32_t*, const WarpCoords&, const WarpSpec&);
template void warp<std::int32_t>(const std::int32_t*, std::int32_t*, const WarpCoords&, const WarpSpec&);
template void warp<std::uint64_t>(const std::uint64_t*, std::uint64_t*, const WarpCoords&, const WarpSpec&);
template void warp<std::int64_t>(const std::int64_t*, std::int64_t*, const WarpCoords&, const WarpSpec&);
template void warp<float>(const float*, float*, const WarpCoords&, const WarpSpec&);
template void warp<double>(const double*, double*, const WarpCoords&, const WarpSpec&);

}  // namespace zeit::warp
