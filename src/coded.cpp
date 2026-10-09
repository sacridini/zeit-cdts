#include "coded.h"

#ifdef _OPENMP
#include <omp.h>
#else
#define omp_get_max_threads() 1
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <Eigen/Dense>

namespace zeit {
namespace coded {

namespace {

constexpr double PI = 3.14159265358979323846;
const double NaN = std::numeric_limits<double>::quiet_NaN();

struct Model {
    double c = NaN, a = NaN, b = NaN, rmse = NaN;
    int n = 0;
    bool ok() const { return n > 0; }
    double predict(double t) const { return c + a * std::sin(2 * PI * t) + b * std::cos(2 * PI * t); }
};

// OLS of y = c + a sin(2 pi t) + b cos(2 pi t) on the observations idx (finite y).
Model fit(const std::vector<double>& t, const double* y, const std::vector<int>& idx, int min_obs) {
    Model m;
    std::vector<int> use;
    for (int i : idx) if (std::isfinite(y[i])) use.push_back(i);
    if (static_cast<int>(use.size()) < std::max(min_obs, 4)) return m;
    Eigen::MatrixXd X(use.size(), 3);
    Eigen::VectorXd Y(use.size());
    for (size_t r = 0; r < use.size(); ++r) {
        const double tt = t[use[r]];
        X(r, 0) = 1.0; X(r, 1) = std::sin(2 * PI * tt); X(r, 2) = std::cos(2 * PI * tt);
        Y(r) = y[use[r]];
    }
    Eigen::Vector3d beta = X.colPivHouseholderQr().solve(Y);
    m.c = beta(0); m.a = beta(1); m.b = beta(2);
    double sse = 0.0;
    for (size_t r = 0; r < use.size(); ++r) {
        const double res = Y(r) - X.row(r).dot(beta);
        sse += res * res;
    }
    m.rmse = std::sqrt(sse / static_cast<double>(use.size()));
    m.n = static_cast<int>(use.size());
    return m;
}

std::vector<int> window(const std::vector<double>& t, double from, double to) {
    std::vector<int> idx;
    for (int i = 0; i < static_cast<int>(t.size()); ++i) if (t[i] >= from && t[i] < to) idx.push_back(i);
    return idx;
}

void put_model(double* out, const Model& m) { out[0] = m.c; out[1] = m.a; out[2] = m.b; }

// The models of the features on the same observations as NDFI (its finite ones).
void put_features(double* out, const std::vector<double>& t, const double* ndfi, const double* features,
                  int n_features, int n_time, const std::vector<int>& idx) {
    std::vector<int> use;
    for (int i : idx) if (std::isfinite(ndfi[i])) use.push_back(i);
    for (int f = 0; f < n_features; ++f) {
        Model m = fit(t, features + static_cast<size_t>(f) * n_time, use, 1);
        put_model(out + 3 * f, m);
    }
}

} // namespace

int coded_size(int n_features, int max_events) {
    const int header = 3 + 1 + 1 + 3 * n_features + 1;
    const int event = 3 + 3 + 1 + 3 * n_features;
    return header + max_events * event;
}

std::vector<double> coded_pixel(const std::vector<double>& t, const std::vector<double>& ndfi,
                                const std::vector<double>& features, int n_features, const Params& p) {
    const int n = static_cast<int>(t.size());
    if (static_cast<int>(ndfi.size()) != n || static_cast<int>(features.size()) != n_features * n)
        throw std::runtime_error("coded: ndfi must have one value per date, features n_features per date");
    for (int i = 1; i < n; ++i)
        if (!(t[i] >= t[i - 1])) throw std::runtime_error("coded: the dates must be sorted");
    std::vector<double> out(coded_size(n_features, p.max_events), NaN);
    const int header = 3 + 1 + 1 + 3 * n_features + 1;
    const int per_event = 3 + 3 + 1 + 3 * n_features;
    const double* y = ndfi.data();
    const double* feats = features.data();

    const double train_end = p.train_start + p.train_years;
    std::vector<int> train = window(t, p.train_start, train_end);
    Model model = fit(t, y, train, p.min_obs);
    out[header - 1] = 0.0;   // number of changes
    if (!model.ok()) return out;
    put_model(out.data(), model);
    out[3] = model.rmse;
    out[4] = model.n;
    put_features(out.data() + 5, t, y, feats, n_features, n, train);

    int events = 0, run = 0, first = -1, last_before = -1, previous_valid = -1;
    double sum = 0.0;
    int i = 0;
    while (i < n && t[i] < train_end) { if (std::isfinite(y[i])) previous_valid = i; ++i; }
    for (; i < n && events < p.max_events; ++i) {
        if (!std::isfinite(y[i])) continue;
        const double res = y[i] - model.predict(t[i]);
        const double z = res / std::max(model.rmse, 1e-12);
        const bool beyond = p.loss_only ? (z < -p.thresh) : (std::abs(z) > p.thresh);
        if (!beyond) {
            run = 0; sum = 0.0; previous_valid = i;
            continue;
        }
        if (run == 0) { first = i; last_before = previous_valid; }
        ++run;
        sum += res;
        previous_valid = i;
        if (run < p.consec) continue;

        // a change: date, the observation before it, magnitude
        double* ev = out.data() + header + events * per_event;
        ev[0] = t[first];
        ev[1] = last_before >= 0 ? t[last_before] : NaN;
        ev[2] = sum / run;
        ++events;
        out[header - 1] = events;
        // the land cover after it: a new model on the next min_years
        const double resume = t[first] + p.min_years;
        std::vector<int> after = window(t, t[first], resume);
        Model next = fit(t, y, after, p.min_obs);
        if (!next.ok()) break;
        put_model(ev + 3, next);
        ev[6] = next.rmse;
        put_features(ev + 7, t, y, feats, n_features, n, after);
        model = next;
        run = 0; sum = 0.0; first = -1;
        previous_valid = -1;
        while (i + 1 < n && t[i + 1] < resume) { ++i; if (std::isfinite(y[i])) previous_valid = i; }
    }
    return out;
}

pybind11::array_t<double> coded_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> t,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> ndfi,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> features,
    const Params& params, int n_jobs) {
    auto tv = t.unchecked<1>();
    auto nd = ndfi.unchecked<2>();
    auto fe = features.unchecked<3>();
    const int n_time = static_cast<int>(tv.shape(0));
    const int pixels = static_cast<int>(nd.shape(0));
    const int n_features = static_cast<int>(fe.shape(1));
    if (static_cast<int>(nd.shape(1)) != n_time || static_cast<int>(fe.shape(0)) != pixels
        || static_cast<int>(fe.shape(2)) != n_time)
        throw std::runtime_error("coded: expected t (time), ndfi (pixels, time), features (pixels, n, time)");
    std::vector<double> times(t.data(), t.data() + n_time);
    const int size = coded_size(n_features, params.max_events);
    pybind11::array_t<double> out_arr({pixels, size});
    double* out = out_arr.mutable_data();
    const double* y = ndfi.data();
    const double* f = features.data();
    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);
    std::string error;
    {
        pybind11::gil_scoped_release release;
        #pragma omp parallel for num_threads(n_jobs) schedule(dynamic, 64)
        for (int px = 0; px < pixels; ++px) {
            std::vector<double> yy(y + static_cast<size_t>(px) * n_time, y + static_cast<size_t>(px + 1) * n_time);
            std::vector<double> ff(f + static_cast<size_t>(px) * n_features * n_time,
                                   f + static_cast<size_t>(px + 1) * n_features * n_time);
            try {
                std::vector<double> r = coded_pixel(times, yy, ff, n_features, params);
                std::copy(r.begin(), r.end(), out + static_cast<size_t>(px) * size);
            } catch (const std::exception& e) {
                #pragma omp critical
                error = e.what();
            }
        }
    }
    if (!error.empty()) throw std::runtime_error(error);
    return out_arr;
}

} // namespace coded
} // namespace zeit
