#include "bfast_monitor.h"

#ifndef _OPENMP
#ifndef OMP_DUMMIES_DEFINED
#define OMP_DUMMIES_DEFINED
#define omp_get_max_threads() 1
#define omp_get_thread_num() 0
#define omp_set_num_threads(x) (void)(x)
#endif
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#include <cmath>
#include <algorithm>
#include <numeric>
#include <limits>
#include <stdexcept>
#include <Eigen/Dense>

namespace zeit {
namespace bfastmonitor {

namespace {

const double PI = 3.14159265358979323846;
const double E_CONST = 2.71828182845904523536; // avoid relying on the non-standard M_E macro

// ---- OLS-MOSUM monitoring critical value table -------------------------
// Ported from strucchangeRcpp's `monitorMECritvalTable` (Zeileis, Leisch,
// Kleiber & Hornik's pre-simulated boundary-crossing quantiles for the
// Chu-Stinchcombe-White (1996) monitoring process), functional="max" slice
// only - the only functional bfastmonitor's default type="OLS-MOSUM" uses.
// winsize grid: {0.25, 0.5, 1.0}; period grid: {2,4,6,8,10}; alpha grid:
// 0.95..0.999 step 0.001 (50 points).
static const double kWinsizeGrid[3] = {0.25, 0.5, 1.0};
static const int kPeriodGrid[5] = {2, 4, 6, 8, 10};
static const int kAlphaGridN = 50;

static const double kMonitorMECritval[3][5][50] = {
    { // winsize = 0.25
        {1.227626658, 1.230669709, 1.232764859, 1.235640555, 1.23847794, 1.241980766, 1.244816372, 1.248790191, 1.252395053, 1.254989075, 1.258229332, 1.262239048, 1.265965555, 1.269239982, 1.272640915, 1.276039923, 1.279592327, 1.284858632, 1.289183505, 1.293118898, 1.297743161, 1.302930273, 1.307095365, 1.311586245, 1.317376461, 1.323351663, 1.327863587, 1.333506967, 1.339192303, 1.344925516, 1.350272749, 1.356196769, 1.362590494, 1.369578943, 1.376457831, 1.384282334, 1.391945016, 1.401007986, 1.411669918, 1.42087735, 1.433262947, 1.443599502, 1.455112312, 1.471607019, 1.48885977, 1.507299645, 1.531755865, 1.560380548, 1.6045884, 1.673976765},
        {1.336231054, 1.33879059, 1.341468238, 1.34417862, 1.346585979, 1.349206694, 1.35202041, 1.354620262, 1.357196733, 1.360499977, 1.363725498, 1.366678838, 1.370524118, 1.373742123, 1.376576972, 1.380221287, 1.383943383, 1.387693334, 1.390483925, 1.394081893, 1.398436377, 1.40284081, 1.407343317, 1.411895724, 1.415970821, 1.420220268, 1.425240572, 1.430859209, 1.435553585, 1.440531219, 1.445439739, 1.450892021, 1.456028139, 1.462781812, 1.469854412, 1.476965897, 1.485231832, 1.492833514, 1.500951151, 1.510675083, 1.519836793, 1.532697402, 1.544403481, 1.5591062, 1.575720718, 1.596956044, 1.61812223, 1.649405243, 1.685943314, 1.745509489},
        {1.341086852, 1.343915838, 1.346241729, 1.348399177, 1.351095832, 1.353765114, 1.356148651, 1.358915303, 1.361947358, 1.365235742, 1.368276811, 1.371642752, 1.374604344, 1.377815266, 1.381152503, 1.38474993, 1.388012051, 1.390912945, 1.394646106, 1.398722491, 1.402654148, 1.40676218, 1.41145745, 1.415373728, 1.419369828, 1.423624595, 1.428682308, 1.433526361, 1.438251781, 1.44291381, 1.448123185, 1.453143331, 1.458898087, 1.46547805, 1.472349056, 1.480404305, 1.487426578, 1.494942706, 1.503252476, 1.511969729, 1.521599882, 1.533837645, 1.545380293, 1.560337768, 1.576581543, 1.597971114, 1.618396722, 1.649405243, 1.685943314, 1.745509489},
        {1.341656815, 1.344137521, 1.34645624, 1.348851916, 1.35155402, 1.353986176, 1.356344915, 1.359234814, 1.362264565, 1.365600269, 1.368504603, 1.372149261, 1.374777259, 1.378134327, 1.381547885, 1.385074234, 1.388367948, 1.391173487, 1.395081461, 1.39885994, 1.403074347, 1.407276113, 1.411639041, 1.415582144, 1.41956165, 1.423804335, 1.428849289, 1.433627525, 1.438392252, 1.442974156, 1.448228241, 1.453262303, 1.458991735, 1.465547489, 1.472452998, 1.480558744, 1.487521886, 1.495068196, 1.503442387, 1.511986496, 1.521628506, 1.534081675, 1.545546897, 1.560337768, 1.576581543, 1.597971114, 1.618396722, 1.649405243, 1.685943314, 1.745509489},
        {1.34182451, 1.344391315, 1.34660319, 1.349151166, 1.351785871, 1.35417886, 1.356683617, 1.359487399, 1.362568985, 1.365772312, 1.36886321, 1.37237427, 1.374851675, 1.378315356, 1.381751205, 1.38537793, 1.388473242, 1.391455971, 1.395329558, 1.39911228, 1.403187733, 1.407489544, 1.411813723, 1.415698402, 1.419777231, 1.42381862, 1.428956694, 1.433639364, 1.438405149, 1.443075762, 1.448236088, 1.453310844, 1.459029083, 1.46557817, 1.472530539, 1.480576361, 1.487592844, 1.495171018, 1.503842297, 1.512083905, 1.521644973, 1.534364548, 1.54556159, 1.560360629, 1.576732064, 1.597971114, 1.618396722, 1.649405243, 1.685943314, 1.745509489},
    },
    { // winsize = 0.5
        {1.687323283, 1.691600842, 1.696333894, 1.701584355, 1.705516895, 1.711072573, 1.716266061, 1.720193477, 1.725836516, 1.730801363, 1.736321524, 1.74196445, 1.747393777, 1.753500032, 1.758804665, 1.765472475, 1.772517878, 1.779402316, 1.788029014, 1.795241195, 1.802222752, 1.809157841, 1.81682673, 1.824857243, 1.833342652, 1.841864166, 1.851771211, 1.861210859, 1.869505044, 1.878586666, 1.886952528, 1.899271004, 1.909989941, 1.920444065, 1.933258819, 1.948464122, 1.961357195, 1.978307317, 1.993349334, 2.010535942, 2.031463394, 2.052895819, 2.073102416, 2.096963859, 2.121519516, 2.15191486, 2.200396702, 2.247113944, 2.308003939, 2.434575851},
        {1.886330901, 1.890237639, 1.895050553, 1.899687207, 1.903961048, 1.908307108, 1.912950187, 1.917516041, 1.92212903, 1.927966983, 1.933243246, 1.939180382, 1.945610384, 1.950990933, 1.957233508, 1.963209729, 1.969691087, 1.975367579, 1.981488337, 1.987604023, 1.99470655, 2.001483195, 2.009906178, 2.018008841, 2.026390466, 2.034021986, 2.042078546, 2.052014283, 2.058945903, 2.066094608, 2.075533463, 2.085648599, 2.095269089, 2.105089456, 2.114998586, 2.126446052, 2.1386016, 2.151764365, 2.165997217, 2.184522374, 2.201169959, 2.218109043, 2.241079711, 2.264122957, 2.290531942, 2.320519775, 2.355904306, 2.408981773, 2.464537075, 2.56886152},
        {1.899584451, 1.903722827, 1.907863498, 1.912100393, 1.916484979, 1.920912858, 1.926253682, 1.93110016, 1.936170831, 1.941870119, 1.947242786, 1.952110922, 1.957768136, 1.963490929, 1.969616084, 1.97448555, 1.980480618, 1.98572396, 1.992285245, 1.997985437, 2.004818758, 2.012325128, 2.019667635, 2.02771381, 2.035115163, 2.042662242, 2.052127057, 2.058427683, 2.065554826, 2.074048453, 2.082447516, 2.092158418, 2.101756462, 2.111554201, 2.121438219, 2.133841187, 2.144181894, 2.156616758, 2.173272005, 2.190797715, 2.208535165, 2.224368909, 2.246534249, 2.269144297, 2.294707725, 2.325254891, 2.358182469, 2.411867368, 2.465796784, 2.570255291},
        {1.90129851, 1.905165709, 1.909371751, 1.913716395, 1.917940561, 1.922321239, 1.927599118, 1.932475888, 1.937709641, 1.943303054, 1.948882928, 1.953539157, 1.959244341, 1.964870519, 1.970575776, 1.975609478, 1.981478028, 1.987006556, 1.993115911, 1.998916074, 2.006502821, 2.013256636, 2.020550829, 2.02869456, 2.035950019, 2.04423035, 2.053172928, 2.059144878, 2.066093646, 2.074707965, 2.082848247, 2.092532956, 2.10192808, 2.111633062, 2.121521342, 2.133935513, 2.144204319, 2.15693361, 2.17354493, 2.191140452, 2.208753774, 2.224911328, 2.246710424, 2.269487088, 2.295161596, 2.325521831, 2.359174289, 2.411867368, 2.465796784, 2.570255291},
        {1.902003179, 1.905759412, 1.910032248, 1.914301165, 1.918521192, 1.923639243, 1.92813013, 1.933183545, 1.938191815, 1.943723943, 1.949207274, 1.954109382, 1.959425746, 1.965069316, 1.970973694, 1.97593044, 1.98160705, 1.987355077, 1.993443035, 1.999078597, 2.006985314, 2.013485116, 2.020959059, 2.029366965, 2.036447662, 2.044387857, 2.053381454, 2.059376758, 2.066161719, 2.074737896, 2.082870276, 2.092568643, 2.101952022, 2.111780137, 2.121701709, 2.13419421, 2.144253049, 2.157614719, 2.173771247, 2.191610807, 2.209072828, 2.225383742, 2.24728555, 2.269782323, 2.295703187, 2.325521831, 2.359174289, 2.411867368, 2.465796784, 2.570255291},
    },
    { // winsize = 1.0
        {2.224088182, 2.231672178, 2.238555626, 2.246716229, 2.254954836, 2.263672475, 2.271981479, 2.280293746, 2.28997972, 2.301391508, 2.310397792, 2.320316491, 2.329219008, 2.339306324, 2.350632084, 2.362392553, 2.373681862, 2.383993334, 2.396559477, 2.409002892, 2.420513032, 2.431878001, 2.44213404, 2.455464542, 2.468239741, 2.483054311, 2.500142642, 2.51277454, 2.527290444, 2.544121264, 2.559929452, 2.579905459, 2.599544951, 2.62226086, 2.643281865, 2.667571212, 2.689126883, 2.716152735, 2.744999292, 2.769564191, 2.799615917, 2.84219703, 2.882627591, 2.923850318, 2.971623997, 3.029458199, 3.087042275, 3.17273589, 3.289425083, 3.454726813},
        {2.704436763, 2.713163975, 2.722307518, 2.730136285, 2.737834433, 2.745289624, 2.753501331, 2.761654792, 2.769950069, 2.777843501, 2.788012924, 2.796900035, 2.807824342, 2.817101311, 2.828102791, 2.8390293, 2.849856029, 2.859437411, 2.872062367, 2.88195698, 2.894229355, 2.905888012, 2.918303141, 2.928892411, 2.941650017, 2.955379687, 2.968287824, 2.983458155, 3.000885641, 3.014506132, 3.030217411, 3.045143052, 3.063189385, 3.082043849, 3.10118718, 3.121373454, 3.146293619, 3.169754121, 3.198711501, 3.225578048, 3.252829636, 3.29448456, 3.328245087, 3.368008513, 3.413465945, 3.460251344, 3.515428961, 3.606242618, 3.718164081, 3.935356962},
        {2.737148076, 2.743205139, 2.750226965, 2.757795346, 2.766094005, 2.7729254, 2.782580948, 2.789957358, 2.797965644, 2.808373675, 2.81622524, 2.825447769, 2.835817425, 2.846077952, 2.856533117, 2.866150611, 2.875077553, 2.885461201, 2.89647861, 2.906463701, 2.918303669, 2.927930492, 2.939940004, 2.951336164, 2.962908062, 2.976537911, 2.989568741, 3.002817145, 3.017674742, 3.031640324, 3.045733102, 3.062709328, 3.081056831, 3.099111816, 3.118645177, 3.143574407, 3.161755511, 3.188961433, 3.214095538, 3.237936304, 3.274006329, 3.309481606, 3.339912377, 3.382316715, 3.423838203, 3.473392644, 3.529341791, 3.61938626, 3.734347666, 3.941029161},
        {2.742879244, 2.749722828, 2.757492485, 2.765450677, 2.772398282, 2.780656384, 2.78832964, 2.795877815, 2.804612885, 2.813310809, 2.821322451, 2.831913078, 2.84091691, 2.851046357, 2.860616433, 2.871058135, 2.878643039, 2.889821139, 2.900193874, 2.911006285, 2.920887002, 2.931366935, 2.942372546, 2.955135021, 2.965145513, 2.979340361, 2.99268186, 3.007144474, 3.021514975, 3.033255324, 3.04844221, 3.065216511, 3.083965006, 3.102762762, 3.121286595, 3.145042068, 3.163725702, 3.192022966, 3.216679962, 3.240049663, 3.274859738, 3.310631241, 3.340922469, 3.383335996, 3.424968182, 3.474226545, 3.529363492, 3.620480509, 3.736920001, 3.941029161},
        {2.745927613, 2.753325774, 2.760330908, 2.767957464, 2.774493419, 2.783771505, 2.790409059, 2.797912698, 2.80812472, 2.815858763, 2.824270337, 2.834507818, 2.843434255, 2.853604285, 2.862432657, 2.872653837, 2.880941991, 2.891401968, 2.90133608, 2.912487406, 2.922340016, 2.933102018, 2.943661808, 2.955941746, 2.966897593, 2.980013964, 2.994808282, 3.008676618, 3.022463244, 3.033942185, 3.049288621, 3.065598476, 3.085387126, 3.103441213, 3.121690132, 3.145468149, 3.164096118, 3.193316423, 3.217121932, 3.2407932, 3.276932457, 3.31144207, 3.341216638, 3.384312582, 3.425138704, 3.474226545, 3.529363492, 3.620958819, 3.736979356, 3.941029161},
    },
};

// Linear interpolation matching R's approx(): h and period must match the
// discrete table grid exactly (strucchangeRcpp errors otherwise, and so do
// we), alpha (as 1-alpha) is linearly interpolated and clamped to the
// grid's endpoints.
double interp_critval(double h, int period, double alpha) {
    int wi = -1;
    for (int i = 0; i < 3; ++i) if (std::fabs(h - kWinsizeGrid[i]) < 1e-9) wi = i;
    if (wi < 0) throw std::runtime_error("h must be one of 0.25, 0.5, 1.0 (bfastmonitor OLS-MOSUM critical-value table grid)");

    int pidx = -1;
    for (int i = 0; i < 5; ++i) if (period == kPeriodGrid[i]) pidx = i;
    if (pidx < 0) throw std::runtime_error("period must be one of 2, 4, 6, 8, 10 (bfastmonitor OLS-MOSUM critical-value table grid)");

    double target = 1.0 - alpha;
    const double lo = 0.95, step = 0.001;
    double pos = (target - lo) / step;
    if (pos <= 0.0) return kMonitorMECritval[wi][pidx][0];
    if (pos >= (double)(kAlphaGridN - 1)) return kMonitorMECritval[wi][pidx][kAlphaGridN - 1];
    int k0 = (int)std::floor(pos);
    double frac = pos - (double)k0;
    return kMonitorMECritval[wi][pidx][k0] * (1.0 - frac) + kMonitorMECritval[wi][pidx][k0 + 1] * frac;
}

double log_plus(double x) {
    return (x <= E_CONST) ? 1.0 : std::log(x);
}

// OLS-MOSUM monitoring boundary (Chu, Stinchcombe & White 1996), matching
// strucchangeRcpp's `border <- function(k) critval*sqrt(2*logPlus(k/histsize))`.
double mosum_boundary(double k_over_histsize, double critval) {
    return critval * std::sqrt(2.0 * log_plus(k_over_histsize));
}

// Builds the trend+harmonic design matrix, matching bfastpp()'s
// `response ~ trend + harmon` formula (R/bfastpp.R): column 0 = intercept,
// column 1 = trend (1-indexed observation count, R's `1:NROW(y)`), then
// `order` cos/sin pairs evaluated at the synthetic regular time
// `start_time + i/frequency` (matching R's `stats::time()` for a `ts`
// object - NOT real per-observation dates), with the Nyquist sine column
// dropped when 2*order == frequency (R keeps this edge case exactly).
Eigen::MatrixXd build_design_matrix(int n_time, double start_time, int frequency, int order) {
    order = std::min(order, frequency);
    bool drop_nyquist = (2 * order == frequency);
    int n_cols = 2 + 2 * order - (drop_nyquist ? 1 : 0);

    Eigen::MatrixXd X(n_time, n_cols);
    for (int i = 0; i < n_time; ++i) {
        double t = start_time + (double)i / (double)frequency;
        X(i, 0) = 1.0;
        X(i, 1) = (double)(i + 1);
        int col = 2;
        for (int k = 1; k <= order; ++k) X(i, col++) = std::cos(2.0 * PI * t * (double)k);
        for (int k = 1; k <= order; ++k) {
            if (drop_nyquist && k == order) continue;
            X(i, col++) = std::sin(2.0 * PI * t * (double)k);
        }
    }
    return X;
}

// Per-thread reusable buffers, sized once from n_time/n_cols (constant
// across a whole batch call - see MKScratch in mann_kendall.cpp for why
// this matters: per-pixel std::vector/Eigen allocation was ~30-80% of
// per-pixel time there for realistic series lengths).
struct BFMScratch {
    std::vector<int> hist_rows;
    std::vector<int> mon_rows;
    std::vector<int> combined;
    std::vector<double> e;
    std::vector<double> mon_resid;
    Eigen::MatrixXd Xh;
    Eigen::VectorXd Yh;

    void reserve_for(int n_time, int n_cols) {
        hist_rows.reserve(n_time);
        mon_rows.reserve(n_time);
        combined.reserve(n_time);
        e.reserve(n_time);
        mon_resid.reserve(n_time);
        Xh.resize(n_time, n_cols);
        Yh.resize(n_time);
    }
};

BFMResult bfast_monitor_impl(
    const double* y, int n_time,
    const Eigen::MatrixXd& X,
    double start_time, double monitor_start_time, int frequency,
    double h, int period, double alpha, int min_valid,
    BFMScratch& sc)
{
    BFMResult res;
    int p = (int)X.cols();

    sc.hist_rows.clear();
    sc.mon_rows.clear();
    for (int i = 0; i < n_time; ++i) {
        double t = start_time + (double)i / (double)frequency;
        bool valid = !std::isnan(y[i]);
        if (t < monitor_start_time) {
            if (valid) sc.hist_rows.push_back(i);
        } else {
            if (valid) sc.mon_rows.push_back(i);
        }
    }

    int n_hist = (int)sc.hist_rows.size();
    res.n_history = (double)n_hist;

    int K = (int)std::floor(h * (double)n_hist);
    bool ok = (K > 1) && (n_hist > p) && (n_hist >= min_valid);
    if (!ok) return res;
    res.valid = 1.0;

    // Fit OLS on the history rows (using preallocated Xh/Yh blocks, no
    // per-pixel Eigen matrix allocation).
    for (int r = 0; r < n_hist; ++r) {
        sc.Xh.row(r) = X.row(sc.hist_rows[r]);
        sc.Yh(r) = y[sc.hist_rows[r]];
    }
    auto Xh_view = sc.Xh.topRows(n_hist);
    auto Yh_view = sc.Yh.head(n_hist);
    Eigen::VectorXd beta = (Xh_view.transpose() * Xh_view).ldlt().solve(Xh_view.transpose() * Yh_view);

    Eigen::VectorXd resid_h = Yh_view - Xh_view * beta;
    int df = n_hist - p;
    double sigma = std::sqrt(resid_h.squaredNorm() / (double)df);
    res.sigma = sigma;
    if (!(sigma > 0.0) || !std::isfinite(sigma)) return res;

    sc.combined.clear();
    sc.combined.insert(sc.combined.end(), sc.hist_rows.begin(), sc.hist_rows.end());
    sc.combined.insert(sc.combined.end(), sc.mon_rows.begin(), sc.mon_rows.end());
    int n_total = (int)sc.combined.size();
    if (K > n_total) return res;

    sc.e.resize(n_total);
    for (int i = 0; i < n_total; ++i) {
        sc.e[i] = y[sc.combined[i]] - X.row(sc.combined[i]).dot(beta);
    }

    int full_len = n_total - K + 1;
    int hist_len = n_hist - K + 1;
    double denom = sigma * std::sqrt((double)n_hist);
    double critval = interp_critval(h, period, alpha);

    double running = 0.0;
    for (int i = 0; i < K; ++i) running += sc.e[i];

    bool found = false;
    for (int j = 0; j < full_len; ++j) {
        if (j > 0) running += sc.e[j + K - 1] - sc.e[j - 1];
        if (j >= hist_len && !found) {
            double process_val = running / denom;
            int k_1indexed = n_hist + 1 + (j - hist_len);
            double boundary = mosum_boundary((double)k_1indexed / (double)n_hist, critval);
            if (std::isfinite(process_val) && std::fabs(process_val) > boundary) {
                found = true;
                res.has_break = 1.0;
                int break_row = sc.combined[k_1indexed - 1];
                res.breakpoint_idx = (double)break_row;
                res.breakpoint = start_time + (double)break_row / (double)frequency;
            }
        }
    }

    // Magnitude: median residual over the monitoring-period rows.
    sc.mon_resid.clear();
    for (int idx : sc.mon_rows) sc.mon_resid.push_back(y[idx] - X.row(idx).dot(beta));
    if (!sc.mon_resid.empty()) {
        std::vector<double> tmp = sc.mon_resid; // median() needs a mutable copy; small (monitoring period size)
        size_t mid = tmp.size() / 2;
        std::nth_element(tmp.begin(), tmp.begin() + mid, tmp.end());
        double hi = tmp[mid];
        if (tmp.size() % 2 == 1) {
            res.magnitude = hi;
        } else {
            std::nth_element(tmp.begin(), tmp.begin() + mid - 1, tmp.begin() + mid);
            res.magnitude = 0.5 * (tmp[mid - 1] + hi);
        }
    }

    return res;
}

} // namespace

BFMResult bfast_monitor(
    const std::vector<double>& y,
    double start_time,
    double monitor_start_time,
    int frequency,
    int order,
    double h,
    int period,
    double alpha)
{
    Eigen::MatrixXd X = build_design_matrix((int)y.size(), start_time, frequency, order);
    BFMScratch sc;
    sc.reserve_for((int)y.size(), (int)X.cols());
    return bfast_monitor_impl(y.data(), (int)y.size(), X, start_time, monitor_start_time, frequency, h, period, alpha, 10, sc);
}

pybind11::array_t<double> fit_bfast_monitor_batch(
    pybind11::array_t<double> values_array,
    double start_time,
    double monitor_start_time,
    int frequency,
    int order,
    double h,
    int period,
    double alpha,
    int min_valid,
    int n_jobs)
{
    auto buf = values_array.request();
    if (buf.ndim != 2) throw std::runtime_error("values_array must be 2D [pixels, time]");

    int n_pixels = (int)buf.shape[0];
    int n_time = (int)buf.shape[1];
    double* ptr = static_cast<double*>(buf.ptr);

    Eigen::MatrixXd X = build_design_matrix(n_time, start_time, frequency, order);
    int n_cols = (int)X.cols();

    const int n_metrics = 7; // breakpoint, breakpoint_idx, magnitude, sigma, n_history, has_break, valid
    pybind11::array_t<double> out_arr({n_metrics, n_pixels});
    double* out_ptr = static_cast<double*>(out_arr.request().ptr);
    for (int i = 0; i < n_metrics * n_pixels; ++i) out_ptr[i] = std::nan("");

    if (n_jobs <= 0) n_jobs = std::max(1, omp_get_max_threads() - 1);

    {
        pybind11::gil_scoped_release release;
        #pragma omp parallel num_threads(n_jobs)
        {
            BFMScratch sc;
            sc.reserve_for(n_time, n_cols);

            #pragma omp for
            for (int p = 0; p < n_pixels; ++p) {
                const double* y = ptr + (size_t)p * n_time;
                BFMResult r = bfast_monitor_impl(y, n_time, X, start_time, monitor_start_time, frequency, h, period, alpha, min_valid, sc);

                out_ptr[0 * n_pixels + p] = r.breakpoint;
                out_ptr[1 * n_pixels + p] = r.breakpoint_idx;
                out_ptr[2 * n_pixels + p] = r.magnitude;
                out_ptr[3 * n_pixels + p] = r.sigma;
                out_ptr[4 * n_pixels + p] = r.n_history;
                out_ptr[5 * n_pixels + p] = r.has_break;
                out_ptr[6 * n_pixels + p] = r.valid;
            }
        }
    }

    return out_arr;
}

} // namespace bfastmonitor
} // namespace zeit
