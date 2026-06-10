//
// Created by Fable 5
//
// Benchmark of the MPA table creation strategies: sequential / parallel / merged.
// It builds a synthetic reference orbit whose period structure mimics a real nested-minibrot
// location, creates the MPA table with every strategy, reports the creation times, and verifies
// that the parallel creation is bit-identical to the sequential creation and that the merged
// creation produces the identical table structure (entries, PA counts, skips) with the an/bn
// values matching within the floating-point reassociation tolerance.
//
// usage : RFF_MPABench [periodExponent=21]
//         longest period = 2^periodExponent

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../calc/complex.hpp"
#include "../formula/MB2Reference.h"
#include "../mrthy/MPAPeriod.h"
#include "../mrthy/MPATable.h"
#include "../parallel/ParallelRenderState.h"

namespace merutilm::rff2::bench {

    static double hash01(const uint64_t i) {
        uint64_t x = i * 0x9E3779B97F4A7C15ull + 0x632BE59BD9B4E019ull;
        x ^= x >> 33;
        x *= 0xFF51AFD7ED558CCDull;
        x ^= x >> 33;
        x *= 0xC4CEB9FE1A85EC53ull;
        x ^= x >> 33;
        return static_cast<double>(x >> 11) * 0x1.0p-53;
    }

    // The orbit magnitudes are generated as |2Z(i)| = exp(w(i) - w(i-1)) with w in [-0.5, 0.5],
    // so the product of |2Z| over any range telescopes into [e^-1, e^1] and the an/bn products
    // stay in range even for millions of iterations, like a real bounded reference orbit.
    template<Number Num>
    std::vector<complex<Num>> synthOrbit(const uint64_t longestPeriod) {
        auto orbit = std::vector<complex<Num>>(longestPeriod + 1);
        orbit[0] = complex<Num>::ZERO;
        double wPrev = hash01(0) - 0.5;

        for (uint64_t i = 1; i <= longestPeriod; ++i) {
            const double w = hash01(i) - 0.5;
            const double mag = 0.5 * std::exp(w - wPrev);
            const double phase = 6.283185307179586 * hash01(i + 0x0123456789ABCDEFull);
            orbit[i] = complex<Num>{Num(mag * std::cos(phase)), Num(mag * std::sin(phase))};
            wPrev = w;
        }
        return orbit;
    }

    inline std::vector<uint64_t> synthPeriods(const uint64_t longestPeriod) {
        static constexpr uint64_t MULTIPLIERS[] = {3, 5, 7, 11};

        auto periods = std::vector<uint64_t>();
        uint64_t p = 29;
        periods.push_back(p);

        for (size_t i = 0;; ++i) {
            // non-exact multiples to exercise the remainder cascades of the block structure
            const uint64_t next = p * MULTIPLIERS[i % (sizeof MULTIPLIERS / sizeof MULTIPLIERS[0])] + p % 97 + 1;
            if (next >= longestPeriod) {
                periods.push_back(longestPeriod);
                break;
            }
            periods.push_back(next);
            p = next;
        }
        return periods;
    }

    template<Number Num>
    std::unique_ptr<MB2Reference<Num>> makeReference(const uint64_t longestPeriod) {
        return std::make_unique<MB2Reference<Num>>(
                fixed_point_complex_i1(0.0, 0.0, 5), synthOrbit<Num>(longestPeriod),
                std::vector<ArrayCompressionTool>(), synthPeriods(longestPeriod),
                fixed_point_complex(0.0, 0.0, 5, 1), fixed_point_complex(0.0, 0.0, 5, 1));
    }

    template<Number Num>
    std::unique_ptr<MPATable<Num>> buildTable(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                                              const FrtMPASettings &settings, const Num dcMax, double *seconds) {
        const auto begin = std::chrono::steady_clock::now();
        auto table = std::make_unique<MPATable<Num>>(state, reference, &settings, dcMax,
                                                     [](uint64_t, double) {});
        const auto end = std::chrono::steady_clock::now();
        *seconds = std::chrono::duration<double>(end - begin).count();
        return table;
    }

    template<Number Num>
    double relErr(const complex<Num> x, const complex<Num> y) {
        const complex<Num> diff = x - y;
        if (diff.is_zero()) {
            return 0;
        }
        if (x.is_zero()) {
            return HUGE_VAL;
        }
        return static_cast<double>(diff.norm_approx() / x.norm_approx());
    }

    struct CompareResult {
        bool structureMatch = true;
        bool bitIdentical = true;
        uint64_t paCount = 0;
        double maxAnBnRelErr = 0;
        double minRadiusRatio = HUGE_VAL;
        double maxRadiusRatio = 0;
        std::vector<double> anBnRelErrs;
        std::vector<double> radiusRatios;
    };

    inline double percentile(std::vector<double> &sorted, const double p) {
        if (sorted.empty()) {
            return 0;
        }
        const size_t index = static_cast<size_t>(p * static_cast<double>(sorted.size() - 1));
        return sorted[index];
    }

    template<Number Num>
    CompareResult compareTables(const MPATable<Num> &expected, const MPATable<Num> &actual) {
        CompareResult result{};
        const auto &te = *expected.tableManager->mpaTable;
        const auto &ta = *actual.tableManager->mpaTable;

        if (te.size() != ta.size()) {
            std::printf("    STRUCTURE MISMATCH : table size %llu != %llu\n",
                        static_cast<unsigned long long>(te.size()), static_cast<unsigned long long>(ta.size()));
            result.structureMatch = false;
            return result;
        }

        for (size_t i = 0; i < te.size(); ++i) {
            if (te[i].size() != ta[i].size()) {
                std::printf("    STRUCTURE MISMATCH : entry %llu has %llu PAs, expected %llu\n",
                            static_cast<unsigned long long>(i), static_cast<unsigned long long>(ta[i].size()),
                            static_cast<unsigned long long>(te[i].size()));
                result.structureMatch = false;
                return result;
            }
            for (size_t j = 0; j < te[i].size(); ++j) {
                const PA<Num> &pe = te[i][j];
                const PA<Num> &pa = ta[i][j];
                if (pe.skip != pa.skip) {
                    std::printf("    STRUCTURE MISMATCH : entry %llu PA %llu skip %llu, expected %llu\n",
                                static_cast<unsigned long long>(i), static_cast<unsigned long long>(j),
                                static_cast<unsigned long long>(pa.skip), static_cast<unsigned long long>(pe.skip));
                    result.structureMatch = false;
                    return result;
                }
                ++result.paCount;
                result.bitIdentical = result.bitIdentical && pe.an.re == pa.an.re && pe.an.im == pa.an.im &&
                                      pe.bn.re == pa.bn.re && pe.bn.im == pa.bn.im && pe.radius == pa.radius;
                const double anBnErr = std::max(relErr(pe.an, pa.an), relErr(pe.bn, pa.bn));
                result.maxAnBnRelErr = std::max(result.maxAnBnRelErr, anBnErr);
                result.anBnRelErrs.push_back(anBnErr);

                if (!calculatable::is_zero(pe.radius)) {
                    const double ratio = static_cast<double>(pa.radius / pe.radius);
                    if (std::isfinite(ratio)) {
                        result.minRadiusRatio = std::min(result.minRadiusRatio, ratio);
                        result.maxRadiusRatio = std::max(result.maxRadiusRatio, ratio);
                        result.radiusRatios.push_back(ratio);
                    }
                }
            }
        }
        return result;
    }

    inline const char *methodName(const FrtMPACompressionMethod method) {
        switch (method) {
            case FrtMPACompressionMethod::NO_COMPRESSION: return "NO_COMPRESSION    ";
            case FrtMPACompressionMethod::LITTLE_COMPRESSION: return "LITTLE_COMPRESSION";
            case FrtMPACompressionMethod::STRONGEST: return "STRONGEST         ";
            default: return "UNKNOWN           ";
        }
    }

    template<Number Num>
    int runBench(const char *name, const ParallelRenderState &state, const uint64_t longestPeriod) {
        const auto reference = makeReference<Num>(longestPeriod);
        const Num dcMax = Num(1e-12);
        int failures = 0;

        for (const FrtMPACompressionMethod method:
             {FrtMPACompressionMethod::NO_COMPRESSION, FrtMPACompressionMethod::LITTLE_COMPRESSION}) {

            auto settings = FrtMPASettings{8, 2, -4.0f, FrtMPASelectionMethod::HIGHEST, method, false, false};

            double seqSeconds = 0;
            const auto sequential = buildTable(state, *reference, settings, dcMax, &seqSeconds);
            const uint64_t levels = sequential->mpaPeriod->tablePeriod.size();

            std::printf("== %s | longestPeriod %" PRIu64 " | levels %" PRIu64 " | %s ==\n", name, longestPeriod,
                        levels, methodName(method));
            std::printf("  sequential : %8.3f s\n", seqSeconds);

            {
                settings.useMergedTableGeneration = true;
                double mergedSeconds = 0;
                const auto merged = buildTable(state, *reference, settings, dcMax, &mergedSeconds);
                settings.useMergedTableGeneration = false;

                CompareResult cmp = compareTables(*sequential, *merged);
                std::printf("  merged     : %8.3f s   (x%.2f)\n", mergedSeconds, seqSeconds / mergedSeconds);
                if (cmp.structureMatch) {
                    std::printf("    structure : OK (%" PRIu64 " PAs) | max an/bn rel err %.3e | radius ratio [%.6f, %.6f]\n",
                                cmp.paCount, cmp.maxAnBnRelErr, cmp.minRadiusRatio, cmp.maxRadiusRatio);

                    std::ranges::sort(cmp.anBnRelErrs);
                    std::ranges::sort(cmp.radiusRatios);
                    std::printf("    an/bn err  p50 %.3e | p90 %.3e | p99 %.3e | p99.9 %.3e\n",
                                percentile(cmp.anBnRelErrs, 0.50), percentile(cmp.anBnRelErrs, 0.90),
                                percentile(cmp.anBnRelErrs, 0.99), percentile(cmp.anBnRelErrs, 0.999));
                    std::printf("    radius     p1 %.4f | p10 %.4f | p50 %.4f | p90 %.4f | p99 %.4f\n",
                                percentile(cmp.radiusRatios, 0.01), percentile(cmp.radiusRatios, 0.10),
                                percentile(cmp.radiusRatios, 0.50), percentile(cmp.radiusRatios, 0.90),
                                percentile(cmp.radiusRatios, 0.99));
                    const auto within10 = static_cast<double>(
                                                  std::ranges::upper_bound(cmp.radiusRatios, 1.1) -
                                                  std::ranges::lower_bound(cmp.radiusRatios, 0.9)) /
                                          static_cast<double>(cmp.radiusRatios.size());
                    const auto below1 = static_cast<double>(std::ranges::lower_bound(cmp.radiusRatios, 1.0) -
                                                            cmp.radiusRatios.begin()) /
                                        static_cast<double>(cmp.radiusRatios.size());
                    std::printf("    radius     within +-10%% : %.2f%% | ratio<1 (conservative) : %.2f%%\n",
                                within10 * 100, below1 * 100);
                } else {
                    std::printf("    structure : FAILED\n");
                    ++failures;
                }
                if (cmp.maxAnBnRelErr > 1e-9) {
                    std::printf("    an/bn     : FAILED (relative error above 1e-9)\n");
                    ++failures;
                }
            }

            {
                settings.useParallelTableGeneration = true;
                double parallelSeconds = 0;
                const auto parallel = buildTable(state, *reference, settings, dcMax, &parallelSeconds);
                settings.useParallelTableGeneration = false;

                const CompareResult cmp = compareTables(*sequential, *parallel);
                std::printf("  parallel   : %8.3f s   (x%.2f)\n", parallelSeconds, seqSeconds / parallelSeconds);
                if (cmp.structureMatch && cmp.bitIdentical) {
                    std::printf("    bit-identical to sequential : OK (%" PRIu64 " PAs)\n", cmp.paCount);
                } else {
                    std::printf("    bit-identical to sequential : FAILED\n");
                    ++failures;
                }
            }
            std::printf("\n");
        }
        return failures;
    }
} // namespace merutilm::rff2::bench

int main(const int argc, char **argv) {
    using namespace merutilm::rff2;

    const uint64_t exponent = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 21;
    const uint64_t longestPeriod = 1ull << exponent;

    // always allow the parallel path so it can be measured at every size
    MPATable<double>::parallelMinLongestPeriod = 0;
    MPATable<dex>::parallelMinLongestPeriod = 0;

    const ParallelRenderState state{};

    int failures = 0;
    failures += bench::runBench<double>("LIGHT (double)", state, longestPeriod);
    failures += bench::runBench<dex>("DEEP (dex)    ", state, longestPeriod);

    if (failures != 0) {
        std::printf("BENCHMARK FINISHED WITH %d FAILURES\n", failures);
        return 1;
    }
    std::printf("BENCHMARK FINISHED : ALL CHECKS PASSED\n");
    return 0;
}
