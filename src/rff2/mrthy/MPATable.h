//
// Created by Merutilm on 2025-05-18.
// Modified by Fable 5
//

#pragma once
#include <vector>

#include <algorithm>
#include <any>
#include <atomic>
#include <mutex>
#include <thread>
#include <xmmintrin.h>

#include "../constants/Constants.hpp"
#include "../data/ApproxTableManager.h"
#include "../formula/MB2Reference.h"
#include "../parallel/ParallelRenderState.h"
#include "../settings/FrtMPACompressionMethod.h"
#include "ArrayCompressionTool.h"
#include "ArrayCompressor.h"
#include "MPAPeriod.h"
#include "PAGenerator.h"

#include "vulkan_helper/base/logger.hpp"

namespace merutilm::rff2 {
    template<Number Num>
    struct MPATable {
        static constexpr int REQUIRED_PERTURBATION = 2;

        // below this longest period, the parallel creation overhead exceeds its gain
        inline static uint64_t parallelMinLongestPeriod = std::is_same_v<Num, dex> ? 1ull << 16 : 1ull << 22;

        const FrtMPASettings mpaSettings;

        // pulled mpa : fill only valid elements from the sparse mpa vector
        // pulled mpa compressor : distinct the elements from "pulled mpa"
        std::vector<ArrayCompressionTool> pulledMPACompressor = std::vector<ArrayCompressionTool>();

        // important data to generate
        std::unique_ptr<MPAPeriod> mpaPeriod = nullptr;

        // table caches
        std::optional<ApproxTableManager<Num>> tableManager;

        explicit MPATable(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                          const FrtMPASettings *mpaSettings, Num dcMax,
                          std::function<void(uint64_t, double)> &&actionPerCreatingTableIteration);


        virtual ~MPATable() = default;

    protected:
        [[nodiscard]] bool tryInit(const MB2Reference<Num> &reference);

        [[nodiscard]] std::vector<ArrayCompressionTool>
        generatePulledMPACompressor(const std::vector<ArrayCompressionTool> &referenceCompressor) const;

        static uint64_t binarySearch(const std::vector<uint64_t> &arr, uint64_t key);

        void fitBufferSize();


        bool tryJumpTableGeneration(const MB2Reference<Num> &reference, double epsilon, Num dcMax,
                                    std::vector<uint64_t> &periodCount,
                                    std::vector<std::optional<PAGenerator<Num>>> &currentPA, uint64_t pulledTableIndex,
                                    uint64_t *currentIteration);


        void stepOnce(const MB2Reference<Num> &reference, double epsilon, Num dcMax,
                      std::vector<uint64_t> &periodCount, std::vector<std::optional<PAGenerator<Num>>> &currentPA,
                      uint64_t pulledTableIndex, const uint64_t *currentIteration, bool jumped);


        void generateTable(const ParallelRenderState &state, const MB2Reference<Num> &reference, Num dcMax,
                           std::function<void(uint64_t, double)> &&actionPerCreatingTableIteration);


        struct MergedBuildState {
            const ParallelRenderState &state;
            const MB2Reference<Num> &reference;
            const double epsilon;
            const Num dcMax;
            const std::function<void(uint64_t, double)> &actionPerCreatingTableIteration;
            uint64_t blockCounter = 0;
        };

        [[nodiscard]] bool generateTableMerged(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                                               Num dcMax,
                                               const std::function<void(uint64_t, double)> &
                                               actionPerCreatingTableIteration);

        [[nodiscard]] std::optional<PA<Num>> buildBlockMerged(MergedBuildState &ctx, uint64_t level, uint64_t start);

        [[nodiscard]] std::optional<PA<Num>> storeMergedPA(PAGenerator<Num> &generator);


        struct ParallelTableJumpStart {
            uint64_t iteration;
            uint64_t toolIndex;
        };

        struct ParallelTableContext {
            // counts the table pushes done per entry so the workers can keep the sequential push order
            std::vector<std::atomic<uint16_t>> entryFill;
            std::vector<ParallelTableJumpStart> jumpStarts;
            std::mutex allocMutex;
            std::atomic<bool> aborted{false};
            std::mutex errorMutex;
            std::exception_ptr error = nullptr;

            explicit ParallelTableContext(const size_t entries) : entryFill(entries) {}
        };

        /**
         * Inverse of @code iterationToPulledTableIndex()@endcode for valid pulled indices.
         */
        static uint64_t pulledTableIndexToIteration(const MPAPeriod &mpaPeriod, uint64_t index);

        [[nodiscard]] bool generateTableParallel(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                                                 Num dcMax,
                                                 const std::function<void(uint64_t, double)> &
                                                 actionPerCreatingTableIteration,
                                                 uint32_t workerCount);

        void generateTableParallelWorker(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                                         double epsilon, Num dcMax, ParallelTableContext &ctx, uint32_t workerId,
                                         uint32_t workerCount,
                                         const std::function<void(uint64_t, double)> *actionPerCreatingTableIteration);

        void allocateWithCheckTableSize(uint64_t index, uint64_t levels);

        /**
         * Gets the pulled table index of MPA Table.
         * @param mpaPeriod The generated MPA Period
         * @param iteration The iteration to pull
         * @return The pulled index. if not found, returns @code UINT64_MAX@endcode
         */
        static uint64_t iterationToPulledTableIndex(const MPAPeriod &mpaPeriod, uint64_t iteration);

        /**
         * Gets the finally compressed table index of MPA Table.
         * @param mpaCompressionMethod The MPA compression Method
         * @param mpaPeriod The generated MPA Period
         * @param pulledMPACompressor The compressor of pulled MPA table
         * @param iteration The iteration to pull
         * @return The finally compressed index. if not found, returns @code UINT64_MAX@endcode
         */
        static uint64_t iterationToCompTableIndex(const FrtMPACompressionMethod &mpaCompressionMethod,
                                                  const MPAPeriod &mpaPeriod,
                                                  const std::vector<ArrayCompressionTool> &pulledMPACompressor,
                                                  uint64_t iteration);


    public:
        [[nodiscard]] PA<Num> *lookup(uint64_t refIteration, complex<Num> dz) const;
        [[nodiscard]] size_t getLength();
    };

    // DEFINITION OF MPA TABLE


    template<Number Num>
    MPATable<Num>::MPATable(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                            const FrtMPASettings *mpaSettings, Num dcMax,
                            std::function<void(uint64_t, double)> &&actionPerCreatingTableIteration) :
        mpaSettings(*mpaSettings) {

        if (tryInit(reference)) {
            generateTable(state, reference, dcMax, std::move(actionPerCreatingTableIteration));
        }
    }


    //[re] init mpa periods and compressors
    template<Number Num>

    bool MPATable<Num>::tryInit(const MB2Reference<Num> &reference) {
        const auto &referencePeriod = reference.period;
        const uint64_t longestPeriod = reference.longestPeriod();

        if (const int minSkip = mpaSettings.minSkipReference; longestPeriod < minSkip) {
            this->mpaPeriod = nullptr;
            this->pulledMPACompressor = std::vector<ArrayCompressionTool>();
            return false;
        }

        const FrtMPACompressionMethod compressionMethod = mpaSettings.mpaCompressionMethod;
        this->mpaPeriod = MPAPeriod::generate(referencePeriod, mpaSettings);
        this->pulledMPACompressor = compressionMethod == FrtMPACompressionMethod::STRONGEST
                                            ? generatePulledMPACompressor(reference.compressor)
                                            : std::vector<ArrayCompressionTool>();
        return true;
    }

    template<Number Num>

    std::vector<ArrayCompressionTool>
    MPATable<Num>::generatePulledMPACompressor(const std::vector<ArrayCompressionTool> &referenceCompressor) const {
        std::vector<ArrayCompressionTool> mpaTools;
        auto &tablePeriod = mpaPeriod->tablePeriod;
        auto &tablePeriodElements = mpaPeriod->skippableIterationsCount;
        auto &isArtificial = mpaPeriod->isArtificial;

        for (ArrayCompressionTool tool: referenceCompressor) {
            const uint64_t start = tool.start;
            const uint64_t length = tool.range();
            const uint64_t index = binarySearch(tablePeriod, length + 1);

            // Check if the reference compressor is same as period.
            // However, The Computer doesn't know whether the compressor's length came from skipping to the periodic
            // point, or being cut off in the middle. So, Do check tableIndex too.

            if (const uint64_t tableIndex = iterationToPulledTableIndex(*mpaPeriod, start);
                index != UINT64_MAX && tableIndex != UINT64_MAX && !isArtificial[index]) {
                const uint64_t periodElements = tablePeriodElements[index];
                mpaTools.emplace_back(1, tableIndex + 1, tableIndex + periodElements - 1);
            }
        }
        return mpaTools;
    }

    template<Number Num>
    uint64_t MPATable<Num>::binarySearch(const std::vector<uint64_t> &arr, const uint64_t key) {
        if (arr.empty()) {
            return UINT64_MAX;
        }

        uint64_t low = 0;
        uint64_t high = arr.size() - 1;

        while (low <= high) {
            const uint64_t mid = (low + high) >> 1;
            if (const uint64_t value = arr[mid]; value < key) {
                low = mid + 1;
            } else if (value > key) {
                if (mid == 0) {
                    return UINT64_MAX;
                }
                high = mid - 1;
            } else
                return mid;
        }
        return UINT64_MAX;
    }


    template<Number Num>

    void MPATable<Num>::fitBufferSize() {


        // no compression : lastCompIndex > skippableIterationsCount.back()
        // pulled : lastCompIndex == skippableIterationsCount.back()
        // compressed : lastCompIndex < skippableIterationsCount.back()

        const uint64_t longestPeriod = mpaPeriod->tablePeriod.back();
        const uint64_t levels = mpaPeriod->tablePeriod.size();
        const uint64_t skippableIterationsCount = mpaPeriod->skippableIterationsCount.back();
        const uint64_t lastCompIndex = iterationToCompTableIndex(mpaSettings.mpaCompressionMethod, *mpaPeriod,
                                                                 pulledMPACompressor, longestPeriod + 1);
        const uint64_t bufferSize = levels * PA<Num>::size() * std::min(lastCompIndex, skippableIterationsCount) +
                                    lastCompIndex * sizeof(std::pmr::vector<PA<Num>>);
        tableManager.emplace(bufferSize, lastCompIndex);
    }

    template<Number Num>
    bool MPATable<Num>::tryJumpTableGeneration(const MB2Reference<Num> &reference, double epsilon, Num dcMax,
                                               std::vector<uint64_t> &periodCount,
                                               std::vector<std::optional<PAGenerator<Num>>> &currentPA,
                                               const uint64_t pulledTableIndex, uint64_t *const currentIteration) {

        const ArrayCompressionTool *containedTool = ArrayCompressor::find(pulledMPACompressor, pulledTableIndex + 1);
        if (containedTool == nullptr || containedTool->start != pulledTableIndex + 1) {
            return false;
        }
        auto &table = *tableManager->mpaTable;
        const auto &tablePeriod = mpaPeriod->tablePeriod;
        const uint64_t levels = tablePeriod.size();
        const std::pmr::vector<PA<Num>> &mainReferenceMPA = table[0];
        const auto &skippableIterationsCount = mpaPeriod->skippableIterationsCount;
        const uint64_t level = binarySearch(skippableIterationsCount, containedTool->end - containedTool->start + 2);
        // count itself and periodic point, +2

        const uint64_t compTableIndex = iterationToCompTableIndex(mpaSettings.mpaCompressionMethod, *mpaPeriod,
                                                                  pulledMPACompressor, *currentIteration);

        allocateWithCheckTableSize(compTableIndex, levels);
        auto &pa = table[compTableIndex];

        const PA<Num> &mainReferencePA = mainReferenceMPA[level];
        const uint64_t skip = mainReferencePA.skip;

        for (uint64_t i = level + 1; i < levels; ++i) {
            if (i <= level && periodCount[i] != 0) {
                vkh::logger::w_log(L"WARNING : Failed to compress!! \n what : the table period count {} is not zero.",
                                   periodCount[i]);
                return false;
            }
            if (periodCount[i] + skip > tablePeriod[i] - REQUIRED_PERTURBATION) {
                vkh::logger::w_log(L"WARNING : Failed to compress!! \n what : the table period count {} + "
                                   L"skip {} exceeds its period {}.",
                                   periodCount[i], skip, tablePeriod[i]);
                return false;
            }
        }


        for (uint64_t i = 0; i < levels; ++i) {
            if (i <= level) {
                pa.push_back(mainReferenceMPA[i]);
                uint64_t count = skip;
                for (uint64_t j = level; j > i; --j) {
                    count %= tablePeriod[j - 1];
                }
                currentPA[i] = std::nullopt;
                periodCount[i] = count;
            } else {
                if (currentPA[i] == std::nullopt) {
                    currentPA[i].emplace(reference, epsilon, dcMax, *currentIteration);
                    currentPA[i]->merge(mainReferencePA);
                } else {
                    currentPA[i]->merge(mainReferencePA);
                }
                periodCount[i] += skip;
            }
        }

        *currentIteration += skip;
        return true;
    }


    template<Number Num>

    void MPATable<Num>::stepOnce(const MB2Reference<Num> &reference, double epsilon, Num dcMax,
                                 std::vector<uint64_t> &periodCount,
                                 std::vector<std::optional<PAGenerator<Num>>> &currentPA,
                                 const uint64_t pulledTableIndex, const uint64_t *const currentIteration,
                                 const bool jumped) {

        bool resetLowerLevel = false;
        const bool independent = ArrayCompressor::isIndependent(pulledMPACompressor, pulledTableIndex);

        auto &table = *tableManager->mpaTable;
        const auto &tablePeriod = mpaPeriod->tablePeriod;
        const uint64_t levels = tablePeriod.size();

        for (uint64_t i = levels; i > 0; --i) {
            const uint64_t level = i - 1;
            std::optional<PAGenerator<Num>> &currentLevel = currentPA[level];
            if (periodCount[level] == 0 && independent && !jumped) {
                currentLevel.emplace(reference, epsilon, dcMax, *currentIteration);
            }

            if (currentLevel != std::nullopt && periodCount[level] + REQUIRED_PERTURBATION < tablePeriod[level]) {
                currentLevel->step();
            }


            ++periodCount[level];

            if (periodCount[level] == tablePeriod[level]) {
                if (currentLevel != std::nullopt &&
                    currentLevel->skip == tablePeriod[level] - REQUIRED_PERTURBATION) {
                    const uint64_t compTableIndex =
                            iterationToCompTableIndex(mpaSettings.mpaCompressionMethod, *mpaPeriod, pulledMPACompressor,
                                                      currentLevel->start);


                    if (compTableIndex == UINT64_MAX) {
                        vkh::logger::w_log(L"FATAL : FAILED TO CREATING TABLE!!\n what : iteration {} is not "
                                           L"pullable. aborting the table creation...",
                                           currentLevel->start);
                        return;
                    }

                    allocateWithCheckTableSize(compTableIndex, levels);
                    auto &pa = table[compTableIndex];

                    if (pa.empty() || pa.back().skip < currentLevel->skip)
                        pa.push_back(currentLevel->build(tableManager->strictTableResource.get()));
                    else {
                        vkh::logger::w_log(L"WARNING : The insertion of pa generated from compressed index {} is not "
                                           L"allowed. It might be a bug!",
                                           compTableIndex);
                    }
                }
                // Stop all lower level iteration for efficiency
                // because it is too hard to skipping to next part of the periodic point
                currentLevel = std::nullopt;
                resetLowerLevel = true;
            }

            if (resetLowerLevel) {
                periodCount[level] = 0;
            }
        }
    }


    template<Number Num>
    void MPATable<Num>::generateTable(const ParallelRenderState &state, const MB2Reference<Num> &reference, Num dcMax,
                                      std::function<void(uint64_t, double)> &&actionPerCreatingTableIteration) {


        const auto &tablePeriod = mpaPeriod->tablePeriod;
        const uint64_t longestPeriod = tablePeriod.back();
        const auto epsilonPower = mpaSettings.epsilonPower;

        if (longestPeriod < mpaSettings.minSkipReference)
            return;

        const size_t levels = tablePeriod.size();
        fitBufferSize();

        const auto func = std::move(actionPerCreatingTableIteration);
        const double epsilon = pow(10, epsilonPower);

        if (mpaSettings.useMergedTableGeneration && generateTableMerged(state, reference, dcMax, func)) {
            return;
        }

        if (const uint32_t workerCount = static_cast<uint32_t>(std::min<uint64_t>(
                    levels, std::max(1u, std::thread::hardware_concurrency())));
            mpaSettings.useParallelTableGeneration && longestPeriod >= parallelMinLongestPeriod &&
            workerCount >= 2 && generateTableParallel(state, reference, dcMax, func, workerCount)) {
            return;
        }

        uint64_t iteration = 1;
        uint64_t absIteration = 0;
        auto periodCount = std::vector<uint64_t>(levels, 0);
        auto currentPA = std::vector<std::optional<PAGenerator<Num>>>(levels);

        while (iteration <= longestPeriod) {
            if (absIteration % Constants::Fractal::EXIT_CHECK_INTERVAL == 0 && state.interruptRequested())
                return;

            func(iteration, static_cast<double>(iteration) / static_cast<double>(longestPeriod));
            const uint64_t pulledTableIndex = iterationToPulledTableIndex(*mpaPeriod, iteration);
            bool jumped = false;

            if (tryJumpTableGeneration(reference, epsilon, dcMax, periodCount, currentPA, pulledTableIndex,
                                       &iteration)) {
                jumped = true;
            }

            stepOnce(reference, epsilon, dcMax, periodCount, currentPA, pulledTableIndex, &iteration, jumped);

            ++iteration;
            ++absIteration;
        }
    }

    // Merge-based table creation. Only the level-0 blocks step the orbit; every upper-level PA is
    // composed from its sub-block PAs with merge() plus the REQUIRED_PERTURBATION raw steps between
    // the blocks, so the total step() count is about the longest period instead of
    // levels * longest period. The block layout replicates the greedy decomposition of stepOnce()
    // (each level tiles from the enclosing block start, the remainders cascade downwards), so the
    // stored skips and entries are identical to the sequential creation. The an/bn values are
    // composed segment-wise instead of per-iteration, which differs from the sequential results only
    // by the floating-point reassociation. The radius is folded through mergeConservative(), so it
    // never exceeds the per-step sequential validity bound and no epsilon adjustment is required.
    template<Number Num>
    bool MPATable<Num>::generateTableMerged(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                                            Num dcMax,
                                            const std::function<void(uint64_t, double)> &
                                            actionPerCreatingTableIteration) {
        if constexpr (PABase<Num>::MAX_DEGREE != 1) {
            // merge() of the higher degrees is not supported
            return false;
        } else {
            if (!pulledMPACompressor.empty()) {
                // the jump restructuring of tryJumpTableGeneration() cannot be replayed by the block
                // recursion, use the sequential/parallel creation
                return false;
            }
            const double epsilon = pow(10, mpaSettings.epsilonPower);
            MergedBuildState ctx{state, reference, epsilon, dcMax, actionPerCreatingTableIteration};
            static_cast<void>(buildBlockMerged(ctx, mpaPeriod->tablePeriod.size() - 1, 1));
            return true;
        }
    }


    template<Number Num>
    std::optional<PA<Num>> MPATable<Num>::buildBlockMerged(MergedBuildState &ctx, const uint64_t level,
                                                           const uint64_t start) {
        const auto &tablePeriod = mpaPeriod->tablePeriod;
        const uint64_t longestPeriod = tablePeriod.back();

        if (level == 0) {
            // every level-0 block spans tablePeriod[0] >= 4 iterations, the mask keeps the
            // interrupt-check cadence near EXIT_CHECK_INTERVAL
            if ((ctx.blockCounter++ & 0x3f) == 0 && ctx.state.interruptRequested()) {
                return std::nullopt;
            }
            ctx.actionPerCreatingTableIteration(start,
                                                static_cast<double>(start) / static_cast<double>(longestPeriod));

            auto generator = PAGenerator<Num>(ctx.reference, ctx.epsilon, ctx.dcMax, start);
            const uint64_t steps = tablePeriod[0] - REQUIRED_PERTURBATION;
            for (uint64_t k = 0; k < steps; ++k) {
                generator.step();
            }
            return storeMergedPA(generator);
        }

        auto composite = PAGenerator<Num>(ctx.reference, ctx.epsilon, ctx.dcMax, start);
        uint64_t remainder = tablePeriod[level];
        uint64_t offset = 0;

        for (uint64_t i = level; i > 0; --i) {
            const uint64_t subPeriod = tablePeriod[i - 1];
            const uint64_t groups = remainder / subPeriod;
            remainder %= subPeriod;

            for (uint64_t g = 0; g < groups; ++g) {
                if (offset != 0) {
                    // a sub-block PA skips its period minus REQUIRED_PERTURBATION iterations,
                    // the gap before the next block start is stepped raw
                    for (int k = 0; k < REQUIRED_PERTURBATION; ++k) {
                        composite.step();
                    }
                }
                const std::optional<PA<Num>> subPA = buildBlockMerged(ctx, i - 1, start + offset);
                if (subPA == std::nullopt) {
                    return std::nullopt;
                }
                composite.mergeConservative(*subPA);
                offset += subPeriod;
            }
        }

        // the unstructured tail whose length is below the shortest period
        const uint64_t targetSkip = tablePeriod[level] - REQUIRED_PERTURBATION;
        while (composite.skip < targetSkip) {
            composite.step();
        }
        return storeMergedPA(composite);
    }


    template<Number Num>
    std::optional<PA<Num>> MPATable<Num>::storeMergedPA(PAGenerator<Num> &generator) {
        const uint64_t levels = mpaPeriod->tablePeriod.size();
        const uint64_t compTableIndex = iterationToCompTableIndex(mpaSettings.mpaCompressionMethod, *mpaPeriod,
                                                                  pulledMPACompressor, generator.start);
        if (compTableIndex == UINT64_MAX) {
            vkh::logger::w_log(L"FATAL : FAILED TO CREATING TABLE!!\n what : iteration {} is not "
                               L"pullable. aborting the table creation...",
                               generator.start);
            return std::nullopt;
        }

        allocateWithCheckTableSize(compTableIndex, levels);
        auto &pa = (*tableManager->mpaTable)[compTableIndex];
        PA<Num> built = generator.build(tableManager->strictTableResource.get());

        if (pa.empty() || pa.back().skip < built.skip) {
            pa.push_back(built);
        } else {
            vkh::logger::w_log(L"WARNING : The insertion of pa generated from compressed index {} is not "
                               L"allowed. It might be a bug!",
                               compTableIndex);
        }
        return built;
    }


    template<Number Num>
    uint64_t MPATable<Num>::pulledTableIndexToIteration(const MPAPeriod &mpaPeriod, const uint64_t index) {
        const auto &tablePeriod = mpaPeriod.tablePeriod;
        const auto &tablePeriodElements = mpaPeriod.skippableIterationsCount;

        uint64_t iteration = 1;
        uint64_t remainder = index;

        for (uint64_t i = tablePeriod.size(); i > 0; --i) {
            iteration += remainder / tablePeriodElements[i - 1] * tablePeriod[i - 1];
            remainder %= tablePeriodElements[i - 1];
        }
        return iteration;
    }


    template<Number Num>
    bool MPATable<Num>::generateTableParallel(const ParallelRenderState &state, const MB2Reference<Num> &reference,
                                              Num dcMax,
                                              const std::function<void(uint64_t, double)> &
                                              actionPerCreatingTableIteration, const uint32_t workerCount) {

        const double epsilon = pow(10, mpaSettings.epsilonPower);
        auto ctx = ParallelTableContext(tableManager->mpaTable->size());

        ctx.jumpStarts.reserve(pulledMPACompressor.size());
        for (uint64_t toolIndex = 0; toolIndex < pulledMPACompressor.size(); ++toolIndex) {
            const uint64_t startIndex = pulledMPACompressor[toolIndex].start - 1;
            const uint64_t startIteration = pulledTableIndexToIteration(*mpaPeriod, startIndex);
            if (iterationToPulledTableIndex(*mpaPeriod, startIteration) != startIndex ||
                !ArrayCompressor::isIndependent(pulledMPACompressor, startIndex)) {
                // the jump entry shares a compressed entry with another start.
                // the parallel push ordering cannot be predetermined, use the sequential creation
                return false;
            }
            ctx.jumpStarts.emplace_back(startIteration, toolIndex);
        }
        std::ranges::sort(ctx.jumpStarts, {}, &ParallelTableJumpStart::iteration);

        {
            auto workers = std::vector<std::jthread>();
            workers.reserve(workerCount - 1);
            for (uint32_t workerId = 1; workerId < workerCount; ++workerId) {
                workers.emplace_back([this, &state, &reference, epsilon, dcMax, &ctx, workerId, workerCount] {
                    generateTableParallelWorker(state, reference, epsilon, dcMax, ctx, workerId, workerCount, nullptr);
                });
            }
            generateTableParallelWorker(state, reference, epsilon, dcMax, ctx, 0, workerCount,
                                        &actionPerCreatingTableIteration);
        }

        if (ctx.error != nullptr) {
            std::rethrow_exception(ctx.error);
        }
        return true;
    }


    template<Number Num>
    void MPATable<Num>::generateTableParallelWorker(const ParallelRenderState &state,
                                                    const MB2Reference<Num> &reference, const double epsilon,
                                                    Num dcMax, ParallelTableContext &ctx, const uint32_t workerId,
                                                    const uint32_t workerCount,
                                                    const std::function<void(uint64_t, double)> *
                                                    actionPerCreatingTableIteration) {
        // Replicates the sequential generateTable() loop. Every worker executes the identical
        // integer control flow (period counters, jump decisions, generator lifetimes as "shadows"),
        // and materializes the floating-point PAGenerator work only for its owned levels
        // (level % workerCount == workerId). Therefore each generator receives exactly the same
        // step()/merge() sequence as the sequential creation, and the result is bit-identical.
        // The per-entry atomic counter "entryFill" enforces the sequential push order of each entry.
        try {
            auto &table = *tableManager->mpaTable;
            const auto &tablePeriod = mpaPeriod->tablePeriod;
            const auto &skippableIterationsCount = mpaPeriod->skippableIterationsCount;
            const uint64_t levels = tablePeriod.size();
            const uint64_t longestPeriod = tablePeriod.back();
            const FrtMPACompressionMethod method = mpaSettings.mpaCompressionMethod;

            auto periodCount = std::vector<uint64_t>(levels, 0);
            auto currentPA = std::vector<std::optional<PAGenerator<Num>>>(levels);

            // integer shadows of every level's generator
            auto genActive = std::vector<uint8_t>(levels, 0);
            auto genStart = std::vector<uint64_t>(levels, 0);
            auto genSkip = std::vector<uint64_t>(levels, 0);
            auto genEntry = std::vector<uint64_t>(levels, UINT64_MAX);
            auto genOrdinal = std::vector<uint16_t>(levels, 0);
            auto genTurnPending = std::vector<uint8_t>(levels, 0);
            auto mainSkip = std::vector<uint64_t>(levels, 0);

            const auto owned = [workerId, workerCount](const uint64_t level) {
                return level % workerCount == workerId;
            };
            const auto interrupted = [&ctx, &state] {
                return ctx.aborted.load(std::memory_order_relaxed) || state.interruptRequested();
            };
            // waits until the given entry reaches the given push turn. returns false on interruption.
            const auto waitTurn = [&ctx, &interrupted](const uint64_t entry, const uint16_t turn) {
                const std::atomic<uint16_t> &fill = ctx.entryFill[entry];
                uint32_t spin = 0;
                while (fill.load(std::memory_order_acquire) < turn) {
                    if ((++spin & 0xfff) == 0 && interrupted()) {
                        return false;
                    }
                    _mm_pause();
                }
                return !ctx.aborted.load(std::memory_order_relaxed);
            };
            const auto finishTurn = [&ctx](const uint64_t entry, const uint16_t turn) {
                ctx.entryFill[entry].store(turn + 1, std::memory_order_release);
            };
            // waits until the main reference PA of the given level is readable
            const auto waitMainPA = [&ctx, &interrupted](const uint64_t level) {
                const std::atomic<uint16_t> &fill = ctx.entryFill[0];
                uint32_t spin = 0;
                while (fill.load(std::memory_order_acquire) <= level) {
                    if ((++spin & 0xfff) == 0 && interrupted()) {
                        return false;
                    }
                    _mm_pause();
                }
                return !ctx.aborted.load(std::memory_order_relaxed);
            };
            // the strict table resource is not thread-safe, so every allocation is locked
            const auto reserveForPush = [&ctx, levels](std::pmr::vector<PA<Num>> &pa) {
                if (pa.size() == pa.capacity()) {
                    std::scoped_lock lock(ctx.allocMutex);
                    pa.reserve(pa.capacity() + levels);
                }
            };
            const auto pushPA = [&ctx](std::pmr::vector<PA<Num>> &pa, auto &&makePA) {
                if constexpr (PABase<Num>::MAX_DEGREE > 1) {
                    // the PA of higher degrees allocates its term vector from the strict table resource
                    std::scoped_lock lock(ctx.allocMutex);
                    pa.push_back(makePA());
                } else {
                    pa.push_back(makePA());
                }
            };
            const auto buildLocked = [this, &ctx](PAGenerator<Num> &generator) {
                if constexpr (PABase<Num>::MAX_DEGREE > 1) {
                    std::scoped_lock lock(ctx.allocMutex);
                    return generator.build(tableManager->strictTableResource.get());
                } else {
                    return generator.build(tableManager->strictTableResource.get());
                }
            };

            // Multiple turns of one iteration must be consumed in ordinal order, but the level loop
            // visits the levels downwards. They are deferred and flushed in a globally consistent
            // (entry, ordinal) order at the end of each iteration to avoid the deadlock.
            enum class TurnKind : uint8_t { EMPTY, COPY_MAIN, STORE };
            struct DeferredTurn {
                uint64_t entry;
                uint16_t ordinal;
                TurnKind kind;
                uint64_t level;
                uint64_t skip;
                std::optional<PA<Num>> built;
            };
            auto deferredTurns = std::vector<DeferredTurn>();

            const auto flushTurns = [&]() -> bool {
                std::ranges::sort(deferredTurns, [](const DeferredTurn &a, const DeferredTurn &b) {
                    return a.entry != b.entry ? a.entry < b.entry : a.ordinal < b.ordinal;
                });
                for (DeferredTurn &turn: deferredTurns) {
                    if (!waitTurn(turn.entry, turn.ordinal)) {
                        return false;
                    }
                    switch (turn.kind) {
                        case TurnKind::EMPTY:
                            break;
                        case TurnKind::COPY_MAIN: {
                            if (!waitMainPA(turn.level)) {
                                return false;
                            }
                            auto &pa = table[turn.entry];
                            reserveForPush(pa);
                            pushPA(pa, [&table, &turn]() -> const PA<Num> & { return table[0][turn.level]; });
                            break;
                        }
                        case TurnKind::STORE: {
                            auto &pa = table[turn.entry];
                            reserveForPush(pa);
                            if (pa.empty() || pa.back().skip < turn.skip) {
                                pushPA(pa, [&turn] { return std::move(*turn.built); });
                            } else {
                                vkh::logger::w_log(L"WARNING : The insertion of pa generated from compressed "
                                                   L"index {} is not allowed. It might be a bug!",
                                                   turn.entry);
                            }
                            break;
                        }
                    }
                    finishTurn(turn.entry, turn.ordinal);
                }
                deferredTurns.clear();
                return true;
            };

            uint64_t iteration = 1;
            uint64_t absIteration = 0;
            size_t jumpCursor = 0;

            while (iteration <= longestPeriod) {
                if (absIteration % Constants::Fractal::EXIT_CHECK_INTERVAL == 0 && interrupted()) {
                    return;
                }
                if (actionPerCreatingTableIteration != nullptr) {
                    (*actionPerCreatingTableIteration)(iteration,
                                                       static_cast<double>(iteration) / static_cast<double>(
                                                           longestPeriod));
                }

                // valid pulled indices only exist where the lowest-level period counter is zero
                const uint64_t pulledTableIndex = periodCount[0] == 0
                                                      ? iterationToPulledTableIndex(*mpaPeriod, iteration)
                                                      : UINT64_MAX;
                bool jumped = false;

                while (jumpCursor < ctx.jumpStarts.size() && ctx.jumpStarts[jumpCursor].iteration < iteration) {
                    ++jumpCursor;
                }

                // tryJumpTableGeneration() replication
                if (jumpCursor < ctx.jumpStarts.size() && ctx.jumpStarts[jumpCursor].iteration == iteration) {
                    const ArrayCompressionTool &tool = pulledMPACompressor[ctx.jumpStarts[jumpCursor].toolIndex];
                    ++jumpCursor;

                    const uint64_t level = binarySearch(skippableIterationsCount, tool.end - tool.start + 2);
                    const uint64_t compTableIndex = iterationToCompTableIndex(method, *mpaPeriod, pulledMPACompressor,
                                                                              iteration);
                    if (level == UINT64_MAX || compTableIndex >= table.size()) {
                        throw vkh::exception_init("index out of range");
                    }

                    const uint64_t skip = mainSkip[level];
                    bool accepted = true;
                    for (uint64_t i = level + 1; i < levels; ++i) {
                        if (periodCount[i] + skip > tablePeriod[i] - REQUIRED_PERTURBATION) {
                            if (workerId == 0) {
                                vkh::logger::w_log(L"WARNING : Failed to compress!! \n what : the table period count "
                                                   L"{} + skip {} exceeds its period {}.",
                                                   periodCount[i], skip, tablePeriod[i]);
                            }
                            accepted = false;
                            break;
                        }
                    }

                    if (accepted) {
                        uint16_t createdOrdinal = static_cast<uint16_t>(level + 1);

                        for (uint64_t i = 0; i < levels; ++i) {
                            if (i <= level) {
                                if (genActive[i] && genTurnPending[i]) {
                                    // generator killed by the jump : consume its pending turn without push
                                    if (owned(i)) {
                                        deferredTurns.push_back({genEntry[i], genOrdinal[i], TurnKind::EMPTY, 0, 0,
                                                                 std::nullopt});
                                    }
                                    genTurnPending[i] = 0;
                                }
                                genActive[i] = 0;

                                uint64_t count = skip;
                                for (uint64_t j = level; j > i; --j) {
                                    count %= tablePeriod[j - 1];
                                }
                                periodCount[i] = count;

                                if (owned(i)) {
                                    currentPA[i] = std::nullopt;
                                    deferredTurns.push_back({compTableIndex, static_cast<uint16_t>(i),
                                                             TurnKind::COPY_MAIN, i, 0, std::nullopt});
                                }
                            } else {
                                const bool wasActive = genActive[i] != 0;
                                if (!wasActive) {
                                    genActive[i] = 1;
                                    genStart[i] = iteration;
                                    genSkip[i] = 0;
                                    genEntry[i] = compTableIndex;
                                    // only block-aligned creations can ever be stored, so only they take a turn
                                    if (periodCount[i] == 0) {
                                        genOrdinal[i] = createdOrdinal++;
                                        genTurnPending[i] = 1;
                                    } else {
                                        genTurnPending[i] = 0;
                                    }
                                }
                                if (owned(i)) {
                                    if (!waitMainPA(level)) {
                                        return;
                                    }
                                    const PA<Num> &mainReferencePA = table[0][level];
                                    if (!wasActive) {
                                        currentPA[i].emplace(reference, epsilon, dcMax, iteration);
                                    }
                                    currentPA[i]->merge(mainReferencePA);
                                }
                                genSkip[i] += skip;
                                periodCount[i] += skip;
                            }
                        }

                        iteration += skip;
                        jumped = true;
                    }
                }

                // stepOnce() replication
                bool resetLowerLevel = false;
                const bool independent = ArrayCompressor::isIndependent(pulledMPACompressor, pulledTableIndex);

                for (uint64_t i = levels; i > 0; --i) {
                    const uint64_t level = i - 1;

                    if (periodCount[level] == 0 && independent && !jumped) {
                        if (genActive[level] && genTurnPending[level]) {
                            // replaced generator : consume its pending turn without push
                            if (owned(level)) {
                                deferredTurns.push_back({genEntry[level], genOrdinal[level], TurnKind::EMPTY, 0, 0,
                                                         std::nullopt});
                            }
                            genTurnPending[level] = 0;
                        }
                        const uint64_t compTableIndex = method == FrtMPACompressionMethod::NO_COMPRESSION
                                                            ? iteration
                                                            : pulledTableIndex == UINT64_MAX
                                                                  ? UINT64_MAX
                                                                  : method == FrtMPACompressionMethod::LITTLE_COMPRESSION
                                                                        ? pulledTableIndex
                                                                        : ArrayCompressor::compress(
                                                                            pulledMPACompressor, pulledTableIndex);
                        genActive[level] = 1;
                        genStart[level] = iteration;
                        genSkip[level] = 0;
                        genEntry[level] = compTableIndex;
                        genOrdinal[level] = static_cast<uint16_t>(level);
                        genTurnPending[level] = compTableIndex != UINT64_MAX && compTableIndex < table.size() ? 1 : 0;
                        if (owned(level)) {
                            currentPA[level].emplace(reference, epsilon, dcMax, iteration);
                        }
                    }

                    if (genActive[level] && periodCount[level] + REQUIRED_PERTURBATION < tablePeriod[level]) {
                        if (owned(level)) {
                            currentPA[level]->step();
                        }
                        ++genSkip[level];
                    }

                    ++periodCount[level];

                    if (periodCount[level] == tablePeriod[level]) {
                        if (genActive[level] && genSkip[level] == tablePeriod[level] - REQUIRED_PERTURBATION) {
                            const uint64_t compTableIndex = genEntry[level];

                            if (compTableIndex == UINT64_MAX) {
                                if (workerId == 0) {
                                    vkh::logger::w_log(L"FATAL : FAILED TO CREATING TABLE!!\n what : iteration {} is "
                                                       L"not pullable. aborting the table creation...",
                                                       genStart[level]);
                                }
                                // the sequential creation skips the lower levels of this iteration
                                break;
                            }
                            if (compTableIndex >= table.size()) {
                                throw vkh::exception_init("index out of range");
                            }
                            if (genStart[level] == 1) {
                                mainSkip[level] = genSkip[level];
                            }
                            if (owned(level)) {
                                deferredTurns.push_back({compTableIndex, genOrdinal[level], TurnKind::STORE, level,
                                                         genSkip[level], buildLocked(*currentPA[level])});
                            }
                            genTurnPending[level] = 0;
                        } else if (genActive[level] && genTurnPending[level]) {
                            // completed without a storable skip : consume the turn without push
                            if (owned(level)) {
                                deferredTurns.push_back({genEntry[level], genOrdinal[level], TurnKind::EMPTY, 0, 0,
                                                         std::nullopt});
                            }
                            genTurnPending[level] = 0;
                        }
                        genActive[level] = 0;
                        if (owned(level)) {
                            currentPA[level] = std::nullopt;
                        }
                        resetLowerLevel = true;
                    }

                    if (resetLowerLevel) {
                        if (periodCount[level] != tablePeriod[level] && genActive[level] && genTurnPending[level]) {
                            // cut by an upper-level completion : the generator lingers but is never stored
                            if (owned(level)) {
                                deferredTurns.push_back({genEntry[level], genOrdinal[level], TurnKind::EMPTY, 0, 0,
                                                         std::nullopt});
                            }
                            genTurnPending[level] = 0;
                        }
                        periodCount[level] = 0;
                    }
                }

                if (!deferredTurns.empty() && !flushTurns()) {
                    return;
                }

                ++iteration;
                ++absIteration;

                // Fast-forwards the plain iterations in bulk : while no level reaches a block
                // boundary and no jump fires, the level loop above only steps the generators and
                // increments the counters, so the duplicated per-level control flow is skipped.
                // The generators receive the identical step() sequence.
                uint64_t gap = iteration <= longestPeriod ? longestPeriod - iteration + 1 : 0;
                if (jumpCursor < ctx.jumpStarts.size()) {
                    const uint64_t nextJump = ctx.jumpStarts[jumpCursor].iteration;
                    gap = std::min(gap, nextJump > iteration ? nextJump - iteration : 0);
                }
                for (uint64_t level = 0; level < levels && gap > 0; ++level) {
                    const uint64_t bound = tablePeriod[level] - REQUIRED_PERTURBATION;
                    gap = periodCount[level] == 0 || periodCount[level] >= bound
                              ? 0
                              : std::min(gap, bound - periodCount[level]);
                }
                if (gap > 0) {
                    if (interrupted()) {
                        return;
                    }
                    for (uint64_t level = 0; level < levels; ++level) {
                        if (genActive[level]) {
                            if (owned(level)) {
                                PAGenerator<Num> &generator = *currentPA[level];
                                for (uint64_t k = 0; k < gap; ++k) {
                                    generator.step();
                                }
                            }
                            genSkip[level] += gap;
                        }
                        periodCount[level] += gap;
                    }
                    if (actionPerCreatingTableIteration != nullptr) {
                        for (uint64_t k = 0; k < gap; ++k) {
                            (*actionPerCreatingTableIteration)(iteration + k,
                                                               static_cast<double>(iteration + k) /
                                                                       static_cast<double>(longestPeriod));
                        }
                    }
                    iteration += gap;
                    absIteration += gap;
                }
            }
        } catch (...) {
            ctx.aborted.store(true, std::memory_order_relaxed);
            std::scoped_lock lock(ctx.errorMutex);
            if (ctx.error == nullptr) {
                ctx.error = std::current_exception();
            }
        }
    }

    template<Number Num>

    uint64_t MPATable<Num>::iterationToPulledTableIndex(const MPAPeriod &mpaPeriod, const uint64_t iteration) {
        //
        // get index <=> Inverse calculation of index compression
        // First approach : check the remainder == 1
        //
        // [3, 11, 26]
        // 1 4 7 12 15 18 23 27 30 33 38
        //
        // test input : 23
        // search period : period 11
        // 23 % 11 = 1, 23/11 = 2.xxx(3*2 elements)
        // 1 % 3 = 1, 1/3 = 0.xxx(1*0 elements)
        // result = 3*2=6
        //
        // test input : 30
        // search period : period 26
        // 30 % 26 = 4, 30/26 = 1.xxx(7*1 elements)
        // 4 % 3 = 1, 4/3 = 1.xxx(1 element)
        // result = 7*1+1=8
        //
        // test input : 29
        // search period : period 26
        // 29 % 26 = 3, 29/26 = 1.xxx(7*1 elements)
        // 3 % 3 = 0, 3/3 = 1.xxx(1 element)
        // result = UINT64_MAX (last remainder is not one)
        //
        //
        //

        if (iteration == 0) {
            return UINT64_MAX;
        }

        const auto &tablePeriod = mpaPeriod.tablePeriod;
        const auto &tablePeriodElements = mpaPeriod.skippableIterationsCount;

        uint64_t index = 0;
        uint64_t remainder = iteration;

        for (uint64_t i = tablePeriod.size(); i > 0; --i) {
            if (remainder < tablePeriod[i - 1]) {
                continue;
            }
            // p[4, 1000]
            // 1 5 9 13 .... 993 997 1001
            // 997 % 1000 = 997
            // 997 % 4 = 1
            // 997 + 4 - 2 + 1 = 1000
            if (i < tablePeriod.size() && remainder + tablePeriod[0] - REQUIRED_PERTURBATION + 1 > tablePeriod[i]) {
                return UINT64_MAX;
                // Insufficient length, ("Pulled Table Index" must be skipped for at least "shortest period")
            }


            index += remainder / tablePeriod[i - 1] * tablePeriodElements[i - 1];
            remainder %= tablePeriod[i - 1];
        }
        return remainder == 1 ? index : UINT64_MAX;
    }


    template<Number Num>

    uint64_t MPATable<Num>::iterationToCompTableIndex(const FrtMPACompressionMethod &mpaCompressionMethod,
                                                      const MPAPeriod &mpaPeriod,
                                                      const std::vector<ArrayCompressionTool> &pulledMPACompressor,
                                                      const uint64_t iteration) {
        switch (mpaCompressionMethod) {
            using enum FrtMPACompressionMethod;
            case NO_COMPRESSION:
                return iteration;
            case LITTLE_COMPRESSION:
                return iterationToPulledTableIndex(mpaPeriod, iteration);
            case STRONGEST: {
                const uint64_t index = iterationToPulledTableIndex(mpaPeriod, iteration);
                return index == UINT64_MAX ? UINT64_MAX : ArrayCompressor::compress(pulledMPACompressor, index);
            }
            default:
                return iteration;
        }
    }


    template<Number Num>
    void MPATable<Num>::allocateWithCheckTableSize(const uint64_t index, const uint64_t levels) {
        auto &table = *tableManager->mpaTable;
        if (table.size() <= index) {
            throw vkh::exception_init("index out of range");
        }
        if (table[index].empty()) {
            table[index].reserve(levels);
        }
    }



    template<Number Num>
    PA<Num> *MPATable<Num>::lookup(const uint64_t refIteration, const complex<Num> dz) const {

        if (refIteration == 0 || mpaPeriod == nullptr) {
            return nullptr;
        }

        const uint64_t index = iterationToCompTableIndex(mpaSettings.mpaCompressionMethod, *mpaPeriod, pulledMPACompressor,
                                                         refIteration);

        if (index >= tableManager->mpaTable->size()) {
            return nullptr;
        }

        std::pmr::vector<PA<Num>> &table = (*this->tableManager->mpaTable)[index];
        if (table.empty()) {
            return nullptr;
        }

        const Num r = dz.norm_approx();

        switch (mpaSettings.mpaSelectionMethod) {
            using enum FrtMPASelectionMethod;
            case LOWEST: {
                PA<Num> *pa = nullptr;

                for (PA<Num> &test: table) {
                    if (test.isValid(r)) {
                        pa = &test;
                    } else return pa;
                }
                return pa;
            }
            case HIGHEST: {
                PA<Num> &pa = table.front();
                //This table cannot be empty because the pre-processing is done.

                if (!pa.isValid(r)) {
                    return nullptr;
                }

                for (uint64_t j = table.size(); j > 0; --j) {
                    PA<Num> &test = table[j - 1];
                    if (test.isValid(r)) {
                        return &test;
                    }
                }

                return &pa;
            }
            default: return nullptr;
        }
    }

    template<Number Num>
    size_t MPATable<Num>::getLength() {
        return tableManager && tableManager->mpaTable ? tableManager->mpaTable->size() : 0;
    }

    using LightMPATable = MPATable<double>;
    using DeepMPATable = MPATable<dex>;
} // namespace merutilm::rff2
