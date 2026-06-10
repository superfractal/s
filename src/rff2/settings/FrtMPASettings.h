//
// Created by Merutilm on 2025-05-04.
// Modified by Fable 5
//

#pragma once
#include <cstdint>

#include "FrtMPACompressionMethod.h"
#include "FrtMPASelectionMethod.h"


namespace merutilm::rff2 {
    struct FrtMPASettings final {
        uint16_t minSkipReference;
        uint8_t maxMultiplierBetweenLevel;
        float epsilonPower;
        FrtMPASelectionMethod mpaSelectionMethod;
        FrtMPACompressionMethod mpaCompressionMethod;
        bool useParallelTableGeneration;
        bool useMergedTableGeneration;
    };
}