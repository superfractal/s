cmake_minimum_required(VERSION 3.31)
project(RFF)


if(NOT DEFINED ENV{MSYS2_ROOT})
    message(FATAL_ERROR "MSYS2_ROOT not set. Add the your Windows environment variable named MSYS2_ROOT.")
endif()

set(CLANG64_DIRS "$ENV{MSYS2_ROOT}/clang64")
set(CLANG64_LIBS "${CLANG64_DIRS}/lib")

# Modified by Fable 5 : prefer the CPU-tuned local GMP build when present.
# The MSYS2 fat-build libgmp.a does not recognize Zen 5 and dispatches the
# mul/sqr basecase to the slow generic C fallback (see extern/gmp-zen1/README.txt).
if(EXISTS "${CMAKE_SOURCE_DIR}/extern/gmp-zen1/lib/libgmp.a")
    set(RFF_GMP_LIB "${CMAKE_SOURCE_DIR}/extern/gmp-zen1/lib/libgmp.a")
    set(RFF_GMP_INCLUDE "${CMAKE_SOURCE_DIR}/extern/gmp-zen1/include")
    message(STATUS "GMP: using CPU-tuned build ${RFF_GMP_LIB}")
else()
    set(RFF_GMP_LIB "${CLANG64_LIBS}/libgmp.a")
    set(RFF_GMP_INCLUDE "${CLANG64_DIRS}/include")
    message(STATUS "GMP: using MSYS2 build ${RFF_GMP_LIB}")
endif()


set(CMAKE_CXX_STANDARD 20)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
set(CMAKE_RC_COMPILER ${CLANG64_DIRS}/bin/windres.exe)


add_executable(
        RFF
        res/resources.rc
        src/rff2/ui/Main.cpp
        src/rff2/io/RFFDynamicMapBinary.cpp
        src/rff2/parallel/ParallelRenderState.cpp
        src/rff2/mrthy/MPAPeriod.cpp
        src/rff2/ui/SettingsWindow.cpp
        src/rff2/ui/CallbackFile.cpp
        src/rff2/ui/CallbackFractal.cpp
        src/rff2/ui/CallbackRender.cpp
        src/rff2/ui/CallbackShader.cpp
        src/rff2/locator/MB2Locator.cpp
        src/rff2/ui/CallbackExplore.cpp
        src/rff2/preset/shader/palette/ShdPalettePresets.cpp
        src/rff2/preset/shader/fog/ShdFogPresets.cpp
        src/rff2/preset/shader/color/ShdColorPresets.cpp
        src/rff2/preset/shader/slope/ShdSlopePresets.cpp
        src/rff2/preset/shader/stripe/ShdStripePresets.cpp
        src/rff2/preset/shader/bloom/ShdBloomPresets.cpp
        src/rff2/preset/resolution/ResolutionPresets.cpp
        src/rff2/preset/render/RenderPresets.cpp
        src/rff2/preset/calc/CalculationPresets.cpp
        src/rff2/ui/CallbackVideo.cpp
        src/rff2/ui/IOUtilities.cpp
        src/rff2/io/RFFStaticMapBinary.cpp
        src/rff2/io/RFFBinary.cpp
        src/rff2/io/RFFLocationBinary.cpp
        src/rff2/io/KFRColorLoader.cpp
        extern/stb_image.c
        src/rff2/vulkan/GPCIterationPalette.cpp
        src/rff2/ui/AppRenderManager.cpp
        src/rff2/ui/AppLauncher.cpp
        src/rff2/vulkan/GPCStripe.cpp
        src/rff2/vulkan/GPCSlope.cpp
        src/rff2/vulkan/GPCColor.cpp
        src/rff2/vulkan/GPCFog.cpp
        src/rff2/vulkan/GPCDownsampleForBlur.cpp
        src/rff2/vulkan/GPCBloomThreshold.cpp
        src/rff2/vulkan/CPCBoxBlur.cpp
        src/rff2/vulkan/GPCBloom.cpp
        src/rff2/vulkan/GPCLinearInterpolation.cpp
        src/rff2/vulkan/GPCPresent.cpp
        src/rff2/ui/VideoWindow.cpp
        src/rff2/ui/VideoWindowRenderManager.cpp
        src/rff2/vulkan/CPC2MapIterationStripe.cpp
        src/rff2/vulkan/GPCStaticImage2Map.cpp
        src/rff2/vulkan/CPCImageRGBA2BGR.cpp
        src/rff2/settings/FrtReferenceSettings.hpp
        src/rff2/settings/FrtPerturbSettings.hpp
        src/rff2/settings/FrtGeneralSettings.hpp
        src/rff2/calc/calculatable.hpp
        src/rff2/calc/complex.hpp
)


target_compile_options(RFF PRIVATE
        #Debug options
        $<$<CONFIG:Debug>:-O0>
        #Release options
        $<$<CONFIG:Release>:-O3 -ffast-math -fno-associative-math -funroll-loops -march=native -mllvm -inline-threshold=4000>
)

target_compile_definitions(RFF
        PRIVATE
        VK_USE_PLATFORM_WIN32_KHR
)

add_subdirectory(src/vulkan_helper)


target_include_directories(RFF PRIVATE
        # GMP include dir (must precede clang64 include so the tuned build's gmp.h wins)
        ${RFF_GMP_INCLUDE}
        # extern sources
        ${CMAKE_SOURCE_DIR}/extern
        # clang include dirs
        ${CMAKE_SOURCE_DIR}/include
        # clang include dirs
        ${CLANG64_DIRS}/include
        # opencv4 include dirs
        ${CLANG64_DIRS}/include/opencv4
)

target_link_libraries(RFF PRIVATE
        # GMP Linker
        ${RFF_GMP_LIB}
        # Win32 Linker
        comctl32
        # OpenCV Linkers
        ${CLANG64_LIBS}/libopencv_core.dll.a
        ${CLANG64_LIBS}/libopencv_imgcodecs.dll.a
        ${CLANG64_LIBS}/libopencv_imgproc.dll.a
        ${CLANG64_LIBS}/libopencv_videoio.dll.a
        ${CLANG64_LIBS}/libopencv_highgui.dll.a
        # Vulkan Helper
        vulkan_helper
)


# Added by Fable 5 : benchmark of the MPA table creation strategies
add_executable(
        RFF_MPABench
        src/rff2/bench/MPATableBench.cpp
        src/rff2/mrthy/MPAPeriod.cpp
        src/rff2/parallel/ParallelRenderState.cpp
)

set_target_properties(RFF_MPABench PROPERTIES CXX_SCAN_FOR_MODULES OFF)

target_compile_options(RFF_MPABench PRIVATE
        $<$<CONFIG:Debug>:-O0>
        $<$<CONFIG:Release>:-O3 -ffast-math -fno-associative-math -funroll-loops -march=native -mllvm -inline-threshold=4000>
)

find_package(Vulkan REQUIRED)

target_include_directories(RFF_MPABench PRIVATE
        ${RFF_GMP_INCLUDE}
        ${CMAKE_SOURCE_DIR}/extern
        ${CMAKE_SOURCE_DIR}/include
        ${CLANG64_DIRS}/include
        ${Vulkan_INCLUDE_DIRS}
)

target_link_libraries(RFF_MPABench PRIVATE
        ${RFF_GMP_LIB}
)


macro(RFF_DEFINITION def str flag)
    option(${def} ${str} ${flag})
    if(${def})
        message(STATUS "${def} is Enabled. ${str}")
        target_compile_definitions(RFF PUBLIC ${def})
    else()
        message(STATUS "${def} is Disabled. Skipping")
    endif ()
endmacro()

RFF_DEFINITION(SAFE_DEX_OPERATOR "Better Quality, Slower Speed for Deep Zoom" OFF)
