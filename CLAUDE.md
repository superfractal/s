# Project Overview

**RFF-2.0** is a Fast Mandelbrot set rendering application. It is built C++20, Vulkan API, GMP library, etc.

- **Fast-period-guessing (FPG):** Automatically determines the longest period of the selected location.
- **Multilevel Periodic Approximation (MPA):** Skipping directly to the periodic point. **This is NOT BLA (Bivariate Linear Approximation).** MPA derives its skip levels from the reference orbit's *periodic structure* (see `MPAPeriod`), not from per-iteration reference magnitude as BLA does. Skip lengths are variable and tied to period levels. Do not refer to it as BLA.
- **Reference Compression:** Allows rendering of extremely long periods (1B+) by compressing the approximation tables, heavily reducing RAM usage and skipping massive table creation processes.
- **Perturbation Theory:** Implements both Light and Deep perturbators to calculate pixel values around a high-precision reference orbit.

## Project Structure

**Core Architecture (`src/rff2/`):**

- `calc/` - Math primitives: complex numbers, double-exponent math (`dex`), fixed-point arithmetic, thread pool (`spin_thread_pool`).
- `constants/` - Compile-time constants for fractals, files, video, Vulkan, Win32, and status codes.
- `data/` - Shared data structures: `ApproxTableManager`, `GraphicsMatrixStagingBuffer`, `Matrix`.
- `formula/` - Fractal perturbation math: `MB2Perturbator`, `MB2Reference`,.
- `io/` - Binary file I/O: KFR color palettes, RFF maps (static/dynamic), location files.
- `locator/` - Minibrot finding algorithm (`MB2Locator`).
- `mem/` - Custom memory resources (`single_alloc_permitted_memory_resource`).
- `mrthy/` - Multilevel Periodic Approximation: `MPAPeriod`, `MPATable`, `ArrayCompressor`, `PAGenerator`.
- `parallel/` - Multi-threaded render management: `BackgroundThreads`, `ParallelDispatcher`, `ParallelRenderState`.
- `preset/` - Built-in presets for calculation, rendering, resolution, and all shader types.
- `settings/` - All settings structs: fractal (`Frt*`), render (`Render*`), shader (`Shd*`), video (`Vid*`).
- `ui/` - Win32 UI: `AppLauncher`, `AppRenderManager`, per-feature `Callback*` handlers, `VideoWindow`, `SettingsWindow`.
- `util/` - Utilities: `ColorUtils`, `profiler`.
- `vulkan/` - Vulkan pipeline objects: `GPC*` (graphics), `CPC*` (compute), `RCC*` (render command chains).

**Vulkan Framework (`src/vulkan_helper/`):**

Object-oriented Vulkan abstraction layer, organized into:

- `core/` - Instance, logical/physical device, validation layers.
- `engine/` - Subsystems: `buffer/`, `cmd/`, `configurator/`, `context/`, `descriptor/`, `executor/`, `graphics/`, `manage/`, `pipeline/`, `repo/`, `sampler/`, `sync/`, `window/`.
- `util/` - Barrier helpers, buffer/image utils, descriptor updater, swapchain utils.

**Assets & Shaders:**

- `shdsrc/` - GLSL shader files (color, bloom, fog, slope, stripe, box-blur, palette, resample, post-process).

## Development Philosophy & Guidelines

**1. Precision and Correctness:**

- Perturbation theory equations are highly sensitive to floating-point precision drift.

**2. Memory Management:**

- Array compressors are heavily used because storing millions of iterations in memory is prohibitive.
- verifying that rendering remains crisp and free of pixelation or visual glitches.

**3. Vulkan Integration:**

- Descriptor sets, push constants, and pipeline layouts must be precisely matched between the C++ pipeline configurator and the GLSL shaders.

## Workflow for AI Assistance

- **Performance Optimizations:** Follow the guidelines below carefully. *Never* precision for speed.
- **Code Modification Tracking:** Whenever you (the AI) modify or add code,  you MUST add it on the line immediately below the "Created by" comment near the top of the file. For example: `// Modified by Fable 5`.
- **Series Approximation (SA) is currently under active development.** Do NOT modify any SA-related code (`SeriesApproximationData`, `FrtSASettings`, or any SA logic) unless explicitly instructed by the user.
