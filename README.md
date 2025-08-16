# CUDA CLAHE — Benchmarking & Optimization Notes

Contrast Limited Adaptive Histogram Equalization (CLAHE) accelerated with CUDA, plus a set of micro-benchmarks and experiments exploring performance trade-offs on different GPUs/Jetsons.

---

## Table of contents
- [Introduction](#introduction)
- [Background](#background)
- [Build & run](#build--run)
- [Part 1 — Device benchmarking](#part-1--device-benchmarking)
- [Part 2 — Optimization experiments](#part-2--optimization-experiments)
- [Results gallery](#results-gallery)
- [Reproducing measurements](#reproducing-measurements)
- [Notes on AI usage](#notes-on-ai-usage)
- [License](#license)
- [Appendix — Tuning tips](#appendix--tuning-tips)

---

## Introduction

MetaVi Labs builds microscope software for cell tracking and analysis. A key first step for users is simply seeing what was captured—fast, reliable contrast enhancement helps.

This project benchmarks a CUDA implementation of CLAHE to:

- **quantify speed** across several devices (Jetsons and desktop GPUs),
- **compare** against OpenCV’s CPU CLAHE,
- and **explore kernel-level optimizations** to understand whether high-end hardware costs are justified by measurable latency improvements.

#### Why CUDA CLAHE?
CLAHE boosts local contrast without over-amplifying noise. A CUDA path enables near-interactive viewing on supported devices.

---

## Background

### What CLAHE does (and why)
CLAHE divides an image into tiles, clips each tile’s histogram to a limit (to avoid over-contrast), builds a per-tile LUT, then bilinearly interpolates between LUTs to avoid tile seams.

### OpenCV vs. proprietary flow
We show side-by-side examples using:

- **OpenCV CLAHE (CPU)**
- **Proprietary/experimental CUDA CLAHE** (this repo)

_Add your figures here:_
![OpenCV CLAHE](docs/images/opencv_clahe.png)
![CUDA CLAHE](docs/images/cuda_clahe.png)

### Why not Canny (for our use-case)
Canny edge detection can look appealing on greyscale imagery, but it extracts edges—**not contrast**. It’s not a drop-in for improving overall visibility.

_Add comparison:_
![Canny vs CLAHE](docs/images/canny_vs_clahe.png)

---

## Build & run

### Requirements
- CUDA toolkit (matching your device)
- CMake ≥ 3.16
- A recent C++ compiler
- OpenCV (core, imgproc, imgcodecs, highgui, videoio)
- (Optional) libpng / libjpeg if your build uses them

### Configure & build
```bash
# From repo root
mkdir -p build && cd build
cmake ..           # add -DCMAKE_BUILD_TYPE=Release for optimized builds
cmake --build . -j
```

### Run
```bash

./test
# The binary prints four CUDA timings (in ms) and four CPU timings (in µs) per run.
```

---

## Part 1 — Device benchmarking

We compare four devices:

- Jetson AGX Orin

- Jetson Orin Nano

- RTX 4090

- RTX 2000 Ada

We also include OpenCV CPU timings for reference.

### Threading

CPU path uses OpenCV’s implementation (multi-threading depends on your OpenCV build and OMP/TBB).

CUDA path uses one block per tile; kernel configuration explained below.

_Insert your summary table (replace “—” with your values or link a CSV):_

_Device	CUDA Time @ Size A	CUDA Time @ Size B	CUDA Time @ Size C	CUDA Time @ Size D	OpenCV CPU (ms)
Jetson AGX Orin	—	—	—	—	—
Jetson Orin Nano	—	—	—	—	—
RTX 4090	—	—	—	—	—
RTX 2000 Ada	—	—	—	—	—_

_If you keep your raw numbers in a CSV, link it here:
See: docs/data/benchmarks.csv_

---


## Part 2 — Optimization experiments

_All optimizations below were prototype kernels generated/assisted with AI to explore directions—not drop-in production code. Several trade-offs and bugs were found (details below)._

### Summary (% change vs. your baseline):

- **Histogram Per Warp: −5.8%** (slower)
Likely due to extra shared-mem footprint and reduction overhead outweighing fewer collisions.

- **__restrict__ qualifiers:** +4.8%
Helps the compiler generate better memory code when aliasing is removed.

- ** Minimizing atomics (warp-aggregated updates):** +11.9%
Reduced contention can help—but note a color inversion bug was observed (LUT mapping issue).

- **Vectorized loads (uchar4):** +11.2%
Better global load efficiency—but again a color inversion bug noted (alignment/mapping mistake).

- **All three combined:** +5.4%
Gains did not add linearly; register pressure, shared-mem usage, and control overhead can interact.

### Why combined can be worse than parts

- **Register pressure** increases with more logic → lower occupancy.

- **Shared memory grows** (per-warp histograms) → fewer concurrent blocks.

- **Instruction mix:** shuffle/match/reduction + vectorization + extra blending math may stall pipelines.

- **Contention pattern:** some techniques reduce atomics in one phase but introduce reductions later.

- **Bugs** (e.g., inverted mapping) can force extra fix-up passes.

---

## Results gallery

_Replace with your actual images. Keep originals and processed side-by-sides consistent in size and naming._

_docs/images/
├─ input/
│  ├─ sample1.png
│  └─ sample2.png
├─ opencv_clahe/
│  ├─ sample1.png
│  └─ sample2.png
└─ cuda_clahe/
   ├─ sample1_baseline.png
   ├─ sample1_opt_minAtomics.png
   └─ sample1_opt_vecLoads.png_


**Example markdown:**

**Sample 1**
| Input | OpenCV CLAHE (CPU) | CUDA CLAHE (baseline) | CUDA (vectorized+warp agg) |
|------:|:-------------------:|:---------------------:|:---------------------------:|
| ![in](docs/images/input/sample1.png) | ![cpu](docs/images/opencv_clahe/sample1.png) | ![cuda](docs/images/cuda_clahe/sample1_baseline.png) | ![opt](docs/images/cuda_clahe/sample1_opt_vecLoads.png) |

---

## Reproducing measurements

- Build as above (prefer -DCMAKE_BUILD_TYPE=Release).

- Run ./test multiple times per device. Each run prints four CUDA times (ms) and four CPU times (µs).

- Collect logs and parse into CSV (one row per trial). Keep image size, tile size, and clip limit fixed during a batch.

- Report mean ± stddev per device/setting.
(Tip: record GPU clocks/power mode on Jetsons.)

## What we time

- CUDA: kernel timing via cudaEventRecord around the CLAHE kernel sections.

- CPU: std::chrono::high_resolution_clock around OpenCV’s clahe->apply().

- If you change what’s inside the timing scope (e.g., include H2D/D2H copies), note that clearly in the results.


___

## Notes on AI usage

This project primarily benchmarks an existing CUDA CLAHE prototype and explores optimizations. Because the work was evaluation-oriented, we used AI-generated code fragments to:

- draft alternative kernels quickly,

- sketch variants (warp-aggregated atomics, vectorized loads, etc.),

- iterate on instrumentation.

**Important:** these kernels are experimental. They are not production-ready and may have edge-case bugs (e.g., LUT mapping inversions if alignment or CDF handling is off). Treat them as starting points for manual tuning.

___

## License

Source files contain license headers describing usage restrictions (MetaVi Labs ↔ ibidi GmbH).

_If you add a LICENSE file, reference it here. Otherwise, keep code headers authoritative._

___

## Appendix — Tuning tips (quick checklist)

- Build with -O3 -DNDEBUG; avoid mixed Debug/Release dependencies.

- Verify tile area normalization for border tiles (prevents vignette).

- If you see darker output: check the CDF mapping formula and whether it’s inadvertently inverted; confirm LUT interpolation is bilinear and indices are clamped.

- When testing vectorized loads:

    - ensure alignment or guard unaligned prologue/tail,

    - confirm the same sampling order (no byte/endianness surprises).

- Measure one change at a time; log GPU clocks/temperature; pin power mode on Jetsons.