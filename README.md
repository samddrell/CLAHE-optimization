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

MetaVi Labs builds microscope software for cell tracking and analysis. As the software displays the images to the user, a series of corrections must be applied to the image. One crucial step in the correction process is contrast enhancement, to provide a clear image with dark cells that pop off of the light background. The contrast algorithm that MetaVi Labs has chosen to utilize is the CLAHE (Contrast-Limited Adaptive Histogram Equalization), which provides an even contrast across the image, with various inputs that allow the contrast and its application to be tuned. CLAHE boosts local contrast without over-amplifying noise.

The Clahe algorithm was chosen for a variety of factors, but partly because the development team at MetaVi labs was able to build their own, proprietary, CLAHE algorithm that could be accelerated via CUDA hardware acceleration. A CUDA path enables near-interactive viewing on supported devices. This project seeks to answer the question: how much faster is MetaVi Lab's proprietary CUDA algorithm when compared to prebuilt, already-optimized, CLAHE algorithms that run on the CPU. In order to fully understand the application-time difference between the CPU and CUDA, the CLAHE algorithms were applied on multiple devices, demonstrating which devices would be most useful for this algorithm. It ought to be noted that the CLAHE algorithm used in this test is not MetaVi's final production algorithm. Rather, the algorithm tested is the prototype program that was characterized as a proof-of-concept program.

This project benchmarks a CUDA implementation of CLAHE to:

- **Quantify speed** across several devices (Jetsons and desktop GPUs),
- **Compare** against OpenCV’s CPU CLAHE,
- and **Explore kernel-level optimizations** to understand whether high-end hardware costs are justified by measurable latency improvements.

---

## Background

### What CLAHE does (and why)
CLAHE is a multi-step process of several key steps. It begins by dividing an image into equal-sized tiles, and it clips each tile’s histogram to a limit (to avoid over-contrast). The excess is then reapplied to histogram, and then the histogram is scaled based off of the histogram's Cummulative Distribution Function. The equalized values are then stored in a LUT (one per tile) to optimize value accessing speeds. Lastly, the whole image is bilinearly interpolated to smooth over tile seams, creating one, cohesive, image.


### OpenCV vs. proprietary flow
Below, we show side-by-side examples using:
**OpenCV CLAHE (CPU)**
and
 **Proprietary/experimental CUDA CLAHE** (this repo).
Two comparisons are shown; the second image below shows the same comparison, only zoomed in.

It may be seen that, by developing a proprietary algorithm, the MetaVi developers where able to fully control how the CLAHE algorithm would be applied. Note how the proprietary CLAHE algorithm is more true to the original image: the background reflects the original with minimal difference, and no lines are overly bold, but the visibility range between each cell is compressed, creating a more even look. In this application, CLAHE functions as a sharpening tool, which is incredibly advantageous when photographing such small subjects, as the difference between focal planes is amplified due to the scale of the cells.

_OpenCV vs Proprietary Algorithms:_
![OpenCV CLAHE](images_for_readme/compare_cuda_cpu_clahe.png)
![OpenCV CLAHE](images_for_readme/compare_cuda_cpu_clahe_zoomed.png)


### Why CLAHE (for our use-case)
To demonstrate why contast adjustment, visually, is the right strategy to use, we compare CLAHE with Canny Edge Detection. Canny Edge Detection may seem like an attractive option to clearly present harsh lines on a light background to the viewer, but, as seen below, edge detection performs a very different role from contrast adjustment, even in a high-contrast, black-and-white, image.
Canny edge detection can look appealing on greyscale imagery, but it extracts edges, not contrast, preventing it from being a drop-in for improving overall visibility.

_Canny Applied:_
![Canny vs CLAHE](images_for_readme/canny_applied.png)

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

In the first part of this expiriement, the application speed of the OpenCV CPU and MetaVi CUDA CLAHE algorithms were compared across four devices and four image sizes. Each test consisted of 10 trials of each method of CLAHE application, on each image size, on each device. The following four devices were compared: 

- Jetson AGX Orin - _Chosen due to it's current use in MetaVi projects._
- Jetson Orin Nano - _Chosen to perform value-cost analysis of the CUDA hardware built into the Jetson Orin line of products._
- RTX 4090 - _Chosen due to it's current use in MetaVi projects._
- RTX 2000 Ada - _Chosen due to it's current use in MetaVi projects._

The following four image sizes were chosen to compare processing speeds on different sized images:
			
- 3300x2200 - _Common JPG image resolution._
- 4096x3000 - _Chosen becuase it is the standard size processed by MetaVi's CLAHE algorithm._
- 8192x6000 - _Double the size of the standard._
- 16384x12000 - _Four times the size of the standard._

### Orin Nano vs AGX Orin Data, with the respective device CPUs


**Jetson Orin Nano**

| Trial | CUDA 3300x2200 (ms) | CUDA 4096x3000 (ms) | CUDA 8192x6000 (ms) | CUDA 16384x12000 (ms) | CPU 3300x2200 (µs) | CPU 4096x3000 (µs) | CPU 8192x6000 (µs) | CPU 16384x12000 (µs) |
|-------|----------------------|----------------------|----------------------|------------------------|--------------------|--------------------|--------------------|----------------------|
| 1     | 24.988               | 40.053               | 156.250              | 634.777                | 12.483             | 20.587             | 74.439             | 287.939              |
| 2     | 24.645               | 39.463               | 155.154              | 692.727                | 13.400             | 24.089             | 73.616             | 287.820              |
| 3     | 23.621               | 36.801               | 146.991              | 651.617                | 12.907             | 21.224             | 70.811             | 280.615              |
| 4     | 24.718               | 39.671               | 154.472              | 614.681                | 14.578             | 23.938             | 73.198             | 281.745              |
| 5     | 24.711               | 39.554               | 154.500              | 614.396                | 14.122             | 24.097             | 73.688             | 286.192              |
| **Avg** | **24.537**            | **39.108**            | **153.473**           | **641.640**             | **13.498**         | **22.787**         | **73.150**         | **284.862**          |

**Jetson Orin AGX**

| Trial | CUDA 3300x2200 (ms) | CUDA 4096x3000 (ms) | CUDA 8192x6000 (ms) | CUDA 16384x12000 (ms) | CPU 3300x2200 (ms) | CPU 4096x3000 (ms) | CPU 8192x6000 (ms) | CPU 16384x12000 (ms) |
|-------|----------------------|----------------------|----------------------|------------------------|--------------------|--------------------|--------------------|----------------------|
| 1     | 35.250               | 58.574               | 181.751              | 607.607                | 9.391              | 14.382             | 46.216             | 178.489              |
| 2     | 35.044               | 55.793               | 177.962              | 601.058                | 9.223              | 14.683             | 46.174             | 178.111              |
| 3     | 35.805               | 77.351               | 167.430              | 607.257                | 8.954              | 14.610             | 46.501             | 178.085              |
| 4     | 34.438               | 57.565               | 189.288              | 607.231                | 9.547              | 15.343             | 46.465             | 177.628              |
| 5     | 35.071               | 77.221               | 293.269              | 1164.869               | 10.731             | 17.252             | 46.978             | 177.826              |
| **Avg** | **35.122**            | **65.301**            | **201.940**           | **717.604**             | **9.569**          | **15.254**         | **46.467**         | **178.028**          |

_CLAHE Application rates on the Jetson Orin Nano and AGX, with their respective CPU applications:_
![AGX vs Nano](images_for_readme/JONanoVSAGX.png)


### ADA 2000 vs RTX 4090 Data, with the respective device CPUs

**RTX 2000 Ada**

| Trial | CUDA 3300x2200 (ms) | CUDA 4096x3000 (ms) | CUDA 8192x6000 (ms) | CUDA 16384x12000 (ms) | CPU 3300x2200 (ms) | CPU 4096x3000 (ms) | CPU 8192x6000 (ms) | CPU 16384x12000 (ms) |
|-------|----------------------|----------------------|----------------------|------------------------|--------------------|--------------------|--------------------|----------------------|
| 1     | 9.838                | 12.761               | 51.899               | 219.524                | 9.781              | 12.082             | 17.014             | 58.816               |
| 2     | 9.208                | 13.160               | 55.709               | 213.811                | 7.705              | 10.852             | 16.810             | 57.891               |
| 3     | 9.849                | 12.763               | 48.105               | 189.864                | 7.708              | 10.879             | 17.025             | 59.809               |
| 4     | 9.527                | 13.316               | 52.256               | 214.402                | 7.849              | 11.354             | 17.634             | 65.781               |
| 5     | 9.200                | 13.404               | 52.220               | 205.363                | 7.647              | 10.912             | 16.473             | 59.866               |
| **Avg** | **9.524**             | **13.081**            | **52.038**            | **208.593**             | **8.138**          | **11.216**         | **16.991**         | **60.433**           |

**RTX 4090**

| Trial | CUDA 3300x2200 (ms) | CUDA 4096x3000 (ms) | CUDA 8192x6000 (ms) | CUDA 16384x12000 (ms) | CPU 3300x2200 (ms) | CPU 4096x3000 (ms) | CPU 8192x6000 (ms) | CPU 16384x12000 (ms) |
|-------|----------------------|----------------------|----------------------|------------------------|--------------------|--------------------|--------------------|----------------------|
| 1     | 4.576                | 7.318                | 28.500               | 110.628                | 6.175              | 9.934              | 20.041             | 77.278               |
| 2     | 4.583                | 7.346                | 28.598               | 109.548                | 6.027              | 9.938              | 19.329             | 77.253               |
| 3     | 4.586                | 7.905                | 28.470               | 110.167                | 6.001              | 9.804              | 19.393             | 77.412               |
| 4     | 4.699                | 7.349                | 28.733               | 110.323                | 5.967              | 9.857              | 19.792             | 77.524               |
| 5     | 4.567                | 7.294                | 86.153               | 109.501                | 6.053              | 9.890              | 19.932             | 77.383               |
| **Avg** | **4.602**             | **7.442**             | **40.091**            | **110.033**             | **6.045**          | **9.885**          | **19.697**         | **77.370**           |


_CLAHE Application rates on the RTX4090 and ADA 2000, with their respective CPU applications:_
![4090 vs 2000](images_for_readme/rtx4090v2000.png)


### Threading

Of course, multithreading is utilized to increase CLAHE algorithm computation speed. Threading, as it is utilized by the CPU-computed OpenCV implementation, depends on your OpenCV build and OMP/TBB.

MetaVi's proprietary CUDA algorithm, however, is built on multithreading. The implementation of multithreading changes, depending on which part of the CLAHE process is being executed. During the tiling and histogram clipping and redistrobuting stages of the CLAHE process, Each thread block is associated with one tile, matching one thread per pixel in the block. When CLAHE itself is applied, each thread continues to match to one pixel.


## Part 2 — Optimization experiments

_All optimizations below were prototype kernels generated/assisted with AI to explore directions—not drop-in production code. Several trade-offs and bugs were found (details below)._

After learning that the proprietary CUDA CLAHE algorithm was actually slower than the OpenCV, CPU-driven, counterpart, a few changes were tried and measured to see if the proprietary algorithm's application speed could be increased. This process began by measuring the time of each kernal seperately, and it was found that the CUDA application time (kernal 2) was much slower than the histogram clipping and LUT-building section (kernal 1). The changes made to each kernal are described below.

### Summary of Kernal 1 Changes(% change vs. baseline):

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

### Summary of Kernal 2 Changes(% change vs. baseline):

### Kernal Explanations:

**originalComputeTileLUT**

Baseline CLAHE LUT builder: one tile per block, a single shared 256-bin histogram updated with shared-mem atomicAdd, followed by clip + redistribute, then a single-thread CDF → LUT pass. Simple and reliable, but heavy atomic contention on hot bins can limit speed. (If your historical version used 255 - (…) in the mapping, that inverts contrast—keep mapping consistent across variants.)

**computeTileLUT**

Same algorithm as the baseline, but all image/LUT pointers are marked __restrict__ to promise no aliasing. This often lets the compiler issue more efficient read-only loads and slightly better scheduling. Expect modest speedups with identical output (assuming the same mapping convention).

**fasterComputeTileLUT_opt**

Uses one histogram per warp in shared memory (WARPS×256), so warps update private bins with far fewer conflicts; then it reduces per-warp histograms into a final 256-bin array before the usual clip + redistribute + CDF/LUT. This can cut atomic contention dramatically, but the extra shared memory and the reduction step can reduce occupancy and even hurt performance on memory-constrained devices if tiles aren’t very contentious.

**test_A**

Implements warp-aggregated atomics: threads in a warp first group identical pixel values using __match_any_sync, and only the leader lane performs a single atomicAdd with the group’s vote count. This slashes atomics when neighboring pixels repeat (typical natural images), but adds warp-wide mask/shuffle overhead and brings little benefit on random/noisy tiles. LUT build is the standard clip → CDF → LUT path.

**test_C**

Accelerates memory access with vectorized loads (uchar4), so each thread processes 4 adjacent pixels per loop. This improves coalescing and cuts address arithmetic, then falls back to a short scalar tail. It’s alignment-sensitive—without a quick alignment prologue, misaligned uchar4 loads may underperform or be suboptimal on some GPUs. Otherwise the histogram and LUT steps are standard.

**applyCLAHE**

The apply stage: per pixel, find the surrounding 2×2 tiles, fetch the mapped values from each tile’s LUT at the input intensity, and write the bilinear interpolation. Work is embarrassingly parallel and largely memory-bound; keep accesses coalesced (blockDim.x multiple of 32), consider an interior-only fast path (no clamps) and optional shared-LUT staging when blocks align to tiles.

**hist_vec_warpagg**

A hybrid “kitchen-sink” fast path: first a short alignment prologue (scalar) to reach 4-byte alignment; then vectorized uchar4 loads for the main body; and for each byte, warp-aggregated atomics to minimize histogram updates. Finally clip → redistribute → CDF/LUT. Often best on desktop GPUs (e.g., Ada/4090) with high bandwidth and cheap warp ops; can be neutral or negative on Jetsons if register pressure lowers occupancy.


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