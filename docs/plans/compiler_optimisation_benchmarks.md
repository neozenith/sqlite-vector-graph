# Compiler Optimisation Benchmarks

Captured: 2026-02-22.

**Status:** Planning. No code changes yet.

**Nature:** Exploratory / throwaway. This benchmark exists to find the optimal compiler flags and compute backend for muninn's release build. Once the winner is identified, the results inform the default `CFLAGS`, llama.cpp build configuration, and the benchmark artifacts can be discarded.

---

## Table of Contents

1. [Motivation](#1-motivation)
2. [Build Matrix](#2-build-matrix)
3. [LTO Tiers Explained](#3-lto-tiers-explained)
4. [Compute Backends Explained](#4-compute-backends-explained)
5. [Naming Convention](#5-naming-convention)
6. [Makefile Integration](#6-makefile-integration)
7. [Phase 0 — Build & Measure](#7-phase-0--build--measure)
8. [Phase 1 — VSS Benchmarks](#8-phase-1--vss-benchmarks)
9. [JSONL Schema](#9-jsonl-schema)
10. [Analysis & Charting](#10-analysis--charting)
11. [Implementation Steps](#11-implementation-steps)
12. [Open Questions](#12-open-questions)

---

## 1. Motivation

The default muninn build uses `-O2` with no link-time optimisation and CPU-only inference (Metal GPU disabled). This is a safe, well-understood baseline — but for a computation-heavy extension doing SIMD distance calculations, HNSW graph traversal, and llama.cpp inference, there may be meaningful performance differences across optimisation levels and compute backends.

The goal is to **quantify the impact** of compiler optimisation and hardware acceleration on muninn's core VSS hot paths, producing data-driven guidance for what to ship as the default release build.

### Why These Flags

| Flag | Rationale |
|------|-----------|
| `-O2` | Current default. Baseline for all comparisons. |
| `-O3` | Enables auto-vectorisation and aggressive inlining — directly relevant to `vec_math.c` distance loops and `hnsw_algo.c` beam search. |
| `-Os` | Optimises for binary size. Often surprisingly fast due to better L1 instruction cache pressure. Apple uses `-Os` for much of macOS. |
| `-flto` | Link-time optimisation allows cross-TU inlining. Critical path: `hnsw_vtab.c` -> `hnsw_algo.c` -> `vec_math.c` are separate translation units — LTO can inline distance functions into hot search loops. |

### Why NOT These Flags

| Flag | Why excluded |
|------|-------------|
| `-O1` | Rarely interesting — either you want debug-friendly (`-O0`) or real optimisation (`-O2+`). |
| `-Ofast` | Enables `-ffast-math` which breaks IEEE 754 compliance. Distance calculations (L2, cosine, inner product) could produce different results, breaking correctness tests. Too dangerous for a library. |
| `-Oz` | Aggressive size reduction, but unlikely to beat `-Os` on performance. Could be added later if binary size becomes a shipping concern. |

---

## 2. Build Matrix

Eighteen variants covering all permutations of `{O2, O3, Os}` x `{LTOOFF, LTOMUNINN, LTOFULL}` x `{CPU, METAL}`:

### CPU Variants

| # | `-O` level | LTO tier | Backend | Suffix | Notes |
|---|-----------|----------|---------|--------|-------|
| 1 | `-O2` | off | CPU | `O2_LTOOFF_CPU` | **Baseline** — current default build |
| 2 | `-O2` | muninn-only | CPU | `O2_LTOMUNINN_CPU` | LTO across muninn `.c` files; llama.cpp linked as regular `.a` |
| 3 | `-O2` | full | CPU | `O2_LTOFULL_CPU` | LTO across everything (llama.cpp rebuilt with `-flto`) |
| 4 | `-O3` | off | CPU | `O3_LTOOFF_CPU` | Aggressive optimisation, no LTO |
| 5 | `-O3` | muninn-only | CPU | `O3_LTOMUNINN_CPU` | Aggressive + partial LTO |
| 6 | `-O3` | full | CPU | `O3_LTOFULL_CPU` | Maximum optimisation, CPU-only |
| 7 | `-Os` | off | CPU | `Os_LTOOFF_CPU` | Size-optimised, no LTO |
| 8 | `-Os` | muninn-only | CPU | `Os_LTOMUNINN_CPU` | Size-optimised + partial LTO |
| 9 | `-Os` | full | CPU | `Os_LTOFULL_CPU` | Size-optimised + full LTO |

### Metal Variants

| # | `-O` level | LTO tier | Backend | Suffix | Notes |
|---|-----------|----------|---------|--------|-------|
| 10 | `-O2` | off | Metal | `O2_LTOOFF_METAL` | Baseline flags + Metal GPU offload |
| 11 | `-O2` | muninn-only | Metal | `O2_LTOMUNINN_METAL` | Partial LTO + Metal |
| 12 | `-O2` | full | Metal | `O2_LTOFULL_METAL` | Full LTO + Metal |
| 13 | `-O3` | off | Metal | `O3_LTOOFF_METAL` | Aggressive + Metal |
| 14 | `-O3` | muninn-only | Metal | `O3_LTOMUNINN_METAL` | Aggressive + partial LTO + Metal |
| 15 | `-O3` | full | Metal | `O3_LTOFULL_METAL` | Maximum optimisation + Metal |
| 16 | `-Os` | off | Metal | `Os_LTOOFF_METAL` | Size-optimised + Metal |
| 17 | `-Os` | muninn-only | Metal | `Os_LTOMUNINN_METAL` | Size-optimised + partial LTO + Metal |
| 18 | `-Os` | full | Metal | `Os_LTOFULL_METAL` | Size-optimised + full LTO + Metal |

All eighteen variants are built with the naming convention to keep the benchmark harness uniform — no special-casing.

---

## 3. LTO Tiers Explained

The key insight is that LTO has **two boundaries** in this project, not one:

```
┌─────────────────────────────────────────────────────────────┐
│  muninn source (.c files)                                   │
│                                                             │
│  hnsw_vtab.c ──> hnsw_algo.c ──> vec_math.c                │
│  graph_tvf.c ──> graph_load.c                               │
│  embed_gguf.c ──┐                                           │
│                 │  ← boundary 1 (muninn ↔ llama.cpp)        │
├─────────────────┼───────────────────────────────────────────┤
│                 ▼                                           │
│  libllama.a  libggml.a  libggml-base.a  libggml-cpu.a      │
│  [+ libggml-metal.a  libggml-blas.a  for Metal builds]     │
│  (pre-built static libraries from vendor/llama.cpp)         │
└─────────────────────────────────────────────────────────────┘
```

| Tier | Name | What gets LTO'd | llama.cpp build |
|------|------|-----------------|-----------------|
| **LTOOFF** | No LTO | Nothing. Each `.o` is optimised in isolation. | Standard (`vendor/llama.cpp/build/` or `build-metal/`) |
| **LTOMUNINN** | Muninn-only LTO | Muninn's `.c` files are compiled to LLVM bitcode and optimised together at link time. llama.cpp `.a` files contain regular machine code — LTO cannot see into them. | Standard (`vendor/llama.cpp/build/` or `build-metal/`) |
| **LTOFULL** | Full LTO | Everything. llama.cpp is rebuilt with `-flto` so its `.a` files contain LLVM bitcode too. The linker optimises across the entire codebase as one unit. | LTO build (`vendor/llama.cpp/build-lto/` or `build-metal-lto/`) |

### Why Three Tiers Matter

- **LTOOFF vs LTOMUNINN** isolates the value of cross-TU inlining within muninn's own code. The hot path `hnsw_search -> vec_l2_distance` spans three `.c` files — this is where LTOMUNINN should shine.
- **LTOMUNINN vs LTOFULL** isolates the value of LTO *across the llama.cpp boundary*. This only matters for the embedding speed metric. If there's no meaningful difference, we can skip the llama.cpp LTO rebuild and save significant build complexity.

### llama.cpp LTO Builds

LTOFULL variants require a separate llama.cpp build with `-flto` flags. There are two LTO builds — one per backend:

```makefile
# CPU + LTO
LLAMA_BUILD_LTO = $(LLAMA_DIR)/build-lto
# Metal + LTO
LLAMA_BUILD_METAL_LTO = $(LLAMA_DIR)/build-metal-lto
```

Each produces static libraries containing LLVM bitcode instead of machine code. LTOOFF and LTOMUNINN variants link against the standard build; LTOFULL variants link against the corresponding LTO build.

---

## 4. Compute Backends Explained

The current build explicitly disables GPU acceleration (`-DGGML_METAL=OFF`). On Apple Silicon, llama.cpp supports Metal compute shaders that offload tensor operations (matrix multiplications, attention, layer norms) to the GPU.

### What's Already Active

**Accelerate BLAS** is already linked in the current build. The `libggml-blas.a` library uses Apple's `vecLib`/`vDSP` for SGEMM/SGEMV operations. This means CPU builds are not purely "naive" — they already benefit from Apple's SIMD-optimised linear algebra via the Accelerate framework.

### What Metal Adds

Metal offloads entire transformer layers to the Apple Silicon GPU via compute shaders. For embedding models like MiniLM (6 transformer layers) or Nomic-embed (12 layers), this means:

1. **Matrix multiplications** — the dominant cost in transformer inference — run on GPU compute units
2. **Attention computation** — parallelised across GPU threads
3. **Layer norms and activations** — fused into Metal compute kernels

The expected speedup depends on model size. Small models (MiniLM, ~22M params) may see modest gains since the GPU dispatch overhead can dominate. Larger models (Nomic-embed-text, ~137M params) should see more significant acceleration.

### Four llama.cpp Builds

The backend x LTO dimensions are orthogonal, producing four distinct llama.cpp builds:

| Build | Directory | CMake flags | Used by |
|-------|-----------|-------------|---------|
| **CPU** | `build/` | `GGML_METAL=OFF` | CPU + LTOOFF/LTOMUNINN |
| **CPU + LTO** | `build-lto/` | `GGML_METAL=OFF` + `-flto` | CPU + LTOFULL |
| **Metal** | `build-metal/` | `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON` | Metal + LTOOFF/LTOMUNINN |
| **Metal + LTO** | `build-metal-lto/` | `GGML_METAL=ON`, `GGML_METAL_EMBED_LIBRARY=ON` + `-flto` | Metal + LTOFULL |

### Metal Build Configuration

```makefile
LLAMA_CMAKE_FLAGS_METAL = \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_HIP=OFF \
    -DGGML_SYCL=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3
```

Key differences from the CPU build:
- `GGML_METAL=ON` — enables Metal backend compilation
- `GGML_METAL_EMBED_LIBRARY=ON` — embeds the `.metal` compute shader source directly into the `libggml-metal.a` static library, so no external shader file is needed at runtime

### Metal Link Flags

Metal variants require additional frameworks at link time:

```makefile
LDFLAGS_METAL = -framework Metal -framework MetalKit -framework Foundation
```

These are in addition to the existing `-framework Accelerate` and `-lc++`.

### Code Change: GPU Layer Offloading

`embed_gguf.c:137` currently hardcodes `mparams.n_gpu_layers = 0` (CPU-only). Metal variants need this set to `99` (offload all layers) to actually use the GPU.

The approach is a compile-time define with runtime override:

```c
/* In embed_gguf.c load_gguf_model() */
#ifndef MUNINN_DEFAULT_GPU_LAYERS
#define MUNINN_DEFAULT_GPU_LAYERS 0
#endif

int ngl = MUNINN_DEFAULT_GPU_LAYERS;
const char *ngl_env = getenv("MUNINN_GPU_LAYERS");
if (ngl_env) ngl = atoi(ngl_env);
mparams.n_gpu_layers = ngl;
```

- **CPU variants:** compiled with default (`MUNINN_DEFAULT_GPU_LAYERS=0`) — all inference on CPU
- **Metal variants:** compiled with `-DMUNINN_DEFAULT_GPU_LAYERS=99` — all layers offloaded to GPU
- **Runtime override:** `MUNINN_GPU_LAYERS=0` env var can force CPU mode even on a Metal build (useful for debugging)

---

## 5. Naming Convention

```
muninn_{OptLevel}_{LTOTier}_{Backend}{ext}
```

Examples (macOS):
```
build/muninn_O2_LTOOFF_CPU.dylib
build/muninn_O2_LTOMUNINN_CPU.dylib
build/muninn_O2_LTOFULL_CPU.dylib
build/muninn_O3_LTOOFF_CPU.dylib
build/muninn_O3_LTOMUNINN_CPU.dylib
build/muninn_O3_LTOFULL_CPU.dylib
build/muninn_Os_LTOOFF_CPU.dylib
build/muninn_Os_LTOMUNINN_CPU.dylib
build/muninn_Os_LTOFULL_CPU.dylib
build/muninn_O2_LTOOFF_METAL.dylib
build/muninn_O2_LTOMUNINN_METAL.dylib
build/muninn_O2_LTOFULL_METAL.dylib
build/muninn_O3_LTOOFF_METAL.dylib
build/muninn_O3_LTOMUNINN_METAL.dylib
build/muninn_O3_LTOFULL_METAL.dylib
build/muninn_Os_LTOOFF_METAL.dylib
build/muninn_Os_LTOMUNINN_METAL.dylib
build/muninn_Os_LTOFULL_METAL.dylib
```

The regular `build/muninn.dylib` remains unchanged (the default `-O2` CPU no-LTO build for development and testing). The optimisation variants are **additional** build targets, not replacements.

---

## 6. Makefile Integration

New targets in the root `Makefile`:

```makefile
######################################################################
# OPTIMISATION VARIANTS (for benchmarking)
######################################################################

# 3x3x2 build matrix: {O2, O3, Os} x {LTOOFF, LTOMUNINN, LTOFULL} x {CPU, METAL}
OPTIM_VARIANTS_CPU = O2_LTOOFF_CPU O2_LTOMUNINN_CPU O2_LTOFULL_CPU \
                     O3_LTOOFF_CPU O3_LTOMUNINN_CPU O3_LTOFULL_CPU \
                     Os_LTOOFF_CPU Os_LTOMUNINN_CPU Os_LTOFULL_CPU

OPTIM_VARIANTS_METAL = O2_LTOOFF_METAL O2_LTOMUNINN_METAL O2_LTOFULL_METAL \
                       O3_LTOOFF_METAL O3_LTOMUNINN_METAL O3_LTOFULL_METAL \
                       Os_LTOOFF_METAL Os_LTOMUNINN_METAL Os_LTOFULL_METAL

OPTIM_VARIANTS = $(OPTIM_VARIANTS_CPU) $(OPTIM_VARIANTS_METAL)

# ── Compiler flags per variant (applied to muninn .c files) ──

# CPU variants
OPTFLAGS_O2_LTOOFF_CPU     = -O2
OPTFLAGS_O2_LTOMUNINN_CPU  = -O2 -flto
OPTFLAGS_O2_LTOFULL_CPU    = -O2 -flto
OPTFLAGS_O3_LTOOFF_CPU     = -O3
OPTFLAGS_O3_LTOMUNINN_CPU  = -O3 -flto
OPTFLAGS_O3_LTOFULL_CPU    = -O3 -flto
OPTFLAGS_Os_LTOOFF_CPU     = -Os
OPTFLAGS_Os_LTOMUNINN_CPU  = -Os -flto
OPTFLAGS_Os_LTOFULL_CPU    = -Os -flto

# Metal variants — same compiler flags + GPU layer offload define
OPTFLAGS_O2_LTOOFF_METAL     = -O2 -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_O2_LTOMUNINN_METAL  = -O2 -flto -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_O2_LTOFULL_METAL    = -O2 -flto -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_O3_LTOOFF_METAL     = -O3 -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_O3_LTOMUNINN_METAL  = -O3 -flto -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_O3_LTOFULL_METAL    = -O3 -flto -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_Os_LTOOFF_METAL     = -Os -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_Os_LTOMUNINN_METAL  = -Os -flto -DMUNINN_DEFAULT_GPU_LAYERS=99
OPTFLAGS_Os_LTOFULL_METAL    = -Os -flto -DMUNINN_DEFAULT_GPU_LAYERS=99

# ── Linker flags per variant ──

# CPU variants
OPTLDFLAGS_O2_LTOOFF_CPU     =
OPTLDFLAGS_O2_LTOMUNINN_CPU  = -flto
OPTLDFLAGS_O2_LTOFULL_CPU    = -flto
OPTLDFLAGS_O3_LTOOFF_CPU     =
OPTLDFLAGS_O3_LTOMUNINN_CPU  = -flto
OPTLDFLAGS_O3_LTOFULL_CPU    = -flto
OPTLDFLAGS_Os_LTOOFF_CPU     =
OPTLDFLAGS_Os_LTOMUNINN_CPU  = -flto
OPTLDFLAGS_Os_LTOFULL_CPU    = -flto

# Metal variants — same LTO linker flags + Metal frameworks
LDFLAGS_METAL = -framework Metal -framework MetalKit -framework Foundation
OPTLDFLAGS_O2_LTOOFF_METAL     = $(LDFLAGS_METAL)
OPTLDFLAGS_O2_LTOMUNINN_METAL  = -flto $(LDFLAGS_METAL)
OPTLDFLAGS_O2_LTOFULL_METAL    = -flto $(LDFLAGS_METAL)
OPTLDFLAGS_O3_LTOOFF_METAL     = $(LDFLAGS_METAL)
OPTLDFLAGS_O3_LTOMUNINN_METAL  = -flto $(LDFLAGS_METAL)
OPTLDFLAGS_O3_LTOFULL_METAL    = -flto $(LDFLAGS_METAL)
OPTLDFLAGS_Os_LTOOFF_METAL     = $(LDFLAGS_METAL)
OPTLDFLAGS_Os_LTOMUNINN_METAL  = -flto $(LDFLAGS_METAL)
OPTLDFLAGS_Os_LTOFULL_METAL    = -flto $(LDFLAGS_METAL)

# ── Which llama.cpp build to link against ──

# CPU variants: standard build or LTO build
LLAMA_LIBS_O2_LTOOFF_CPU     = $(LLAMA_LIBS)
LLAMA_LIBS_O2_LTOMUNINN_CPU  = $(LLAMA_LIBS)
LLAMA_LIBS_O2_LTOFULL_CPU    = $(LLAMA_LIBS_LTO)
LLAMA_LIBS_O3_LTOOFF_CPU     = $(LLAMA_LIBS)
LLAMA_LIBS_O3_LTOMUNINN_CPU  = $(LLAMA_LIBS)
LLAMA_LIBS_O3_LTOFULL_CPU    = $(LLAMA_LIBS_LTO)
LLAMA_LIBS_Os_LTOOFF_CPU     = $(LLAMA_LIBS)
LLAMA_LIBS_Os_LTOMUNINN_CPU  = $(LLAMA_LIBS)
LLAMA_LIBS_Os_LTOFULL_CPU    = $(LLAMA_LIBS_LTO)

# Metal variants: Metal build or Metal+LTO build
LLAMA_LIBS_O2_LTOOFF_METAL     = $(LLAMA_LIBS_METAL)
LLAMA_LIBS_O2_LTOMUNINN_METAL  = $(LLAMA_LIBS_METAL)
LLAMA_LIBS_O2_LTOFULL_METAL    = $(LLAMA_LIBS_METAL_LTO)
LLAMA_LIBS_O3_LTOOFF_METAL     = $(LLAMA_LIBS_METAL)
LLAMA_LIBS_O3_LTOMUNINN_METAL  = $(LLAMA_LIBS_METAL)
LLAMA_LIBS_O3_LTOFULL_METAL    = $(LLAMA_LIBS_METAL_LTO)
LLAMA_LIBS_Os_LTOOFF_METAL     = $(LLAMA_LIBS_METAL)
LLAMA_LIBS_Os_LTOMUNINN_METAL  = $(LLAMA_LIBS_METAL)
LLAMA_LIBS_Os_LTOFULL_METAL    = $(LLAMA_LIBS_METAL_LTO)

# ── llama.cpp build configurations ──

# CPU + LTO
LLAMA_BUILD_LTO = $(LLAMA_DIR)/build-lto
LLAMA_LIBS_LTO_CORE = $(LLAMA_BUILD_LTO)/src/libllama.a \
                      $(LLAMA_BUILD_LTO)/ggml/src/libggml.a \
                      $(LLAMA_BUILD_LTO)/ggml/src/libggml-base.a \
                      $(LLAMA_BUILD_LTO)/ggml/src/libggml-cpu.a
ifeq ($(UNAME_S),Darwin)
    LLAMA_LIBS_LTO = $(LLAMA_LIBS_LTO_CORE) $(LLAMA_BUILD_LTO)/ggml/src/ggml-blas/libggml-blas.a
else
    LLAMA_LIBS_LTO = $(LLAMA_LIBS_LTO_CORE) $(wildcard $(LLAMA_BUILD_LTO)/ggml/src/ggml-blas/libggml-blas.a)
endif

# Metal (no LTO)
LLAMA_BUILD_METAL = $(LLAMA_DIR)/build-metal
LLAMA_CMAKE_FLAGS_METAL = \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON \
    -DGGML_CUDA=OFF \
    -DGGML_VULKAN=OFF \
    -DGGML_HIP=OFF \
    -DGGML_SYCL=OFF \
    -DGGML_OPENMP=OFF \
    -DGGML_BACKEND_DL=OFF \
    -DLLAMA_BUILD_COMMON=OFF \
    -DLLAMA_BUILD_TESTS=OFF \
    -DLLAMA_BUILD_TOOLS=OFF \
    -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_SERVER=OFF \
    -DCMAKE_BUILD_TYPE=MinSizeRel \
    -DCMAKE_OSX_DEPLOYMENT_TARGET=13.3
LLAMA_LIBS_METAL_CORE = $(LLAMA_BUILD_METAL)/src/libllama.a \
                        $(LLAMA_BUILD_METAL)/ggml/src/libggml.a \
                        $(LLAMA_BUILD_METAL)/ggml/src/libggml-base.a \
                        $(LLAMA_BUILD_METAL)/ggml/src/libggml-cpu.a \
                        $(LLAMA_BUILD_METAL)/ggml/src/ggml-metal/libggml-metal.a
LLAMA_LIBS_METAL = $(LLAMA_LIBS_METAL_CORE) $(LLAMA_BUILD_METAL)/ggml/src/ggml-blas/libggml-blas.a

# Metal + LTO
LLAMA_BUILD_METAL_LTO = $(LLAMA_DIR)/build-metal-lto
LLAMA_LIBS_METAL_LTO_CORE = $(LLAMA_BUILD_METAL_LTO)/src/libllama.a \
                            $(LLAMA_BUILD_METAL_LTO)/ggml/src/libggml.a \
                            $(LLAMA_BUILD_METAL_LTO)/ggml/src/libggml-base.a \
                            $(LLAMA_BUILD_METAL_LTO)/ggml/src/libggml-cpu.a \
                            $(LLAMA_BUILD_METAL_LTO)/ggml/src/ggml-metal/libggml-metal.a
LLAMA_LIBS_METAL_LTO = $(LLAMA_LIBS_METAL_LTO_CORE) $(LLAMA_BUILD_METAL_LTO)/ggml/src/ggml-blas/libggml-blas.a

# ── Build targets for llama.cpp variants ──

llamacpp-lto: $(LLAMA_LIBS_LTO_CORE)     ## Build llama.cpp static libs with LTO
$(LLAMA_LIBS_LTO_CORE): | $(LLAMA_DIR)/CMakeLists.txt
	@echo "######### Building llama.cpp with LTO..."
	cmake -B $(LLAMA_BUILD_LTO) -S $(LLAMA_DIR) $(LLAMA_CMAKE_FLAGS) \
		-DCMAKE_C_FLAGS="-flto" -DCMAKE_CXX_FLAGS="-flto"
	cmake --build $(LLAMA_BUILD_LTO) --config MinSizeRel -j

llamacpp-metal: $(LLAMA_LIBS_METAL_CORE)  ## Build llama.cpp static libs with Metal
$(LLAMA_LIBS_METAL_CORE): | $(LLAMA_DIR)/CMakeLists.txt
	@echo "######### Building llama.cpp with Metal..."
	cmake -B $(LLAMA_BUILD_METAL) -S $(LLAMA_DIR) $(LLAMA_CMAKE_FLAGS_METAL)
	cmake --build $(LLAMA_BUILD_METAL) --config MinSizeRel -j

llamacpp-metal-lto: $(LLAMA_LIBS_METAL_LTO_CORE)  ## Build llama.cpp with Metal + LTO
$(LLAMA_LIBS_METAL_LTO_CORE): | $(LLAMA_DIR)/CMakeLists.txt
	@echo "######### Building llama.cpp with Metal + LTO..."
	cmake -B $(LLAMA_BUILD_METAL_LTO) -S $(LLAMA_DIR) $(LLAMA_CMAKE_FLAGS_METAL) \
		-DCMAKE_C_FLAGS="-flto" -DCMAKE_CXX_FLAGS="-flto"
	cmake --build $(LLAMA_BUILD_METAL_LTO) --config MinSizeRel -j

# Build a single variant: make optim-build VARIANT=O3_LTOFULL_CPU
optim-build:                              ## Build one optimisation variant (VARIANT=...)
	@test -n "$(VARIANT)" || (echo "Usage: make optim-build VARIANT=O3_LTOFULL_CPU" && exit 1)
	@mkdir -p build
	$(CC) $(filter-out -O%,$(CFLAGS_BASE)) $(OPTFLAGS_$(VARIANT)) $(CFLAGS_EXTRA) \
		$(SHARED_FLAGS) -Isrc $(LLAMA_INCLUDE) \
		-o build/muninn_$(VARIANT)$(EXT) $(SRC) \
		$(LLAMA_LIBS_$(VARIANT)) $(LDFLAGS) $(OPTLDFLAGS_$(VARIANT))

# Build all 18 variants (builds all four llama.cpp versions first)
optim-all: $(LLAMA_LIBS) llamacpp-lto llamacpp-metal llamacpp-metal-lto  ## Build all 18 optimisation variants
	@for v in $(OPTIM_VARIANTS); do \
		echo ""; \
		echo "======== Building muninn_$$v$(EXT) ========"; \
		time $(MAKE) optim-build VARIANT=$$v; \
	done

# Build only CPU variants (9)
optim-cpu: $(LLAMA_LIBS) llamacpp-lto     ## Build 9 CPU optimisation variants
	@for v in $(OPTIM_VARIANTS_CPU); do \
		echo ""; \
		echo "======== Building muninn_$$v$(EXT) ========"; \
		time $(MAKE) optim-build VARIANT=$$v; \
	done

# Build only Metal variants (9)
optim-metal: llamacpp-metal llamacpp-metal-lto  ## Build 9 Metal optimisation variants
	@for v in $(OPTIM_VARIANTS_METAL); do \
		echo ""; \
		echo "======== Building muninn_$$v$(EXT) ========"; \
		time $(MAKE) optim-build VARIANT=$$v; \
	done

optim-clean:                              ## Remove optimisation variant binaries
	rm -f build/muninn_O*$(EXT)
```

Key details:
- `$(filter-out -O%,$(CFLAGS_BASE))` strips the default `-O2` before applying the variant's flag
- LTOFULL variants link against the corresponding LTO build directory
- Metal variants add `-DMUNINN_DEFAULT_GPU_LAYERS=99` to compiler flags and Metal frameworks to linker flags
- Metal builds produce `libggml-metal.a` with embedded shaders (`GGML_METAL_EMBED_LIBRARY=ON`)
- `optim-all` builds all four llama.cpp variants first, then all 18 muninn variants
- `optim-cpu` and `optim-metal` allow building just one backend's 9 variants
- `optim-build` has no prerequisite on llama libs — the aggregate targets handle dependencies

---

## 7. Phase 0 — Build & Measure

Before running any benchmarks, build all eighteen variants and record compile time and binary size. This is a worthwhile exercise on its own and may eliminate variants from the benchmark matrix.

### Procedure

```bash
# 1. Build all 18 variants (time is captured per-variant)
make optim-all 2>&1 | tee build_results.log

# 2. Record binary sizes
ls -la build/muninn_*$(EXT) | awk '{print $5, $NF}'

# 3. Verify correctness — run tests against each variant
for v in build/muninn_O*.dylib; do
    echo "Testing $v..."
    MUNINN_EXT_PATH=$v .venv/bin/python -m pytest pytests/ -x -q
done
```

### Expected Output Table

#### CPU Variants

```
| Variant              | Compile (s) | Binary Size (MB) | Tests Pass |
|----------------------|-------------|-------------------|------------|
| O2_LTOOFF_CPU        |             |                   |            |
| O2_LTOMUNINN_CPU     |             |                   |            |
| O2_LTOFULL_CPU       |             |                   |            |
| O3_LTOOFF_CPU        |             |                   |            |
| O3_LTOMUNINN_CPU     |             |                   |            |
| O3_LTOFULL_CPU       |             |                   |            |
| Os_LTOOFF_CPU        |             |                   |            |
| Os_LTOMUNINN_CPU     |             |                   |            |
| Os_LTOFULL_CPU       |             |                   |            |
```

#### Metal Variants

```
| Variant              | Compile (s) | Binary Size (MB) | Tests Pass |
|----------------------|-------------|-------------------|------------|
| O2_LTOOFF_METAL      |             |                   |            |
| O2_LTOMUNINN_METAL   |             |                   |            |
| O2_LTOFULL_METAL     |             |                   |            |
| O3_LTOOFF_METAL      |             |                   |            |
| O3_LTOMUNINN_METAL   |             |                   |            |
| O3_LTOFULL_METAL     |             |                   |            |
| Os_LTOOFF_METAL      |             |                   |            |
| Os_LTOMUNINN_METAL   |             |                   |            |
| Os_LTOFULL_METAL     |             |                   |            |
```

### Elimination Criteria

Drop a variant from Phase 1 if:
- **Compile time is prohibitive** — e.g., LTOFULL takes 10x longer than LTOOFF and the binary size difference is negligible
- **Binary size is unreasonable** — e.g., Metal + `-O3` + LTOFULL balloons excessively (note: Metal variants will be larger due to embedded shaders)
- **Tests fail** — correctness regression means the flags are unusable
- **Diminishing returns** — if LTOMUNINN and LTOFULL produce identical binaries (same size, same `objdump` profile), there's no point benchmarking both
- **Metal overhead** — if Metal variants are consistently slower than CPU for small embedding models (due to GPU dispatch overhead), document this and potentially drop them

The survivors proceed to Phase 1 VSS benchmarking.

---

## 8. Phase 1 — VSS Benchmarks

Narrowed to VSS-only operations. Graph TVFs and adjacency VTs are excluded — the compiler optimisation impact on those is expected to be minimal compared to the tight numerical loops in VSS.

### Metrics

| Metric | What it measures | Hot path under test | Backend-sensitive? |
|--------|-----------------|---------------------|-------------------|
| **HNSW insertion** | Node insertion rate (nodes/sec) | `hnsw_algo.c` — insert + neighbor selection + distance computation | No — pure CPU, no llama.cpp |
| **Vector embedding** | GGUF model inference throughput (texts/sec) | `embed_gguf.c` -> llama.cpp | **Yes** — Metal vs CPU is the key comparison |
| **VSS search** | HNSW k-NN query latency (ms/query) | `hnsw_algo.c` beam search + `vec_math.c` distance functions | No — pure CPU, no llama.cpp |

### Expected Backend Impact

- **HNSW insertion and search:** These operations are pure muninn C code (vec_math distance calculations, priority queue operations, hash-map lookups). Metal has **zero effect** — the CPU/Metal difference should be within noise. The compiler optimisation level and LTO tier are the only variables that matter here.
- **Vector embedding:** This is the operation where Metal should shine. The entire llama.cpp inference pipeline (tokenize → encode → pool → normalize) can benefit from GPU offload. The CPU/Metal speedup ratio is the primary question this benchmark answers.

### Why This Still Justifies Full Cross-Product

Even though Metal only affects embedding, the **surrounding muninn code** still runs on CPU: the L2 normalization loop (`embed_gguf.c:252-259`), token buffer allocation, SQLite result marshalling. Different `-O` levels affect this code differently. The full cross-product captures any interaction effects between compiler optimisation and backend choice.

### Control Variables

- **Dataset:** Same pre-built `.npy` vector caches across all variants
- **Dimensions:** 128, 384 (two data points to check if dim affects the relative speedup)
- **N:** 10K nodes (large enough that insertion and search are non-trivial)
- **Hardware:** Same machine, sequential runs (not parallel)
- **Iterations:** Min 5 per measurement, report median + p5/p95
- **Warmup:** Discard first run to avoid cold-cache effects
- **Embedding model:** Same GGUF model for all variants (e.g., `all-MiniLM-L6-v2-f16.gguf`)
- **Metal state:** For Metal variants, ensure GPU is idle before each run (no other Metal workloads)

---

## 9. JSONL Schema

Results accumulate in `benchmarks/results/compiler_optimisation.jsonl`:

```json
{
    "timestamp": "2026-02-22T14:30:00Z",
    "variant": "O3_LTOMUNINN_CPU",
    "opt_level": "O3",
    "lto_tier": "LTOMUNINN",
    "backend": "CPU",
    "binary_size_bytes": 3145728,

    "operation": "hnsw_insert",
    "dataset": "random_128d",
    "n": 10000,
    "dim": 128,
    "iterations": 10,

    "median_ms": 45.2,
    "p5_ms": 43.1,
    "p95_ms": 48.7,
    "ops_per_sec": 221.2,

    "platform": "darwin-arm64",
    "compiler": "Apple clang 16.0.0"
}
```

Field enums:
- `lto_tier`: `"LTOOFF"`, `"LTOMUNINN"`, `"LTOFULL"`
- `backend`: `"CPU"`, `"METAL"`
- `operation`: `"hnsw_insert"`, `"hnsw_search"`, `"embed_text"`

---

## 10. Analysis & Charting

### Primary Charts: Speedup Heatmaps (one per backend)

Two 3x3 grids (opt level x LTO tier), one for CPU and one for Metal, showing relative speedup vs baseline (`O2_LTOOFF_CPU = 1.0x`):

```
CPU Backend                                 Metal Backend
                LTOOFF  LTOMUNINN  LTOFULL                  LTOOFF  LTOMUNINN  LTOFULL
         ┌────────┬──────────┬────────┐          ┌────────┬──────────┬────────┐
   -O2   │ 1.00x  │  1.05x   │ 1.06x  │    -O2   │ 3.50x  │  3.55x   │ 3.56x  │
   -O3   │ 1.08x  │  1.15x   │ 1.16x  │    -O3   │ 3.60x  │  3.70x   │ 3.71x  │
   -Os   │ 0.97x  │  1.02x   │ 1.03x  │    -Os   │ 3.45x  │  3.50x   │ 3.52x  │
         └────────┴──────────┴────────┘          └────────┴──────────┴────────┘
```

(The Metal numbers above are illustrative — actual GPU speedups depend heavily on model size and batch characteristics. Small embedding models may see modest gains; larger models should see significant acceleration.)

One pair of heatmaps per VSS operation (3 operations x 2 backends), plus geometric mean heatmaps.

### Backend Comparison Chart

The most important new chart: **CPU vs Metal speedup for embedding inference**, holding compiler flags constant:

```
Embedding Throughput: Metal Speedup over CPU
         ┌─────────────────────────────────────────┐
   O2    │████████████████████  3.2x                │
   O3    │█████████████████████ 3.4x                │
   Os    │███████████████████   3.1x                │
         └─────────────────────────────────────────┘
         (LTO tier averaged within each opt level)
```

This answers the key question: **is Metal worth the build complexity for embedding workloads?**

### Secondary Charts

- **Binary size comparison:** Bar chart across all 18 variants (Metal variants expected to be larger due to embedded shaders)
- **Build time comparison:** Bar chart (from Phase 0 data, including llama.cpp rebuild times)
- **Pareto frontier:** Binary size (x) vs speedup (y) — identifies the best size/speed tradeoff, with CPU and Metal variants plotted in different colours
- **HNSW-only heatmap:** Since HNSW operations are backend-independent, overlay CPU and Metal results to confirm they're within noise — this validates the benchmark methodology

### Decision Framework

The "winner" depends on the deployment target:

**For CPU-only deployment** (portability, WASM, CI):
1. Passes all tests
2. Acceptable build time
3. Sits on the CPU Pareto frontier

**For macOS native deployment** (maximum performance):
1. Passes all tests
2. Metal embedding speedup is >2x over CPU equivalent
3. Binary size increase from embedded shaders is acceptable
4. Sits on the Metal Pareto frontier

If `-Os` + LTO is within 5% of `-O3` + LTO but is significantly smaller, `-Os` wins. If `-O3` is 15%+ faster, it wins despite size.

If Metal embedding speedup is <1.5x for the target model sizes, the added build complexity may not be worth it — CPU with Accelerate BLAS may be "good enough".

---

## 11. Implementation Steps

```
Phase 0 — Build & measure (this alone is valuable):
  1. Modify embed_gguf.c: add MUNINN_DEFAULT_GPU_LAYERS compile-time
     define with MUNINN_GPU_LAYERS env var override
  2. Add llama.cpp Metal build targets (llamacpp-metal, llamacpp-metal-lto)
  3. Add optim-build / optim-all / optim-cpu / optim-metal / optim-clean
     targets to root Makefile
  4. Build all 4 llama.cpp variants (CPU, CPU+LTO, Metal, Metal+LTO)
  5. Build all 18 muninn variants, record compile times
  6. Record binary sizes
  7. Run test suite against each variant to verify correctness
  8. Fill in the Phase 0 results tables in this doc
  9. Decide which variants to carry forward to Phase 1

Phase 1 — VSS benchmarks (on surviving variants):
  10. Create benchmark script for VSS operations
      - Loads each variant via sqlite3.load_extension()
      - Runs HNSW insertion, VSS search, embedding generation
      - Records backend field in JSONL output
  11. Execute benchmarks across surviving variants
  12. Generate heatmaps (per-backend) + backend comparison + Pareto charts
  13. Select the winning flag combination for each deployment target
  14. Update default CFLAGS in root Makefile if warranted
  15. Optionally: update default LLAMA_CMAKE_FLAGS to include Metal
      if Metal proves worthwhile for macOS builds
```

---

## 12. Open Questions

1. **`-march=native`:** Should we also test with `-march=native`? This is orthogonal to the `-O`/LTO/backend matrix — it enables hardware-specific instruction selection (NEON on Apple Silicon). Could be a follow-up if the `-O`/LTO results are inconclusive. Note: llama.cpp uses `-DGGML_NATIVE=OFF` because `NATIVE=ON` hangs on Apple Silicon, but muninn's own code could use `-march=native` independently.

2. **PGO (Profile-Guided Optimisation):** A natural follow-up — instrument with `-fprofile-generate`, run benchmarks as the training workload, rebuild with `-fprofile-use`. Larger effort but could yield significant wins for branch-heavy code like HNSW beam search.

3. **Linker flags (`-dead_strip`, `-exported_symbols_list`):** Apply universally to all variants rather than making it a dimension. The extension only exports `sqlite3_muninn_init`, so everything unreachable from that symbol can be stripped. Primarily affects binary size, not speed.

4. **LTOFULL build isolation:** The `build-lto/`, `build-metal/`, and `build-metal-lto/` directories keep all llama.cpp builds separate. Should `optim-clean` also clean these? Probably not — add separate `llama-*-clean` targets, since rebuilding llama.cpp is expensive.

5. **Metal shader compilation overhead:** The first Metal inference call may incur shader compilation latency (Metal compiles `.metal` source to GPU-specific code on first use). The warmup run should absorb this, but verify that subsequent iterations are stable.

6. **Metal memory residency:** Embedding models loaded with `n_gpu_layers=99` allocate GPU-side buffers. Verify that loading a model into Metal doesn't impact subsequent HNSW operations (which are CPU-only) via memory pressure.

7. **Batch embedding performance:** The current `embed_text()` processes one text at a time. Metal's advantage is most pronounced with larger batch sizes due to GPU utilisation. Consider adding a batch embedding metric as a stretch goal — this would require API changes to `muninn_embed()` to accept multiple texts.

8. **Model size sensitivity:** The MiniLM model (6 layers, ~22M params) may be too small for Metal to outperform CPU (GPU dispatch overhead can dominate). Consider also benchmarking with a larger model like `nomic-embed-text-v1.5` (12 layers, ~137M params) to find the crossover point.
