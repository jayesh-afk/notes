# Mastering Tenstorrent tt-llk: From Zero to Custom Kernel
### A Complete, Ground-Up Guide — Bird's Eye to Atomic Structure

---

## Table of Contents

1. [What is tt-llk and Why Does it Exist?](#1-what-is-tt-llk-and-why-does-it-exist)
2. [The Big Picture: The Tenstorrent Chip Architecture](#2-the-big-picture-the-tenstorrent-chip-architecture)
3. [Zooming In: The Tensix Core — Your Compute Unit](#3-zooming-in-the-tensix-core--your-compute-unit)
4. [The Five RISC-V Cores Inside One Tensix Core](#4-the-five-risc-v-cores-inside-one-tensix-core)
5. [The Three-Stage Pipeline — Heart of LLK](#5-the-three-stage-pipeline--heart-of-llk)
6. [Registers: Where the Math Actually Happens](#6-registers-where-the-math-actually-happens)
7. [Tiles and Faces: How Data is Organized](#7-tiles-and-faces-how-data-is-organized)
8. [Data Formats: The Number Types tt-llk Supports](#8-data-formats-the-number-types-tt-llk-supports)
9. [The Full Repository Structure](#9-the-full-repository-structure)
10. [C++ and C Concepts Used in This Repo (With Examples)](#10-c-and-c-concepts-used-in-this-repo-with-examples)
11. [The Core Kernel Layer: `ckernel.h`](#11-the-core-kernel-layer-ckernel-h)
12. [The Unpacker: Stage 1 Deep Dive](#12-the-unpacker-stage-1-deep-dive)
13. [The Math Unit: Stage 2 Deep Dive](#13-the-math-unit-stage-2-deep-dive)
14. [The SFPU: The Vector Engine Inside Math](#14-the-sfpu-the-vector-engine-inside-math)
15. [The Packer: Stage 3 Deep Dive](#15-the-packer-stage-3-deep-dive)
16. [Synchronization: How the Three Stages Talk to Each Other](#16-synchronization-how-the-three-stages-talk-to-each-other)
17. [Address Modifiers and MOPs: The Loop Optimization System](#17-address-modifiers-and-mops-the-loop-optimization-system)
18. [Multi-Architecture Support: WH vs BH vs Quasar](#18-multi-architecture-support-wh-vs-bh-vs-quasar)
19. [A Complete Operation Flow: Worked Example (Matmul)](#19-a-complete-operation-flow-worked-example-matmul)
20. [How tt-llk Connects to tt-metal (the Full Stack)](#20-how-tt-llk-connects-to-tt-metal-the-full-stack)
21. [The Testing Framework](#21-the-testing-framework)
22. [The Build System and SFPI Toolchain](#22-the-build-system-and-sfpi-toolchain)
23. [How to Write Your Own Custom Kernel](#23-how-to-write-your-own-custom-kernel)
24. [Quick Reference Cheat Sheet](#24-quick-reference-cheat-sheet)

---

## 1. What is tt-llk and Why Does it Exist?

### The One-Sentence Definition

`tt-llk` is a **C++ header-only library** that lets you directly program the compute engines inside Tenstorrent's AI chips at the lowest possible level — without the overhead of a runtime, OS, or abstraction layer in between.

### The "Why" in Plain Language

When you run a neural network, every operation (matrix multiply, ReLU, softmax, etc.) ultimately needs to execute on hardware. On a GPU, CUDA handles this. On Tenstorrent hardware, the bottom layer is `tt-llk`.

Think of it like this:

```
PyTorch / JAX / ONNX
        ↓
    tt-nn (operator library)
        ↓
   tt-metal (runtime / SDK)
        ↓
   tt-llk  ← YOU ARE HERE
        ↓
   SFPI Compiler (riscv-tt-elf-g++)
        ↓
   Tensix Hardware (Wormhole / Blackhole / Quasar)
```

tt-llk is the **last software layer** before metal. Everything below is silicon.

### What "Header-Only" Means

All tt-llk code lives in `.h` files (header files). There is no separate `.cpp` file to compile into a library. When you use tt-llk, the compiler includes the header and compiles the functions directly into your kernel binary. This is intentional — it lets the compiler inline everything, maximizing performance and allowing heavy compile-time optimization.

---

## 2. The Big Picture: The Tenstorrent Chip Architecture

### The Grid of Cores Design

Unlike a GPU (which has thousands of identical shader cores sharing a large L2 cache), a Tenstorrent chip is organized as a **2D grid of independent nodes**. Each node is mostly self-contained with its own memory and compute.

```
Tenstorrent Chip (e.g., Wormhole N150) — Top View

  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │  T = Tensix Core
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │DRAM│ T  │ T  │ T  │ T  │ T  │ T  │DRAM│
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │ T  │
  └────┴────┴────┴────┴────┴────┴────┴────┘

Nodes communicate via NOC (Network on Chip) — like a tiny internet inside the chip.
```

There are ~64 Tensix cores on a Wormhole chip, spread across this grid. The grid also contains:
- **DRAM banks** — for storing large tensors
- **Ethernet cores** — for chip-to-chip communication (multi-chip setups)
- **PCIe core** — interface to the host CPU

### Why a Grid Instead of Shared Memory?

The key insight: for AI workloads (matmul, attention, convolutions), data access patterns are **predictable and local**. A core doing part of a matrix multiply only needs data from its neighbors — not from across the chip. This means:
- No need for a huge shared cache (saves area + power)
- No cache coherency problem (each core has its own 1.5MB L1 SRAM)
- Data movement is explicit and software-controlled — fast when done right

---

## 3. Zooming In: The Tensix Core — Your Compute Unit

A single Tensix core contains:

| Component | Description |
|-----------|-------------|
| **L1 SRAM** | 1.5MB local memory — holds input/output tiles and kernel code |
| **5 RISC-V CPUs** | The "brains" that control all compute and data movement |
| **FPU (Matrix Engine)** | Hardware matrix multiplier — the workhorse for GEMM |
| **SFPU (Vector Engine)** | Programmable vector unit — for elementwise ops (exp, sqrt, gelu...) |
| **Unpacker (×2)** | DMA engine: copies tiles from L1 → SrcA/SrcB registers |
| **Packer (×4)** | DMA engine: copies results from Dest register → L1 |
| **2 NOC Routers** | Connect to other cores and DRAM |

The crucial thing to internalize: **the math engines (FPU and SFPU) do NOT directly touch L1 SRAM**. Data must first be *unpacked* from SRAM into special registers (SrcA, SrcB) before computation. After computation, results sit in the Dest register and must be *packed* back into SRAM. This is why the three-stage pipeline exists.

---

## 4. The Five RISC-V Cores Inside One Tensix Core

This is probably the most surprising thing about Tenstorrent. What looks like "one compute core" is actually **five separate RISC-V CPUs running concurrently**.

```
┌─────────────────────────────────────────────────────────────┐
│                     ONE Tensix Core                         │
│                                                             │
│  ┌──────────┐  ┌──────────┐  ┌────────┐  ┌──────┐  ┌────┐ │
│  │  BRISC   │  │  NCRISC  │  │ TRISC0 │  │TRISC1│  │TRISC2│ │
│  │(Data Mov)│  │(Data Mov)│  │(Unpack)│  │(Math)│  │(Pack)│ │
│  └──────────┘  └──────────┘  └────────┘  └──────┘  └────┘ │
│                                                             │
│   ← NOC / DMA side →          ← Compute side →             │
└─────────────────────────────────────────────────────────────┘
```

### Each Core's Job

**BRISC (Broadside RISC)** — also called "Data Movement 0" (DM0)
- Runs the *reader kernel*
- Talks to NOC (Network on Chip) to pull data from DRAM or other cores
- Pushes data into L1 circular buffers for TRISC0 to process
- Does NOT touch the math engines

**NCRISC (Network RISC)** — also called "Data Movement 1" (DM1)
- Runs the *writer kernel*
- Reads results from L1 circular buffers
- Sends processed tiles out via NOC to DRAM or other cores
- Does NOT touch the math engines

**TRISC0** — Controls the **Unpacker**
- Reads tiles from L1 SRAM circular buffers
- Triggers the unpacker hardware to DMA the tile into SrcA or SrcB registers
- Handles format conversion during unpacking (e.g., BFP8 → BF16)

**TRISC1** — Controls the **Math engine (FPU + SFPU)**
- Triggers matrix multiply operations on SrcA × SrcB → Dest
- Triggers SFPU elementwise operations on Dest values
- Does NOT move data itself — just sends commands to hardware

**TRISC2** — Controls the **Packer**
- Reads results from the Dest register
- Triggers packer hardware to DMA results into L1 SRAM output buffers
- Handles format conversion during packing (e.g., BF16 → BFP8)

### The Critical Insight: Macro Magic

When you write a compute kernel (the thing tt-llk powers), the **same source code** runs on TRISC0, TRISC1, and TRISC2 simultaneously. The runtime uses C preprocessor macros (`#ifdef TRISC0`, etc.) to split the code:

```cpp
namespace NAMESPACE {
void MAIN {
    // This code gets compiled THREE times,
    // each version for a different TRISC.
    // Macros route the right calls to the right core.

    // TRISC0 compiles the unpack_* calls
    // TRISC1 compiles the math_* calls  
    // TRISC2 compiles the pack_* calls
}
}
```

This is why the programming model feels a bit magical — one source, three simultaneous executors.

---

## 5. The Three-Stage Pipeline — Heart of LLK

All computation in tt-llk follows one pattern: **Unpack → Math → Pack**.

```
L1 SRAM                  Registers                L1 SRAM
(Input Tiles)                                    (Output Tiles)
     │                                                 ▲
     │  TRISC0                                         │
     ▼  (controls)                   TRISC2 (controls)│
  ┌──────────┐    SrcA/SrcB    ┌──────────┐    Dest   │
  │ UNPACKER │ ─────────────▶  │   MATH   │ ─────────▶│PACKER│
  │ (stage 1)│                 │ (stage 2)│           └──────┘
  └──────────┘                 └──────────┘
                                    ▲
                              TRISC1 (controls)
```

### Stage 1 — Unpack
Takes a tile from L1 SRAM, converts its data format if necessary, and loads it into either the **SrcA** or **SrcB** register file, making it ready for the math engine.

### Stage 2 — Math
The FPU reads SrcA and SrcB, computes (e.g., matrix multiply), and writes results into the **Dest** register. The SFPU can then operate on Dest values for elementwise ops (exp, relu, gelu, etc.).

### Stage 3 — Pack
Reads values from Dest, converts format if necessary, and writes the tile back into L1 SRAM (into an output circular buffer).

### The Pipeline is Concurrent

All three stages run **at the same time** on different tiles:
- While stage 1 is loading tile N+1...
- Stage 2 is computing tile N...
- Stage 3 is packing tile N-1...

This is achieved via **double buffering** — the Dest register has two halves, and semaphores coordinate which half is being written vs. read.

---

## 6. Registers: Where the Math Actually Happens

### SrcA and SrcB
Two input register files, each holding one tile (32×32 values). The unpacker loads into these. The FPU reads from these.

- **Unpacker 0** → SrcA (and also Dest, for special unpack-to-dest flows)
- **Unpacker 1** → SrcB

### Dest Register
The output of the FPU and the working space of the SFPU. It holds:
- **16 tiles of BF16** (in normal mode), OR
- **8 tiles of FP32** (in FP32 accumulate mode)

It is physically split into two halves (double buffered). This is called `DstSync` — controlling which half the math engine writes to and which the packer reads from.

```
Dest Register (conceptual):
┌─────────────────────────────────────────┐
│  Half 0 (8 tiles BF16 or 4 tiles FP32) │  ← Math writes here
│  Half 1 (8 tiles BF16 or 4 tiles FP32) │  ← Packer reads here
└─────────────────────────────────────────┘
     ↕ Flip after each operation (dest_offset_id)
```

### Why Double-Buffer Dest?

Without double buffering, you'd have to wait for the packer to finish saving results before the math engine can compute the next tile. With double buffering, math and pack can run simultaneously on different halves → **2× throughput**.

---

## 7. Tiles and Faces: How Data is Organized

### The Tile
The fundamental unit of computation in tt-llk is the **tile** — a **32×32 matrix** of numbers. All operations work on whole tiles. You don't operate on individual values; you operate on 32×32 blocks.

### The Face
Each 32×32 tile is subdivided into four **16×16 faces**:

```
32×32 Tile Layout:
┌──────────────┬──────────────┐
│   Face 0     │   Face 1     │
│  (rows 0-15) │  (rows 0-15) │
│  (cols 0-15) │  (cols 16-31)│
├──────────────┼──────────────┤
│   Face 2     │   Face 3     │
│  (rows 16-31)│  (rows 16-31)│
│  (cols 0-15) │  (cols 16-31)│
└──────────────┴──────────────┘

Face offsets in address space: 0x00, 0x10, 0x20, 0x30
```

Why 16×16 faces? That's what the FPU hardware naturally works on — the 16×16 systolic array size is baked into silicon.

### Tile in Memory (Row-Major, Face-by-Face)
Tiles are stored in L1 with faces contiguous in memory. Within a face, values are stored row-major. The unpacker knows how to read this layout and load it into the register file.

### Circular Buffers (CB)
In L1 SRAM, tiles are organized into **circular buffers** — ring buffers where:
- BRISC pushes tiles into the back (coming from NOC/DRAM)
- TRISC0 reads from the front (to unpack for compute)
- TRISC2 pushes results into a separate output CB
- NCRISC reads results from that output CB (to send out via NOC)

Circular buffers are the handoff mechanism between the data movement cores and the compute cores.

---

## 8. Data Formats: The Number Types tt-llk Supports

tt-llk supports many numerical formats. Understanding these is essential because the unpacker and packer both do **format conversion** automatically.

| Format | Bits | Description | When to Use |
|--------|------|-------------|-------------|
| `Float32` | 32 | Standard IEEE float | High-precision accumulation |
| `Float16` | 16 | IEEE half-precision | Balance of speed and precision |
| `Bfloat16` | 16 | Brain float (1 exp byte) | Default for ML; fast on Tensix |
| `Bfp8_b` | 8 | Block float, shared exponent | Quantized inference |
| `Bfp4_b` | 4 | Block float 4-bit | Ultra-low precision |
| `Bfp2_b` | 2 | Block float 2-bit | Extreme compression |
| `Int32` | 32 | Integer | Integer operations |
| `Int8` | 8 | Signed 8-bit integer | Quantized weights |
| `UInt8` | 8 | Unsigned 8-bit integer | Activations, indices |

### Block Floating Point (BFP) — Key Concept

BFP formats store a **shared exponent** for a block of numbers and then store only the mantissa bits for each number individually. This saves memory bandwidth while preserving relative precision within the block.

Example (BFP8): One 16-value block shares one 8-bit exponent. Each value stores an 8-bit mantissa. Effective precision: ~8-bit per value stored in ~9 bits average — a great tradeoff for inference.

### Format Conversion in the Pipeline

```
L1 SRAM        Unpack converts         Registers       Pack converts        L1 SRAM
[BFP8 tile] ──────────────────────▶ [BF16 in SrcA] ─────────────────▶ [BFP8 tile]
                  (hardware)                               (hardware)
```

You specify `in_data_format` (what's in L1) and `out_data_format` (what goes into registers) in the unpacker config. The hardware does the conversion for free — no CPU cycles spent.

---

## 9. The Full Repository Structure

```
tt-llk/
│
├── tt_llk_wormhole_b0/          # Wormhole B0 (WH) implementation
│   ├── common/inc/              # Core headers (the foundation layer)
│   │   ├── ckernel.h            # ★ The most important file — hardware primitives
│   │   ├── ckernel_ops.h        # TTI (Tensix Tightly-coupled Instructions) macros
│   │   ├── ckernel_template.h   # MOP (Micro-Operation) template system
│   │   ├── ckernel_instr_params.h # Instruction parameter constants
│   │   ├── cunpack_common.h     # Unpacker config structs + init functions
│   │   ├── cpack_common.h       # Packer config structs + init functions
│   │   └── cmath_common.h       # Math unit config + ALU setup
│   │
│   └── llk_lib/                 # LLK API — what you actually call
│       │
│       ├── [UNPACK FILES]
│       ├── llk_unpack_A.h          # Unpack one tile to SrcA
│       ├── llk_unpack_AB.h         # Unpack two tiles to SrcA + SrcB
│       ├── llk_unpack_AB_matmul.h  # Unpack optimized for matmul (32×32 tiles)
│       ├── llk_unpack_common.h     # Shared unpack utilities
│       ├── llk_unpack_tilize.h     # Tilize: convert row-major → tiled layout
│       ├── llk_unpack_untilize.h   # Untilize: convert tiled → row-major layout
│       ├── llk_unpack_reduce.h     # Unpack with reduce configuration
│       │
│       ├── [MATH FILES]
│       ├── llk_math_common.h          # Init / config shared math utilities
│       ├── llk_math_matmul.h          # Matrix multiply (GEMM)
│       ├── llk_math_eltwise_binary.h  # Binary elementwise (add, mul, sub, etc.)
│       ├── llk_math_eltwise_unary_datacopy.h # Unary copy / identity
│       ├── llk_math_eltwise_unary_sfpu.h     # SFPU unary ops (exp, relu, gelu...)
│       ├── llk_math_reduce.h          # Reduction operations (sum, max)
│       │
│       ├── [PACK FILES]
│       ├── llk_pack.h             # Core pack function — write Dest → L1
│       ├── llk_pack_common.h      # Shared pack utilities
│       └── llk_pack_untilize.h    # Pack with untilize (tile → row-major)
│
├── tt_llk_blackhole/            # Blackhole (BH) implementation — same structure
│   ├── common/inc/              # Similar headers with BH-specific extensions
│   │   ├── ckernel.h            # Extended — CSR direct access, LLTT support
│   │   └── ...
│   └── llk_lib/                 # Same API surface, BH-specific implementations
│
├── tt_llk_quasar/               # Quasar — minimal stubs (compilation testing only)
│
├── common/                      # Files shared across all architectures
│   └── inc/                     # (e.g., shared data format definitions)
│
├── tests/                       # Test framework
│   ├── common/                  # Shared test infrastructure
│   ├── helpers/
│   │   └── include/
│   │       └── ckernel_helper.h # Test kernel helpers
│   ├── python_tests/            # Python test orchestration
│   │   ├── conftest.py          # pytest configuration
│   │   ├── helpers/
│   │   │   ├── device.py        # Device interface (talks to hardware)
│   │   │   ├── test_config.py   # TestConfig class — parameterizes tests
│   │   │   ├── utils.py         # Shared test utilities
│   │   │   └── golden/          # Reference (CPU) implementations to compare against
│   │   └── test_*.py            # Test files per operation
│   ├── eltwise/                 # Elementwise op test kernels
│   ├── matmul/                  # Matmul test kernels
│   ├── tilize/                  # Tilize/untilize test kernels
│   └── requirements.txt         # Python deps for tests
│
├── docs/
│   └── llk/
│       ├── l1/intro.md           # Conceptual intro
│       ├── l2/top_level_overview.md  # Architecture overview
│       └── l3/programming_model.md   # Programming model (for kernel writers)
│
├── infra/                       # CI/CD scripts and Docker configs
├── .github/workflows/           # GitHub Actions pipelines
├── CLAUDE.md                    # Instructions for AI coding assistants using this repo
├── CONTRIBUTING.md              # Contribution guide
├── README.md                    # Project overview
└── pyproject.toml               # Python project config (for test tooling)
```

---

## 10. C++ and C Concepts Used in This Repo (With Examples)

This section teaches you every language feature the repo uses. Understanding these is required to read and write LLK code.

### 10.1 Header-Only Libraries (`inline`, no `.cpp`)

All LLK functions are defined directly in `.h` files, not split into `.h` (declaration) and `.cpp` (definition).

**Why:** The TRISC cores compile one translation unit per kernel. Keeping everything in headers means the compiler sees all code and can aggressively inline and optimize.

**Key keyword: `inline`**
```cpp
// In llk_pack.h
inline void _llk_pack_(uint32_t tile_index, uint32_t address) {
    // full implementation here, in the .h file
}
```

`inline` tells the compiler: "place this function's code directly at every call site — no function call overhead." At the hardware level (where every instruction cycle matters), this is critical.

### 10.2 Templates (`template<>`)

Templates let you write code that is parameterized by a **type** or a **constant value**, resolved entirely at compile time.

```cpp
// Template parameterized by a compile-time BOOL
template <bool is_fp32_dest_acc_en = false, bool untilize = false>
inline void _llk_pack_(uint32_t tile_index, uint32_t address) {
    if constexpr (is_fp32_dest_acc_en) {
        // This branch compiled only when is_fp32_dest_acc_en == true
        // Hardware handles FP32 Dest accumulation
    } else {
        // This branch compiled only for BF16 mode
    }
}
```

**`if constexpr`:** Unlike a regular `if`, this is evaluated at compile time. Only the relevant branch is compiled into the binary — no runtime branching overhead.

**`template<auto>` or `template<uint32_t N>`:** Used for compile-time integer constants:
```cpp
template <uint32_t fidelity_phases>
inline void _llk_math_matmul_(uint32_t dst_index) {
    // fidelity_phases is a constant the compiler knows at compile time
    // Can be used as loop bounds, array sizes, etc.
}
```

### 10.3 Namespaces

tt-llk organizes code into C++ namespaces to avoid naming collisions between subsystems.

```cpp
namespace ckernel {           // Everything lives here at the top level
    namespace unpacker {      // Unpacker-specific types and config
        struct unpack_config_t { ... };
    }
    namespace packer {        // Packer-specific types and config
        struct pack_config_t { ... };
    }
    namespace math {          // Math-specific types
        struct alu_config_t { ... };
    }
}
```

Usage:
```cpp
ckernel::unpacker::unpack_config_t cfg;
ckernel::tensix_sync();
```

### 10.4 Structs with Bit Fields

Hardware registers are often packed — one 32-bit register holds multiple fields. Bit fields let you address them by name:

```cpp
// From cunpack_common.h
struct unpack_tile_descriptor_t {
    uint32_t in_data_format  : 4;   // bits [3:0]  — input data format
    uint32_t uncompressed    : 1;   // bit [4]     — is data compressed?
    uint32_t reserved_0      : 3;   // bits [7:5]  — unused
    uint32_t blobs_per_xy_plane : 8; // bits [15:8] — number of blobs
    uint32_t x_dim           : 16;  // bits [31:16] — tile X dimension
    // ... more fields ...
};
```

This struct maps directly onto a hardware configuration register. You fill in the fields, then write the struct's bits to the register address. The compiler packs the bits exactly as specified.

### 10.5 Unions

Unions allow the same memory to be interpreted as different types:

```cpp
union alu_config_t {
    struct {
        uint32_t ALU_FORMAT_SPEC_REG0_SrcA : 4;
        uint32_t ALU_FORMAT_SPEC_REG1_SrcB : 4;
        uint32_t ALU_FORMAT_SPEC_REG2_Dstacc : 4;
        uint32_t reserved : 20;
    };
    uint32_t val;  // Read/write the whole register as one 32-bit value
};

// Usage:
alu_config_t cfg;
cfg.ALU_FORMAT_SPEC_REG0_SrcA = (uint32_t)DataFormat::Float16_b;
cfg.ALU_FORMAT_SPEC_REG2_Dstacc = (uint32_t)DataFormat::Float32;
cfg_write(ALU_ROUNDING_MODE_addr, cfg.val);  // Write as one integer
```

### 10.6 Enums and Enum Classes

Used for type-safe constants:

```cpp
enum class DataFormat : uint8_t {
    Float32   = 0,
    Float16   = 1,
    Bfloat16  = 2,
    Bfp8_b    = 3,
    Bfp4_b    = 4,
    // ...
};

enum class DstSync {
    SyncFull,   // Full sync — Math waits for Pack to finish before writing
    SyncHalf,   // Half sync — Use double buffering, Math and Pack overlap
};

enum class MathFidelity {
    LoFi    = 0,  // Low fidelity (fewer multiply phases, faster)
    HiFi2   = 2,  // Medium fidelity
    HiFi3   = 3,
    HiFi4   = 4,  // Full fidelity (most accurate, slowest)
};
```

### 10.7 `#define` Macros and the Preprocessor

tt-llk uses macros extensively for two reasons:
1. **Hardware instruction emission:** The SFPI compiler uses macros to emit specific Tensix machine instructions
2. **Architecture selection:** `#ifdef ARCH_WORMHOLE` / `#ifdef ARCH_BLACKHOLE` gates architecture-specific code

```cpp
// Macro that expands to a hardware TTI instruction
// "STALLWAIT" = stall until condition P is met
#define TTI_STALLWAIT(p_stall, p_wait) \
    t6_insn(p_stall::STALL_MATH, p_wait::WAIT_SFPU)

// Usage — wait for SFPU to finish before packing:
TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU);

// Architecture gate:
#ifdef ARCH_BLACKHOLE
    // Blackhole-specific code path
    load_replay_buf();
#else
    // Wormhole code path
#endif
```

### 10.8 `constexpr` — Compile-Time Constants

```cpp
constexpr uint32_t TILE_FACE_WIDTH  = 16;
constexpr uint32_t TILE_FACE_HEIGHT = 16;
constexpr uint32_t TILE_WIDTH  = 32;
constexpr uint32_t TILE_HEIGHT = 32;
constexpr uint32_t NUM_FACES_PER_TILE = 4;  // 2x2 grid of 16x16 faces
```

`constexpr` means the value is computed at compile time and can be used in template arguments, array sizes, and `if constexpr` conditions.

### 10.9 Volatile Pointers (MMIO Register Access)

Tenstorrent hardware registers are accessed via **memory-mapped I/O (MMIO)**. The CPU reads/writes a physical memory address, and that read/write actually configures hardware.

```cpp
// volatile tells the compiler: NEVER optimize away this access!
// Hardware actually reads from / writes to this address.
volatile uint32_t* reg_ptr = reinterpret_cast<volatile uint32_t*>(REGISTER_ADDRESS);
*reg_ptr = config_value;  // This MUST happen — don't cache it!

// In ckernel.h:
inline void cfg_write(uint32_t addr32, uint32_t data) {
    volatile uint32_t* cfg = reinterpret_cast<volatile uint32_t*>(TENSIX_CFG_BASE + addr32 * 4);
    *cfg = data;
}
```

Without `volatile`, the compiler might cache the value in a register and never actually write to hardware. That would be catastrophic for correctness.

### 10.10 `reinterpret_cast`

Used when you need to treat a block of memory as a different type — very common when working with hardware registers:

```cpp
unpack_tile_descriptor_t tile_desc;
// ... fill in fields ...

// Write the struct as raw 32-bit words to hardware register address:
uint32_t* dest = reinterpret_cast<uint32_t*>(UNPACK_TILE_DESCRIPTOR_ADDR);
uint32_t* src  = reinterpret_cast<uint32_t*>(&tile_desc);
dest[0] = src[0];
dest[1] = src[1];
```

### 10.11 Global Pointer Variables

The RISC-V cores on Tensix have direct access to hardware through memory-mapped registers. `ckernel.h` sets up global pointer variables:

```cpp
// These point to actual hardware registers:
volatile uint32_t* reg_base;        // General register file base
volatile uint32_t* pc_buf_base;     // Program counter buffer
volatile uint32_t* regfile;         // Register file
volatile uint32_t* mailbox_base[4]; // 4 mailboxes for inter-core messaging
```

### 10.12 Function Pointer Pattern (`_llk_*_` Naming)

You'll notice functions named with leading and trailing underscores: `_llk_unpack_A_()`, `_llk_math_matmul_()`, `_llk_pack_()`.

This is a naming convention: the double-underscore versions are the **low-level internal implementations**. Higher-level wrappers (used by tt-metal) call these and may add logging, error checking, or synchronization on top.

---

## 11. The Core Kernel Layer: `ckernel.h`

`ckernel.h` is the foundation. Every other file in tt-llk depends on it. It provides:

### Global Hardware Pointers
```cpp
namespace ckernel {
    extern volatile uint32_t* reg_base;
    extern volatile uint32_t* regfile;
    extern volatile uint32_t mailbox_base[4];
}
```

### Configuration State Management (Ping-Pong Buffering)

The hardware supports **two configuration states** (0 and 1). This lets you program the next operation's config while the current operation runs — no stall needed.

```cpp
namespace ckernel {
    extern uint32_t cfg_state_id;         // Current state: 0 or 1

    inline void flip_cfg_state_id() {
        cfg_state_id ^= 1;                // Toggle between 0 and 1
    }

    inline void reset_cfg_state_id() {
        cfg_state_id = 0;
    }

    // Get pointer to current hardware config block:
    inline volatile uint32_t* get_cfg_pointer() {
        return cfg_state_id ? cfg_base_ptr_1 : cfg_base_ptr_0;
    }
}
```

Think of it as: config bank 0 is "currently running", bank 1 is "being set up". When setup is done, you flip — bank 1 becomes "currently running" and you set up bank 0 for the next op.

### Register Read/Write
```cpp
inline uint32_t cfg_read(uint32_t addr32) {
    return *(get_cfg_pointer() + addr32);
}

inline void cfg_write(uint32_t addr32, uint32_t data) {
    *(get_cfg_pointer() + addr32) = data;
}

// Read-Modify-Write (change just some bits in a register):
template <uint32_t ADDR, uint32_t SHIFT, uint32_t MASK>
inline void cfg_reg_rmw_tensix(uint32_t val) {
    uint32_t current = cfg_read(ADDR);
    current &= ~(MASK << SHIFT);   // Clear the bits we're changing
    current |= (val << SHIFT);     // Set new value
    cfg_write(ADDR, current);
}
```

### Synchronization Primitives

```cpp
// Block until all in-flight Tensix operations are done
inline void tensix_sync() {
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_MATH_DONE);
}

// Block until all MOP (micro-operation) instructions complete
inline void mop_sync() {
    TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::WAIT_MOP_DONE);
}

// Hardware semaphore operations:
template <uint32_t WaitRes>
inline void t6_semaphore_post(uint32_t index) {
    // Increment semaphore — signal that something is ready
    TTI_SEMINC(WaitRes, index);
}

template <uint32_t WaitRes>
inline void t6_semaphore_get(uint32_t index) {
    // Decrement semaphore — wait until it's >0, then decrement
    TTI_SEMWAIT(WaitRes, index, p_stall::STALL_ON_ZERO);
    TTI_SEMDEC(WaitRes, index);
}
```

### Destination Register Management

```cpp
extern uint32_t dest_offset_id;     // Which Dest half is active: 0 or 1

inline void flip_dest_offset_id() {
    dest_offset_id ^= 1;            // Toggle Dest half
}
```

---

## 12. The Unpacker: Stage 1 Deep Dive

The unpacker is a **DMA engine** that moves tile data from L1 SRAM into the SrcA/SrcB register files, performing format conversion along the way.

### Key Files
- `cunpack_common.h` — Configuration structs, init functions
- `llk_unpack_A.h` — Load one tile into SrcA
- `llk_unpack_AB.h` — Load two tiles into SrcA + SrcB (for binary ops)
- `llk_unpack_AB_matmul.h` — Optimized matmul unpacking
- `llk_unpack_tilize.h` — Convert row-major data into tiled format while unpacking
- `llk_unpack_reduce.h` — Unpack with reduction configuration

### Configuration Structures

```cpp
// Describes the TILE in L1 (input format, dimensions)
struct unpack_tile_descriptor_t {
    uint32_t in_data_format  : 4;   // DataFormat of tile in L1
    uint32_t uncompressed    : 1;   // 1 = raw data, 0 = compressed
    uint32_t reserved_0      : 3;
    uint32_t blobs_per_xy_plane : 8; // Number of BFP exponent blocks
    uint32_t x_dim           : 16;  // Tile width in elements (usually 16)
    // continues...
};

// Describes HOW to unpack (output format for SrcA/SrcB)
struct unpack_config_t {
    uint32_t out_data_format  : 4;  // DataFormat loaded into SrcA/SrcB
    uint32_t throttle_mode    : 2;  // Throttle to avoid stalls
    uint32_t context_count    : 2;  // Number of double-buffer contexts
    uint32_t haloize_mode     : 1;  // For convolution halo operations
    // continues...
};
```

### Main Unpack Functions

**Unpack a single tile to SrcA:**
```cpp
// From llk_unpack_A.h
// address = L1 address of the tile to unpack
// face_r_dim = number of face rows (usually 1 for 16x16 faces)
// within_face_16x16_transpose = rotate the face? (for transpose ops)
inline void _llk_unpack_A_(uint32_t address, 
                            uint32_t face_r_dim = FACE_R_DIM,
                            bool within_face_16x16_transpose = false) {
    // 1. Acquire unpacker context (semaphore wait)
    // 2. Set tile descriptor in hardware config
    // 3. Issue TTI_UNPACR instruction to trigger DMA
    // 4. Wait for completion
    // 5. Signal done (semaphore post)
}
```

**Unpack two tiles (A+B) for binary operations:**
```cpp
// From llk_unpack_AB.h
inline void _llk_unpack_AB_(uint32_t operand_A, uint32_t operand_B,
                              uint32_t transpose_of_faces = 0) {
    // Loads tile at operand_A address into SrcA
    // Loads tile at operand_B address into SrcB
}
```

### Context Switching (Double Buffering in Unpack)

The unpacker supports **two contexts** — it can start loading the next tile into context 1 while math is still using context 0. This hides memory latency.

```cpp
// Managed internally:
uint32_t unpack_src_format;    // Current source format
uint32_t unpack_dst_format;    // Current dest format

void wait_for_next_context() {
    // Wait until the next unpack context is free
    t6_semaphore_get<semaphore::UNPACK_SYNC>(unpack_sync_id);
}
```

---

## 13. The Math Unit: Stage 2 Deep Dive

The Math unit contains two sub-engines:

**FPU (Floating Point Unit) — The Matrix Engine**
- Performs 16×16 matrix multiply in one shot
- Works on SrcA and SrcB register files
- Writes results to Dest register
- Uses "fidelity phases" to control precision

**SFPU (Special Function Processing Unit) — The Vector Engine**
- Operates on values already in Dest
- Does elementwise operations: exp, sqrt, reciprocal, gelu, tanh, etc.
- Works lane-by-lane (vector SIMD style)
- Has its own instruction stream (separate from the RISC-V instruction stream)

### Key Files
- `cmath_common.h` — ALU config struct, math init
- `llk_math_common.h` — Shared math init/uninit
- `llk_math_matmul.h` — GEMM operations
- `llk_math_eltwise_binary.h` — Binary elementwise (add, multiply, etc.)
- `llk_math_eltwise_unary_sfpu.h` — SFPU unary operations

### Math Configuration

```cpp
struct alu_config_t {
    uint32_t ALU_FORMAT_SPEC_REG0_SrcA   : 4;  // Format of SrcA input
    uint32_t ALU_FORMAT_SPEC_REG1_SrcB   : 4;  // Format of SrcB input
    uint32_t ALU_FORMAT_SPEC_REG2_Dstacc : 4;  // Format of Dest accumulator
    uint32_t reserved                    : 20;
};
```

### Matrix Multiply — The Core Operation

The FPU does a **16×16 matrix multiply** (one face × one face) per cycle. To multiply 32×32 matrices, it needs 4 phases (2×2 face decomposition):

```
A (32×32) × B (32×32) = C (32×32)

Implemented as:
C[0,0] += A[0,0] × B[0,0]   (face-level multiply)
C[0,1] += A[0,0] × B[0,1]
C[1,0] += A[1,0] × B[0,0]
C[1,1] += A[1,0] × B[0,1]
```

```cpp
// From llk_math_matmul.h
template <int MATH_FIDELITY_DESC = 0>
inline void _llk_math_matmul_(uint32_t dst_index) {
    // Configure address modifiers for face iteration
    TTI_SETRWC(p_setrwc::CLR_AB, ...);
    
    // Issue matmul instruction for each fidelity phase
    // MATH_FIDELITY_DESC controls how many sub-phases are run
    TTI_MVMUL(p_mvmul::FIDELITY_INCREMENT, dst_index, ...);
}
```

### Math Fidelity

For BFP8 and similar low-precision formats, a single multiply phase doesn't capture all the precision. **Math fidelity** controls how many phases are run:

| Fidelity | Phases | Speed | Accuracy |
|----------|--------|-------|----------|
| LoFi | 1 | Fastest | Lowest |
| HiFi2 | 2 | Medium | Medium |
| HiFi3 | 3 | Slower | Good |
| HiFi4 | 4 | Slowest | Highest |

For Float32 and BFloat16, HiFi4 is used by default (full precision in one effective phase).

---

## 14. The SFPU: The Vector Engine Inside Math

The SFPU is a **vector processing unit** bolted onto the Dest register. It can apply functions elementwise to all 32×32 values in a tile.

### How SFPU Programming Feels

SFPU code looks like C++ but uses special vector types (`vFloat`, `vInt`) and predicated conditionals (`v_if`, `v_elseif`, `v_else`, `v_endif`). These map directly to vector lane predicates — **both branches are always executed** (like old GPU SIMD).

```cpp
// SFPU code (runs on MATH core / TRISC1)
void _sfpu_exp_(uint32_t iterations) {
    for (uint32_t i = 0; i < iterations; i++) {
        vFloat val = dst_reg[0];   // Read from Dest register
        
        v_if (val > OVERFLOW_THRESHOLD) {
            dst_reg[0] = std::numeric_limits<float>::infinity();
        }
        v_elseif (val < UNDERFLOW_THRESHOLD) {
            dst_reg[0] = 0.0f;
        }
        v_else {
            // Polynomial approximation of exp()
            vFloat result = sfpu_exp_core(val);
            dst_reg[0] = result;
        }
        v_endif;
        
        dst_reg++;  // Move to next element in Dest
    }
}
```

**Important: `dst_reg++` instead of `dst_reg[i]`**
This lets the SFPU's internal macro unit handle the iteration automatically. The SFPU records a macro (a short sequence of instructions) and replays it in hardware. The baby RISC-V core is free to do other work while the SFPU replays the macro.

### SFPU Init Pattern

Every SFPU operation needs an init call first (to preload constants into SFPU's internal vector registers):

```cpp
// Must call before any exp operations:
inline void _sfpu_exp_init_() {
    // Load polynomial coefficients into vConst registers:
    vConstFloatPrgm0 = 0.3232325017452239990234375f;  // ln(2) related
    vConstFloatPrgm1 = 1.4545459747314453125f;
    vConstFloatPrgm2 = 2.121212482452392578125f;
}

// Then for each tile:
inline void _sfpu_exp_() {
    // ... uses vConstFloatPrgm0/1/2 that were preloaded ...
}
```

### Available SFPU Operations (Wormhole B0 — 56 total)

Elementwise ops: `exp`, `log`, `sqrt`, `reciprocal`, `sigmoid`, `tanh`, `gelu`, `relu`, `elu`, `leaky_relu`, `hardtanh`, `sin`, `cos`, `abs`, `sign`, `square`, `power`, `add1`, `log1p`, `erfinv`, `dropout`, `topk`, `welfords_reduce_sum`, and more.

---

## 15. The Packer: Stage 3 Deep Dive

The packer reads values from the Dest register, optionally converts format, and writes the resulting tile into L1 SRAM output buffers.

### Key Files
- `cpack_common.h` — Config structs, init functions
- `llk_pack.h` — Core pack function
- `llk_pack_common.h` — Shared pack utilities (sync, dest section management)
- `llk_pack_untilize.h` — Pack with untilize (tile layout → row-major)

### Configuration Structure

```cpp
struct pack_config_t {
    uint32_t in_data_format      : 4;  // Format in Dest (BF16, FP32, etc.)
    uint32_t out_data_format     : 4;  // Format to write to L1
    uint32_t exp_section_size    : 8;  // BFP exponent block size
    uint32_t pack_per_xy_plane   : 8;  // Number of tiles to pack per call
    uint32_t concat_inner_dim_en : 1;  // Enable inner dimension concatenation
    // ... more fields ...
};
```

### The Core Pack Function

```cpp
template <DstSync Dst = DstSync::SyncFull, 
          bool is_fp32_dest_acc_en = false,
          bool untilize = false>
inline void _llk_pack_(uint32_t tile_index, uint32_t address) {
    // tile_index: which tile in Dest to pack (0-15 for BF16, 0-7 for FP32)
    // address:    L1 SRAM address to write the output tile

    // 1. Wait for math to be done (MATH_PACK semaphore)
    _llk_packer_wait_for_math_done_();
    
    // 2. Trigger packer DMA: Dest[tile_index] → L1[address]
    TTI_PACR(p_pacr::B_ROW_SET_OVR_ZERO_FLG, ...);
    
    // 3. Wait for packer to finish
    // 4. (If untilize): convert from tile layout to row-major
}
```

### Dest Section Done

After packing all tiles from one Dest half, you must explicitly "close" that section and signal back to math:

```cpp
template <DstSync Dst = DstSync::SyncFull, bool is_fp32_dest_acc_en = false>
inline void _llk_pack_dest_section_done_() {
    // 1. Flip dest_offset_id (switch to other Dest half)
    flip_dest_offset_id();
    
    // 2. Signal to Math that this Dest half is now free
    t6_semaphore_post<semaphore::MATH_PACK>(pack_sync_id);
}
```

### ReLU in the Packer

A hardware optimization: the packer can apply ReLU during the pack step, for free:

```cpp
struct relu_config_t {
    uint32_t STACC_RELU_ApplyRelu    : 4;  // Which relu mode
    uint32_t STACC_RELU_ReluThreshold: 16; // Threshold value
};

// Called once before packing:
inline void _llk_pack_relu_config_(uint32_t relu_mode, float threshold) {
    relu_config_t cfg;
    cfg.STACC_RELU_ApplyRelu = relu_mode;
    cfg.STACC_RELU_ReluThreshold = float_to_bfp16(threshold);
    cfg_write(STACC_RELU_CONFIG_addr, cfg.val);
}
```

---

## 16. Synchronization: How the Three Stages Talk to Each Other

With three separate RISC-V cores (TRISC0/1/2) running simultaneously, you need explicit synchronization to prevent data corruption. tt-llk uses **hardware semaphores**.

### The Three Semaphores

| Semaphore | Controls | Posted By | Waited By |
|-----------|----------|-----------|-----------|
| `UNPACK_SYNC` | Unpacker context availability | Packer (after pack done) | Unpacker (before starting) |
| `UNPACK_TO_DEST` | Unpack→Math for 32-bit direct | Unpacker | Math |
| `MATH_PACK` | Math→Packer handoff | Math (after filling Dest) | Packer (before reading Dest) |

### Semaphore Operations

```cpp
// Increment (signal): "I'm done, next stage can proceed"
t6_semaphore_post<WaitRes>(semaphore_index);

// Decrement (wait): "Wait until ready, then claim it"
t6_semaphore_get<WaitRes>(semaphore_index);

// Hardware wait instruction (no CPU burn):
TTI_SEMWAIT(wait_res, semaphore_index, p_stall::STALL_ON_ZERO);
```

The RISC-V core **hardware-stalls** (not busy-waits) when waiting on a semaphore. The hardware freezes that TRISC core's pipeline until the semaphore condition is met — no wasted instruction cycles.

### The Full Synchronization Dance

```
TRISC0 (Unpack)          TRISC1 (Math)            TRISC2 (Pack)
──────────────────────────────────────────────────────────────────
Wait UNPACK_SYNC
  ↓
Load tile to SrcA
  ↓
Post UNPACK_SYNC ──────▶ (Math can now run)
                          Wait for SrcA ready
                          Run FPU (SrcA×SrcB→Dest)
                          Run SFPU (on Dest)
                          Post MATH_PACK ─────────▶ Wait MATH_PACK
                                                       ↓
                                                     Read from Dest
                                                     Write to L1
                                                     Post UNPACK_SYNC
                                                       ↓ (back to Unpack)
```

### Double-Buffering the Destination

In `DstSync::SyncHalf` mode, Dest has two halves. Math writes to half 0 while pack reads from half 1, then they swap. This makes math and pack truly concurrent:

```
Time →
Math:  [write half0] [write half1] [write half0] ...
Pack:              [read half1]  [read half0]  [read half1] ...
                   ↑ OVERLAP! Both running simultaneously
```

---

## 17. Address Modifiers and MOPs: The Loop Optimization System

### The Problem: Looping is Expensive

The TRISC RISC-V cores are simple (MIPS R3000-class). A loop iteration costs multiple instructions: compare, branch, update counter, compute address... all overhead. For an operation that needs to loop 16+ times (once per face row), this overhead adds up.

### The Solution: Address Modifiers (ADDR_MOD)

Address modifiers are hardware-configured **auto-increment rules** for the SrcA/SrcB/Dest register counters. Instead of the CPU computing the next address every loop iteration, the hardware does it automatically based on a preset pattern.

tt-llk supports 8 address modifier slots: `ADDR_MOD_0` through `ADDR_MOD_7`.

```cpp
// Configure address modifier 0 to auto-increment SrcA by 1 row each step:
ckernel_template::set_addr_mod_base_addr_0(
    TT_OP_SETRWC(p_setrwc::CLR_NONE, 
                 p_setrwc::CR_AB,    // Which counter to modify
                 0, 0, 0,            // Increment values
                 p_setrwc::SET_ABD)  // What to set
);
```

### MOPs: Micro-Operation Templates

**MOPs (Micro-Operations)** take this further — they let you define a small loop at the hardware level. The TRISC core says "execute this sequence of TTI instructions N times" and the hardware does it, allowing the RISC-V core to issue other work in parallel.

```cpp
// ckernel_template.h — defines a MOP loop
struct ckernel_template {
    // The template defines:
    // - Which instructions to repeat
    // - How many times to repeat (outer/inner loop counts)
    // - Which address modifiers to use at each step
    
    static void set_start_op(uint32_t instrn);
    static void set_loop_op(uint32_t instrn);    // Repeated operation
    static void set_end_op(uint32_t instrn);
    static void program(volatile uint32_t* cfg);
    static void run(uint32_t count);  // Fire the MOP
};
```

**In practice (matmul):**
```cpp
// Set up MOP to run 4 face-level matmuls with auto-incrementing addresses
ckernel_template temp(4 /*outer*/, 4 /*inner*/,
                      TT_OP_MVMUL(p_mvmul::FIDELITY_INCREMENT, ...));
temp.set_addr_mod(0, ADDR_MOD_0, ...);
temp.program(cfg_ptr);

// Single instruction fires 4×4 = 16 matmul operations:
temp.run(1);
```

---

## 18. Multi-Architecture Support: WH vs BH vs Quasar

### Wormhole B0 (WH) — The Primary Reference

- Primary development target, most complete implementation
- 56 SFPU operations
- Standard context switching in unpacker
- 4 packers (`NUM_PACKERS = 4`)
- Boot mode: `BootMode.BRISC` (BRISC boots first, then initializes TRISCs)
- Located: `tt_llk_wormhole_b0/`

### Blackhole (BH) — Enhanced Generation

Same overall architecture but with key enhancements:

| Feature | WH | BH |
|---------|----|----|
| SFPU ops | 56 | 55 (no welfords variants) |
| Replay buffers | No | Yes (`load_replay_buf()`) |
| CSR direct access | No | Yes (`csr_read<T>()`) |
| HW dependency tracking | No | Yes (`set_ttsync_enables()`) |
| LLTT instruction recording | No | Yes (`lltt::record()`, `lltt::replay_insn()`) |

**Blackhole-specific code example:**
```cpp
#ifdef ARCH_BLACKHOLE
// BH-specific: load a sequence into the replay buffer
// Then replay it without re-issuing from the RISC-V core:
lltt::record();
    TTI_UNPACR(...);  // This instruction gets recorded
lltt::stop_record();
lltt::replay_insn(8); // Replay that one instruction 8 times
#endif
```

### Quasar — Newest Architecture (Minimal)

- Stubs only for now — used for compilation testing
- No BRISC; 4 TRISCs (11-14) instead of 3+BRISC
- Boot mode: `BootMode.TRISC`
- Active development — not production-ready yet
- Located: `tt_llk_quasar/`

### How Architecture Selection Works

When you compile a kernel:
1. The build system sets `ARCH_WORMHOLE_B0` or `ARCH_BLACKHOLE` as a compiler define
2. `ckernel.h` uses `#ifdef` to include the right platform headers
3. The right `llk_lib/` directory's headers are pulled in

```makefile
# In build system:
CFLAGS += -DARCH_BLACKHOLE   # or -DARCH_WORMHOLE_B0
```

---

## 19. A Complete Operation Flow: Worked Example (Matmul)

Let's trace a full tile matrix multiply, from L1 input tiles to L1 output tile, through every layer.

### Setup Phase (done once, before the tile loop)

```cpp
// 1. Configure the unpacker for two operands (A and B matrices)
//    Tells unpacker: input tiles are BFP8, convert to BF16 in SrcA/SrcB
configure_unpack_AB<
    false,   // is_fp32_dest_acc_en: use BF16 dest
    false,   // row_pool: not a pooling op
    false,   // fpu_srnd_en: stochastic rounding off
    false    // pack_srnd_en: stochastic rounding off
>(DataFormat::Bfp8_b,   // in_data_format (what's in L1)
  DataFormat::Float16_b, // out_data_format (what goes to SrcA/SrcB)
  TILE_R_DIM, TILE_C_DIM, TILE_R_DIM);

// 2. Configure the math ALU
_llk_math_hw_configure_<false>(DataFormat::Float16_b, DataFormat::Float16_b);

// 3. Configure the packer
//    Reads BF16 from Dest, writes BFP8 to L1
configure_pack<false /*fp32*/, false /*untilize*/>(
    DataFormat::Float16_b, // in_data_format (Dest format)
    DataFormat::Bfp8_b,    // out_data_format (L1 output format)
    TILE_R_DIM, TILE_C_DIM);

// 4. Initialize the matmul address modifiers and MOP template
_llk_math_matmul_init_<MathFidelity::HiFi4>(TILE_R_DIM, TILE_C_DIM, TILE_R_DIM);
```

### Tile Loop (repeated for each output tile)

```cpp
for (uint32_t i = 0; i < num_output_tiles; i++) {
    
    //-------- TRISC0 (Unpacker) --------
    // Load matrix A tile from L1 into SrcA
    _llk_unpack_AB_(tile_A_addr, tile_B_addr);
    
    //-------- TRISC1 (Math) --------
    // Multiply SrcA × SrcB → Dest[dst_index]
    // HiFi4 = full precision (4 fidelity phases)
    _llk_math_matmul_<MathFidelity::HiFi4>(dst_index);
    
    //-------- TRISC2 (Packer) --------
    // Write Dest[dst_index] to output L1 address
    _llk_pack_<DstSync::SyncHalf, false /*fp32*/, false /*untilize*/>(
        dst_index, output_addr);
    
    // Signal that packing is done for this tile
    _llk_pack_dest_section_done_<DstSync::SyncHalf, false>();
    
    // Update addresses for next tile...
}
```

All three stages run concurrently via the semaphore system. While TRISC0 is loading tile N+1, TRISC1 is computing tile N, and TRISC2 is packing tile N-1.

---

## 20. How tt-llk Connects to tt-metal (the Full Stack)

```
User Code (Python/C++)
       │
    tt-nn (torch-like ops: linear, conv2d, softmax...)
       │
    tt-metal (the SDK / runtime)
       │  Manages: kernel compilation, CB allocation,
       │           NOC programming, device management
       │
    Compute kernels (.cpp files, compiled for TRISC0/1/2)
       │  These kernels call:
       ▼
    tt-llk (the library YOU are studying)
    llk_unpack_*(), llk_math_*(), llk_pack_*()
       │
    SFPI Compiler (riscv-tt-elf-g++)
    Compiles RISC-V + TTI instructions
       │
    Tensix Hardware
```

### The Three Kernel Types in tt-metal

Every kernel task on a Tensix core involves three separate kernel programs:

**Reader kernel** (runs on BRISC/DM0):
```cpp
void kernel_main() {
    uint32_t tile_addr = get_arg_val<uint32_t>(0);
    cb_reserve_back(cb_in0, 1);           // Reserve space in circular buffer
    noc_async_read_tile(tile_addr, ...);   // Pull from DRAM via NOC
    noc_async_read_barrier();              // Wait for NOC read to complete
    cb_push_back(cb_in0, 1);             // Signal: tile is ready in CB
}
```

**Compute kernel** (runs on TRISC0/1/2, uses tt-llk):
```cpp
namespace NAMESPACE {
void MAIN {
    // LLK calls go here — routed to the right TRISC by the runtime
    init_sfpu(cb_in0, cb_out0);
    
    cb_wait_front(cb_in0, 1);    // Wait for reader to push a tile
    tile_regs_acquire();         // Acquire Dest for writing (TRISC0+1)
    
    copy_tile(cb_in0, 0, 0);     // Unpack + math (TRISC0+1 do this)
    exp_tile(0);                  // SFPU exp on tile 0 (TRISC1)
    
    tile_regs_commit();          // Signal: Dest is filled (TRISC1)
    tile_regs_wait();            // Wait for Dest to be ready (TRISC2)
    
    cb_reserve_back(cb_out0, 1);
    pack_tile(0, cb_out0);       // Pack Dest → CB (TRISC2)
    cb_pop_front(cb_in0, 1);
    tile_regs_release();         // Release Dest
    cb_push_back(cb_out0, 1);   // Signal: output tile ready
}
}
```

**Writer kernel** (runs on NCRISC/DM1):
```cpp
void kernel_main() {
    cb_wait_front(cb_out0, 1);          // Wait for compute to push result
    noc_async_write_tile(out_addr, ...); // Send to DRAM via NOC
    noc_async_write_barrier();
    cb_pop_front(cb_out0, 1);
}
```

### What `tile_regs_acquire/commit/wait/release` Actually Are

These are the high-level wrappers over the semaphore system:

```
tile_regs_acquire()  →  TRISC0 and TRISC1: wait for MATH_PACK semaphore
                         (ensures Dest is free from previous pack)

tile_regs_commit()   →  TRISC1 signals MATH_PACK semaphore
                         (tells TRISC2 "Dest is full, start packing")

tile_regs_wait()     →  TRISC2 waits for MATH_PACK semaphore

tile_regs_release()  →  TRISC2 signals that Dest is free again
```

---

## 21. The Testing Framework

tt-llk has a Python-based test framework that compiles kernels, runs them on hardware (or a simulator), and compares results against a golden CPU reference.

### Test Architecture

```
tests/
├── python_tests/
│   ├── conftest.py              # pytest fixtures — device setup, teardown
│   ├── helpers/
│   │   ├── device.py            # Device class — wraps tt-metal device API
│   │   ├── test_config.py       # TestConfig — parameterizes format sweeps
│   │   ├── utils.py             # Data generation, comparison utilities
│   │   └── golden/              # CPU reference implementations
│   │       ├── golden_matmul.py # NumPy matmul for comparison
│   │       ├── golden_eltwise.py
│   │       └── ...
│   └── test_eltwise_unary.py    # Example test file
│
└── eltwise/
    ├── kernels/
    │   └── eltwise_unary.cpp    # The actual LLK kernel being tested
    └── CMakeLists.txt
```

### TestConfig — The Parameterization System

```python
class TestConfig:
    def __init__(self,
                 in_df: DataFormat,     # Input data format
                 out_df: DataFormat,    # Output data format
                 math_fidelity: MathFidelity,
                 num_tiles: int = 1):
        self.in_df = in_df
        self.out_df = out_df
        self.math_fidelity = math_fidelity
        self.num_tiles = num_tiles
```

Tests are parameterized across all relevant format combinations:
```python
@pytest.mark.parametrize("in_df,out_df", [
    (DataFormat.Float16_b, DataFormat.Float16_b),
    (DataFormat.Bfp8_b, DataFormat.Float16_b),
    (DataFormat.Float32, DataFormat.Float32),
])
def test_exp(in_df, out_df, device):
    config = TestConfig(in_df, out_df, MathFidelity.HiFi4)
    run_llk_test(device, "exp", config)
    compare_with_golden("exp", config)
```

### Writing a New Test

1. Create the kernel `.cpp` file in `tests/<operation>/kernels/`
2. Call LLK functions (init → unpack → math → pack)
3. Create a Python test in `tests/python_tests/test_<operation>.py`
4. Implement the CPU golden reference in `tests/python_tests/helpers/golden/`
5. Add to CMakeLists.txt

---

## 22. The Build System and SFPI Toolchain

### The SFPI Compiler

Standard `g++` cannot compile tt-llk because some constructs (`TTI_*` macros, `vFloat`, `dst_reg`) require the **SFPI compiler**: `riscv-tt-elf-g++`.

This is a custom RISC-V GCC toolchain that:
- Targets the Tensix RISC-V cores (RV32IM + Tenstorrent extensions)
- Understands `TTI_*` macros and translates them to Tensix machine instructions
- Supports the `sfpi::` namespace for SFPU vector programming
- Is downloaded automatically from `https://github.com/tenstorrent/sfpi`

### Compilation Flow

```
Your kernel .cpp
        │
        ▼ riscv-tt-elf-g++ (SFPI compiler)
   Preprocess (#include, #ifdef, #define expansion)
        │
        ▼ Compile
   RISC-V assembly + Tensix TTI instructions
        │
        ▼ Link
   kernel_trisc0.elf  (for TRISC0 / Unpacker)
   kernel_trisc1.elf  (for TRISC1 / Math)
   kernel_trisc2.elf  (for TRISC2 / Pack)
        │
        ▼ tt-metal runtime loads these to Tensix L1 and boots the TRISCs
```

### Setting Up the Environment

```bash
# Clone the repo
git clone https://github.com/tenstorrent/tt-llk.git
cd tt-llk

# Set up the test environment (installs SFPI, Python deps, pre-commit hooks)
cd tests && pip install -r requirements.txt

# SFPI toolchain is automatically downloaded during build
# from https://github.com/tenstorrent/sfpi

# Run a test (requires Tenstorrent hardware or QEMU):
pytest tests/python_tests/test_eltwise_unary.py -k "exp"
```

### Key Build Files

```
pyproject.toml         — Python project metadata, tool versions
package.json           — Node.js dependencies (for test infra scripts)
.pre-commit-config.yaml — Code quality hooks (clang-format, codespell, yamllint)
.clang-format          — C++ formatting rules
.clangd                — clangd LSP configuration
setup_clangd.sh        — Script to configure IDE for the SFPI toolchain paths
```

---

## 23. How to Write Your Own Custom Kernel

Here is a step-by-step guide to adding a new operation (example: element-wise `square` = x²).

### Step 1: Write the SFPU Operation (if not already in llk_math_eltwise_unary_sfpu.h)

In `tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_unary_sfpu.h`, add:

```cpp
// Init: preload any constants
inline void _sfpu_square_init_() {
    // No special constants needed for x² — it's just x * x
}

// Per-tile: compute x² for all elements in Dest
inline void _sfpu_square_() {
    // Using SFPI: operate on dst_reg elements
    for (uint32_t i = 0; i < 16; i++) {   // 16 elements per SFPU pass
        vFloat val = dst_reg[0];
        dst_reg[0] = val * val;
        dst_reg++;
    }
}
```

### Step 2: Expose via LLK API (add to llk_math_eltwise_unary_sfpu.h)

```cpp
// Init (call once before the tile loop)
template <bool APPROXIMATE = false>
inline void llk_math_eltwise_unary_sfpu_square_init() {
    llk_math_eltwise_unary_sfpu_init<APPROXIMATE>(_sfpu_square_init_);
}

// Per-tile call (inside the tile loop)
template <bool APPROXIMATE = false>
inline void llk_math_eltwise_unary_sfpu_square(uint32_t dst_index) {
    llk_math_eltwise_unary_sfpu_0_param<APPROXIMATE>(
        ckernel::sfpu::_sfpu_square_,   // The SFPU function
        ckernel::sfpu::_sfpu_square_,
        dst_index, 
        (int)VectorMode::RC             // Which elements (full tile)
    );
}
```

### Step 3: Write the Compute Kernel

Create `tests/square/kernels/square_kernel.cpp`:

```cpp
#include "llk_math_eltwise_unary_sfpu.h"
#include "llk_unpack_A.h"
#include "llk_pack.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    // ── INIT PHASE ──────────────────────────────────────────
    // Configure unpacker: BF16 → BF16 (no conversion)
    llk_unpack_A_hw_configure_disaggregated<false>(
        DataFormat::Float16_b, DataFormat::Float16_b);
    
    // Configure math
    llk_math_eltwise_unary_sfpu_square_init();
    
    // Configure packer: BF16 → BF16
    llk_pack_hw_configure_disaggregated<false>(
        DataFormat::Float16_b, DataFormat::Float16_b);
    
    llk_pack_dest_init<false, DstSync::SyncHalf>();
    
    // ── TILE LOOP ────────────────────────────────────────────
    for (uint32_t tile = 0; tile < num_tiles; tile++) {
        cb_wait_front(tt::CBIndex::c_0, 1);    // Wait for input tile
        
        // Stage 1: Unpack to SrcA
        llk_unpack_A(tt::CBIndex::c_0, 0);
        
        // Stage 2: Copy SrcA→Dest, then run SFPU square
        tile_regs_acquire();
        llk_math_eltwise_unary_datacopy<A2D, BroadcastType::NONE, false>(0);
        llk_math_eltwise_unary_sfpu_square<false>(0);
        tile_regs_commit();
        
        // Stage 3: Pack Dest→output CB
        tile_regs_wait();
        cb_reserve_back(tt::CBIndex::c_16, 1);
        llk_pack<false, DstSync::SyncHalf, false>(0, tt::CBIndex::c_16);
        llk_pack_dest_section_done<DstSync::SyncHalf, false>();
        cb_pop_front(tt::CBIndex::c_0, 1);
        tile_regs_release();
        cb_push_back(tt::CBIndex::c_16, 1);
    }
}
} // namespace NAMESPACE
```

### Step 4: Write the Python Test

Create `tests/python_tests/test_square.py`:

```python
import pytest
import torch
from helpers.test_config import TestConfig
from helpers.device import Device

def golden_square(tensor):
    return tensor * tensor   # CPU reference

@pytest.mark.parametrize("in_df", [DataFormat.Float16_b, DataFormat.Bfp8_b])
def test_square(in_df, device: Device):
    # Generate random input
    input_tensor = torch.randn(32, 32)
    
    # Run on Tenstorrent hardware via tt-metal
    config = TestConfig(in_df=in_df, out_df=DataFormat.Float16_b,
                        math_fidelity=MathFidelity.HiFi4)
    result = device.run_kernel("square_kernel", input_tensor, config)
    
    # Compare with CPU reference
    expected = golden_square(input_tensor)
    assert torch.allclose(result, expected, atol=1e-2)
```

---

## 24. Quick Reference Cheat Sheet

### The Three-Stage Call Pattern (Every Operation Uses This)

```
INIT:   configure_unpack_*()  → _llk_math_hw_configure_()  → configure_pack()
LOOP:   _llk_unpack_*_()      → _llk_math_*_()             → _llk_pack_()
END:    _llk_unpack_*_uninit_()                              → _llk_pack_uninit_()
```

### Key Files by Subsystem

| What you want | File |
|--------------|------|
| Hardware primitives, semaphores | `common/inc/ckernel.h` |
| Unpacker config structs | `common/inc/cunpack_common.h` |
| Packer config structs | `common/inc/cpack_common.h` |
| Unpack one tile (SrcA) | `llk_lib/llk_unpack_A.h` |
| Unpack two tiles (SrcA+SrcB) | `llk_lib/llk_unpack_AB.h` |
| Matmul (GEMM) | `llk_lib/llk_math_matmul.h` |
| Binary elementwise (add/mul/sub) | `llk_lib/llk_math_eltwise_binary.h` |
| SFPU unary ops (exp/relu/gelu) | `llk_lib/llk_math_eltwise_unary_sfpu.h` |
| Pack Dest → L1 | `llk_lib/llk_pack.h` |
| MOP/template system | `common/inc/ckernel_template.h` |
| TTI instruction macros | `common/inc/ckernel_ops.h` |

### Data Format Constants

| DataFormat Enum | Value | Description |
|----------------|-------|-------------|
| `Float32` | 0 | 32-bit IEEE float |
| `Float16` | 1 | 16-bit IEEE half |
| `Bfloat16` | 2 | 16-bit Brain float |
| `Bfp8_b` | 3 | 8-bit block float |
| `Bfp4_b` | 4 | 4-bit block float |
| `Int32` | 8 | 32-bit integer |

### DstSync Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `SyncFull` | Pack waits for all math before reading Dest | Simple ops, correctness first |
| `SyncHalf` | Double-buffer Dest; math+pack overlap | Performance-critical paths |

### Math Fidelity

| Fidelity | Phases | Use For |
|----------|--------|---------|
| `LoFi` | 1 | Exploration, very coarse |
| `HiFi2` | 2 | BFP8 inference |
| `HiFi3` | 3 | Mixed precision |
| `HiFi4` | 4 | Full precision, BF16/FP32 |

### Semaphore Names

| Semaphore | Index | Meaning |
|-----------|-------|---------|
| `UNPACK_SYNC` | 0 | Unpacker context ready |
| `UNPACK_TO_DEST` | 1 | Unpack→Dest for FP32 |
| `MATH_PACK` | 2 | Math→Packer handoff |

### Architecture Defines

| Define | Architecture |
|--------|-------------|
| `ARCH_WORMHOLE_B0` | Wormhole B0 |
| `ARCH_BLACKHOLE` | Blackhole |
| `ARCH_QUASAR` | Quasar |

---

## Appendix: Glossary

| Term | Meaning |
|------|---------|
| **LLK** | Low Level Kernel — this library |
| **Tensix** | Tenstorrent's compute tile/processor |
| **RISC-V** | Open ISA used by all 5 cores inside a Tensix core |
| **TRISC** | Tensix RISC — the 3 cores driving Unpack/Math/Pack |
| **BRISC** | Broadside RISC — data movement core 0 (reader) |
| **NCRISC** | Network RISC — data movement core 1 (writer) |
| **FPU** | Floating Point Unit — the matrix multiply engine |
| **SFPU** | Special Function Processing Unit — the vector engine |
| **SrcA/SrcB** | Input register files for the FPU |
| **Dest** | Output register — FPU writes here, SFPU reads/writes here |
| **TTI** | Tensix Tightly-coupled Instruction — native HW instruction |
| **MOP** | Micro-Operation — hardware loop template |
| **NOC** | Network on Chip — connects Tensix cores and DRAM |
| **CB** | Circular Buffer — ring buffer in L1 for tile handoff |
| **MMIO** | Memory Mapped I/O — accessing hardware registers via memory reads/writes |
| **BFP** | Block Floating Point — shared exponent compression format |
| **Tile** | 32×32 matrix of numbers — fundamental compute unit |
| **Face** | 16×16 sub-tile — fundamental hardware unit of the FPU |
| **SFPI** | Stream Float Processing Interface — the custom RISC-V compiler toolchain |
| **DstSync** | Destination synchronization mode (Full or Half-buffer) |
| **Fidelity** | Number of multiply sub-phases — trades speed for precision |
| **WH** | Wormhole (Wormhole B0 chip) |
| **BH** | Blackhole chip |

---

*Guide compiled from: tenstorrent/tt-llk source, DeepWiki, Metalium Guide, FOSDEM 2026 talk by Martin Chang (ex-Tenstorrent), and official Tenstorrent documentation.*
