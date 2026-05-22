# Tenstorrent Architecture 

> **Goal:** Understand how Tenstorrent executes AI models at the silicon level by optimising data movement, and why it avoids HBM while the rest of the industry uses it.

**Official sources used throughout this document:**
- GitHub: [tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal)
- GitHub: [tenstorrent/tt-forge](https://github.com/tenstorrent/tt-forge)
- GitHub: [tenstorrent/tt-llk](https://github.com/tenstorrent/tt-llk)
- GitHub: [tenstorrent/tt-lang](https://github.com/tenstorrent/tt-lang)
- Tenstorrent documentation: [docs.tenstorrent.com](https://docs.tenstorrent.com)

---

## Table of Contents

- [Phase 1 — Foundations](#phase-1--foundations)
  - [1.1 The Memory Wall Problem](#11-the-memory-wall-problem)
  - [1.2 Memory Hierarchy Fundamentals](#12-memory-hierarchy-fundamentals)
  - [1.3 HBM, GDDR6, and High Bandwidth Flash](#13-hbm-gddr6-and-high-bandwidth-flash)
  - [1.4 How a GPU Executes an LLM (the baseline)](#14-how-a-gpu-executes-an-llm-the-baseline)
- [Phase 2 — Tenstorrent Silicon Architecture](#phase-2--tenstorrent-silicon-architecture)
  - [2.1 The Tensix Core](#21-the-tensix-core)
  - [2.2 Network-on-Chip (NoC) — the 2D Mesh](#22-network-on-chip-noc--the-2d-mesh)
  - [2.3 Blackhole Chip — Full Die Floorplan](#23-blackhole-chip--full-die-floorplan)
  - [2.4 Distributed SRAM vs Central HBM](#24-distributed-sram-vs-central-hbm)
  - [2.5 Chip Generations — Grayskull, Wormhole, Blackhole](#25-chip-generations--grayskull-wormhole-blackhole)
- [Phase 3 — Dataflow Execution Model](#phase-3--dataflow-execution-model)
  - [3.1 Von Neumann vs Dataflow Architectures](#31-von-neumann-vs-dataflow-architectures)
  - [3.2 Spatial Pipelining — the Core Technique](#32-spatial-pipelining--the-core-technique)
  - [3.3 Full LLM Forward Pass — Step by Step](#33-full-llm-forward-pass--step-by-step)
  - [3.4 Explicit Memory Management — No Hardware Prefetcher](#34-explicit-memory-management--no-hardware-prefetcher)
- [Phase 4 — Compiler and Software Stack](#phase-4--compiler-and-software-stack)
  - [4.1 Compiler Fundamentals (prerequisite)](#41-compiler-fundamentals-prerequisite)
  - [4.2 TT-Forge — the ML Compiler](#42-tt-forge--the-ml-compiler)
  - [4.3 TTNN — the Tensor Operation Library](#43-ttnn--the-tensor-operation-library)
  - [4.4 TT-Metalium — Bare-Metal Programming SDK](#44-tt-metalium--bare-metal-programming-sdk)
  - [4.5 TT-LLK — Low-Level Kernels](#45-tt-llk--low-level-kernels)
  - [4.6 Deterministic Routing — How the Compiler Hardcodes Data Paths](#46-deterministic-routing--how-the-compiler-hardcodes-data-paths)
  - [4.7 TT-Lang — the Future Programming Model](#47-tt-lang--the-future-programming-model)
- [Phase 5 — Scale-Out: Multi-Chip and Cluster](#phase-5--scale-out-multi-chip-and-cluster)
  - [5.1 Ethernet Scale-Out — NoC Beyond the Chip](#51-ethernet-scale-out--noc-beyond-the-chip)
  - [5.2 Galaxy System — the Server Rack as One Chip](#52-galaxy-system--the-server-rack-as-one-chip)
  - [5.3 Model Parallelism Strategies](#53-model-parallelism-strategies)
- [Phase 6 — Advanced Topics and Trade-offs](#phase-6--advanced-topics-and-trade-offs)
  - [6.1 Quantisation and Data Formats](#61-quantisation-and-data-formats)
  - [6.2 When Tenstorrent Loses — Honest Limits](#62-when-tenstorrent-loses--honest-limits)
  - [6.3 Competitive Landscape](#63-competitive-landscape)
  - [6.4 Reading Primary Sources](#64-reading-primary-sources)
- [Quick Reference](#quick-reference)

---

# Phase 1 — Foundations

## 1.1 The Memory Wall Problem

### What it is

A processor can do math operations (multiplications, additions) much faster than memory can deliver the numbers to work on. This gap has been growing since the 1990s. It is called the **memory wall**.

Numbers that make this concrete (2024 hardware):

| Hardware | Peak Compute | Peak Memory BW | Arithmetic Intensity needed to saturate compute |
|---|---|---|---|
| NVIDIA H100 SXM | 989 TFLOP/s (FP16) | 3,350 GB/s | ~295 FLOP/byte |
| Tenstorrent Blackhole | 786 TFLOP/s (FP16) | 576 GB/s | ~1,364 FLOP/byte |
| AMD MI300X | 1,307 TFLOP/s (FP16) | 5,300 GB/s | ~246 FLOP/byte |

**Arithmetic intensity** = how many math operations you do per byte of data you load from memory.

```
Arithmetic Intensity = FLOP / bytes_accessed
```

If your workload has lower arithmetic intensity than what the chip needs to stay busy, the chip sits idle waiting for data. That is a **memory-bound** workload.

### The Roofline Model

The roofline model tells you whether a workload is compute-bound or memory-bound.

```
Performance (FLOP/s)
        |
        |                      _______________  <- compute peak (flat ceiling)
        |                 ____/
        |            ____/
        |       ____/   <- memory bandwidth slope
        |  ____/
        | /
        |/________________________ Arithmetic Intensity (FLOP/byte)
                ^
                ridge point: the crossover
```

- Left of the ridge point: **memory-bound**. Buying more compute does nothing. You need more bandwidth or less data movement.
- Right of the ridge point: **compute-bound**. Buying more memory bandwidth does nothing.

### Where LLM inference falls

LLM inference (specifically the **autoregressive decode** step — generating one token at a time) is **strongly memory-bound**.

Why: During decode, you process one token but you must load the entire model's weight matrices and the entire KV cache from memory to produce that one token. The number of bytes loaded vastly exceeds the number of FLOPs done.

A rough example with LLaMA-2 70B (FP16):
- Model weights: ~140 GB
- Per token, nearly the full weight set is touched
- FLOPs per token: ~140 GFLOPs
- Arithmetic intensity: ~1 FLOP/byte — extremely low

This means the chip's math units are idle 99% of the time waiting for data, on most hardware.

### How different architectures respond

```
Problem: Math units starve waiting for data
                        |
         _______________+_______________
        |                               |
  NVIDIA / AMD approach         Tenstorrent approach
  Brute-force bandwidth:        Reduce data movement:
  Use HBM (3.35 TB/s)          Keep data in SRAM near
  Stack DRAM right on chip      the math units
  Very expensive ($25K+)        Cheaper GDDR6 + massive SRAM
```

---

## 1.2 Memory Hierarchy Fundamentals

Every computer system has a hierarchy of memory types. They trade off speed, capacity, and cost.

```
SPEED (fast to slow) / COST (expensive to cheap) / CAPACITY (small to large)

  ┌─────────────────────────────────────────────────────┐
  │  REGISTERS                                          │
  │  Speed: ~1 cycle   Size: bytes   Location: in core  │
  ├─────────────────────────────────────────────────────┤
  │  SRAM (L1/L2/L3 cache or dedicated)                 │
  │  Speed: 1–10 ns    Size: KB–MB   Location: on chip  │
  ├─────────────────────────────────────────────────────┤
  │  DRAM (HBM / GDDR6 / LPDDR5)                        │
  │  Speed: 50–100 ns  Size: GB      Location: off chip │
  ├─────────────────────────────────────────────────────┤
  │  Flash / NVMe SSD (PCIe 5.0, NVMe, HBF)            │
  │  Speed: 0.1–1 ms   Size: TB      Location: storage  │
  └─────────────────────────────────────────────────────┘
```

### SRAM (Static Random Access Memory)

- Built from 6 transistors per bit.
- Fast because access does not require a refresh cycle.
- No capacitor, so data does not leak and does not need recharging.
- Power-hungry and large per bit — this limits how much you can fit on a chip.
- On Tenstorrent Blackhole: each of 120 Tensix cores has 1.5 MB = **180 MB total on-chip SRAM**.
- On NVIDIA H100: 50 MB of L2 cache + 256 KB L1 per SM.

### DRAM (Dynamic Random Access Memory)

- Built from 1 transistor + 1 capacitor per bit.
- Much denser than SRAM — far more capacity per mm².
- Requires periodic refresh (the capacitor leaks charge) — this adds latency.
- Sits off the chip die, connected via buses.
- Two main variants for AI chips: **HBM** and **GDDR6** (covered in detail in section 1.3).

### NAND Flash / SSD

- Stores data as charge in floating-gate transistors.
- Non-volatile: data persists without power.
- Used as storage only (model weight files, datasets).
- During LLM execution, flash is only accessed once at startup to load weights into DRAM.
- **High Bandwidth Flash (HBF)** is just a marketing term for fast PCIe 5.0 NVMe drives. They load faster (up to ~14 GB/s), but they are still storage — they are never accessed during forward pass computation.

### Cache hierarchy in CPUs (what Tenstorrent replaces)

Standard CPUs use caches as hardware-managed buffers between registers and DRAM:

```
CPU core
  └── L1 cache (256 KB, ~4 ns)    ← checked first
        └── L2 cache (1–4 MB, ~12 ns)
              └── L3 cache (16–64 MB, ~40 ns)  ← shared across cores
                    └── DRAM (~80 ns)
```

The hardware **prefetcher** watches your access patterns and tries to guess what data you'll need next, loading it into cache before you ask.

Tenstorrent has **no hardware cache hierarchy and no hardware prefetcher**. Instead, the RISC-V processors inside each Tensix core issue explicit instructions to move exactly the data they need. This is covered in section 3.4.

---

## 1.3 HBM, GDDR6, and High Bandwidth Flash

### HBM — High Bandwidth Memory

HBM is DRAM that is physically stacked on top of the processor die using Through-Silicon Vias (TSVs).

```
Physical construction of HBM:
                                                    
  ┌─────────────────────────────────┐               
  │         GPU / AI chip die       │               
  │                                 │               
  │    ┌───────┐     ┌───────┐      │               
  │    │  HBM  │     │  HBM  │      │               
  │    │ stack │     │ stack │      │               
  │    │[DRAM4]│     │[DRAM4]│      │               
  │    │[DRAM3]│     │[DRAM3]│      │               
  │    │[DRAM2]│     │[DRAM2]│      │               
  │    │[DRAM1]│     │[DRAM1]│      │               
  │    │[BASE ]│     │[BASE ]│      │               
  │    └──↕↕↕↕┘     └──↕↕↕↕┘      │               
  │       TSVs          TSVs        │               
  │   ┌─────────────────────────┐   │               
  │   │    Silicon Interposer   │   │               
  │   └─────────────────────────┘   │               
  └─────────────────────────────────┘               
```

- DRAM layers are stacked vertically and connected via TSVs (tiny vertical copper pillars).
- The interposer is a separate piece of silicon that routes the thousands of connections between the HBM stack and the GPU chip.
- The entire assembly is called 2.5D or 3D packaging (CoWoS at TSMC, InFO at others).
- Interface width: 1024 bits per HBM stack (vs 32 bits for GDDR6).
- This wide interface is why HBM has much higher bandwidth.

**H100 SXM HBM3 numbers:**
- 6 HBM3 stacks × 80 GB each = 80 GB total
- Bandwidth: 3,350 GB/s
- Manufacturing cost: the interposer is expensive. H100 SXM costs ~$30,000+ at retail.

### GDDR6 — Graphics Double Data Rate 6

GDDR6 is standard DRAM mounted on the circuit board beside the chip, connected via PCB traces.

```
Physical construction of GDDR6:

  ┌─────────────────────────────────────────────┐
  │                  PCB (board)                │
  │                                             │
  │  ┌──────────┐    ┌────┐ ┌────┐ ┌────┐ ┌────┐ │
  │  │  AI chip │    │GDDR│ │GDDR│ │GDDR│ │GDDR│ │
  │  │          │    │ 6  │ │ 6  │ │ 6  │ │ 6  │ │
  │  └────┬─────┘    └──┬─┘ └──┬─┘ └──┬─┘ └──┬─┘ │
  │       └─────────────┴──────┴──────┴──────┘   │
  │              PCB trace connections             │
  └─────────────────────────────────────────────┘
```

- Each GDDR6 chip has a 32-bit interface. You connect many chips in parallel.
- Tenstorrent Wormhole n300: 24 GB GDDR6, 576 GB/s bandwidth.
- Cheaper to manufacture — no interposer needed.
- Lower bandwidth than HBM, but Tenstorrent's architecture is designed to need less of it.

### Bandwidth comparison

| Memory type | Bandwidth | Capacity (typical AI card) | Cost driver |
|---|---|---|---|
| HBM3 (H100) | 3,350 GB/s | 80 GB | Silicon interposer |
| HBM3e (MI300X) | 5,300 GB/s | 192 GB | Larger interposer |
| GDDR6 (Wormhole n300) | 576 GB/s | 24 GB | Standard PCB |
| LPDDR5 (edge devices) | ~68 GB/s | 16–32 GB | Mobile standard |

### High Bandwidth Flash (HBF)

HBF is an emerging technology from SanDisk/SK Hynix:
- Stacks NAND flash layers similar to how HBM stacks DRAM.
- Target: huge capacity (TB-scale) for models too large to fit in DRAM.
- Still fundamentally storage — higher latency than any DRAM variant.
- **Tenstorrent does not use HBF.** Flash of any kind is cold storage only — models load from it into GDDR6 at startup, then flash is not accessed again during execution.

---

## 1.4 How a GPU Executes an LLM (the baseline)

Before understanding Tenstorrent's approach, you need to understand what a GPU does. This is your reference point.

### GPU hardware structure

```
NVIDIA H100 (simplified)

  ┌───────────────────────────────────────────────────────┐
  │                         H100 Die                       │
  │                                                        │
  │  ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐   │
  │  │  SM  ││  SM  ││  SM  ││  SM  ││  SM  ││  SM  │   │
  │  │  #1  ││  #2  ││  #3  ││  #4  ││ ...  ││ #132 │   │
  │  └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘   │
  │                                                        │
  │              L2 Cache (50 MB shared)                   │
  │                                                        │
  │        Memory Controllers (6 HBM stacks)               │
  └───────────────────────────────────────────────────────┘
          │          │          │
       HBM3        HBM3        HBM3
       stack       stack       stack
       (×6 total, 80 GB, 3.35 TB/s)
```

Each **Streaming Multiprocessor (SM)** contains:
- 128 CUDA cores (FP32 ALUs)
- 4 Tensor Cores (matrix multiply units)
- 256 KB register file
- 256 KB L1 cache / shared memory (configurable split)
- Warp schedulers

### CUDA execution model

```
Thread hierarchy:

  Grid (entire kernel launch)
  └── Block (runs on one SM)
        └── Warp (32 threads, execute in lockstep)
              └── Thread (one CUDA core)
```

- A **warp** is the hardware execution unit: 32 threads that run the same instruction simultaneously (SIMT = Single Instruction Multiple Threads).
- The SM schedules warps. When one warp stalls waiting for memory, the SM switches to another warp — this is **latency hiding**.
- The SM needs enough in-flight warps to keep the tensor cores busy. This is called **occupancy**.

### How an LLM transformer layer executes on a GPU

A transformer layer has these main operations:
1. QKV projection (3 matrix multiplications)
2. Attention scores (Q × K^T)
3. Softmax
4. Attention output (scores × V)
5. Output projection
6. FFN layer 1 (matrix multiply + activation)
7. FFN layer 2 (matrix multiply)
8. LayerNorm

```
Execution flow on GPU for one transformer layer:

  DRAM (HBM)                GPU SM
  ─────────────             ──────────────────────────────
  
  Weight matrices  ──────►  Load W_Q, W_K, W_V into L1
  (W_Q, W_K, W_V)           ↓
                             MatMul: Input × W_Q → Q
                             MatMul: Input × W_K → K   (Tensor Cores)
                             MatMul: Input × W_V → V
                             ↓
  KV Cache ────────────────► Load K_cache, V_cache
  (in HBM)                   ↓
                             Q × K^T → attention scores
                             Softmax (vector operation)
                             scores × V → attention output
                             ↓
  Result written ◄──────────  Write intermediate result back to HBM
  back to HBM                ↓ (next layer reads this from HBM)
  
  Weight matrices  ──────►  Load W_O, W_FF1, W_FF2
  (FFN weights)              ↓
                             FFN computation
                             ↓
  Result ◄───────────────── Write result back to HBM
```

**The key pattern:** Every intermediate result goes back to HBM. The next operation reads it back from HBM. HBM is the shared staging area for all data between operations.

### LLM inference phases

**Prefill phase:**
- Input: the entire prompt (e.g., 1000 tokens at once)
- All tokens processed in parallel
- High arithmetic intensity — closer to compute-bound
- Produces the initial KV cache entries for all prompt tokens

**Decode phase:**
- Generates one new token per forward pass
- Only the new token passes through attention (cached tokens are in KV cache)
- Low arithmetic intensity — heavily memory-bound
- Must load full weight matrices each step to generate one token

```
Prefill:   [T1 T2 T3 T4 T5 T6 T7] → processed in parallel → KV cache filled
Decode:    [T8] → uses KV cache → output T9
           [T9] → uses KV cache → output T10
           [T10] → ... (one step at a time, very slow on memory-bound hardware)
```

### The GPU's bottleneck during decode

```
Each decode step:
  Load ~140 GB of weights from HBM (for 70B model in FP16)
  Do ~140 GFLOPs of math
  
  At 3,350 GB/s: loading takes ~42 ms
  At 989 TFLOP/s: math takes ~0.14 ms
  
  Chip is idle doing math 99.7% of the time
  → The math units are waiting for HBM to feed them
```

This is exactly what Tenstorrent is trying to solve differently.

---

# Phase 2 — Tenstorrent Silicon Architecture

## 2.1 The Tensix Core

The Tensix core is the basic building block of every Tenstorrent chip. Every compute operation and every data movement happens inside or between Tensix cores.

### Internal structure

```
One Tensix Core (1.5 MB SRAM)
┌──────────────────────────────────────────────────────────────────┐
│                                                                    │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                     SRAM (1.5 MB)                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │  │
│  │  │ CB (in)  │ │ CB (out) │ │  Param   │ │  Scratch /   │  │  │
│  │  │ circular │ │ circular │ │  buffer  │ │  temp data   │  │  │
│  │  │  buffer  │ │  buffer  │ │          │ │              │  │  │
│  │  └────┬─────┘ └────┬─────┘ └──────────┘ └──────────────┘  │  │
│  └───────┼─────────────┼─────────────────────────────────────┘  │
│          │             │                                          │
│  ┌───────▼─────────────▼──────────────────────────────────────┐  │
│  │                 Data Movement Pipeline                       │  │
│  │                                                              │  │
│  │   ┌──────────┐         ┌──────────┐         ┌──────────┐   │  │
│  │   │ Unpacker │         │  Matrix  │         │  Packer  │   │  │
│  │   │ (reads   │────────►│  Engine  │────────►│ (writes  │   │  │
│  │   │ from CB) │         │  (MXU)   │         │ to CB)   │   │  │
│  │   └──────────┘         └──────────┘         └──────────┘   │  │
│  │                              │                               │  │
│  │                        ┌─────▼──────┐                       │  │
│  │                        │   Vector   │                       │  │
│  │                        │   Unit     │                       │  │
│  │                        │  (SFPU)    │                       │  │
│  │                        └────────────┘                       │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              5 Baby RISC-V Processors                        │  │
│  │                                                              │  │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌──────┐ │  │
│  │  │ BRISC   │ │ NCRISC  │ │ TRISC0  │ │ TRISC1  │ │TRISC2│ │  │
│  │  │ (data   │ │ (data   │ │(compute)│ │(compute)│ │(comp)│ │  │
│  │  │  move   │ │  move   │ │         │ │         │ │      │ │  │
│  │  │  NoC→CB)│ │  CB→NoC)│ │  math   │ │  math   │ │ math │ │  │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘ └──────┘ │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                    │
│  ┌───────────────────────────────────────────────────────────┐    │
│  │              NoC Interface (to/from other cores)          │    │
│  └───────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────┘
```

### Matrix Engine (MXU)

- Performs dense matrix multiply-accumulate (MAC) operations.
- Works on **tiles**: the smallest unit of data it processes is a **32×32 tile** of numbers.
- Supported formats: BFP8, FP16, BFP4 (covered in section 6.1).
- This is where the actual math (dot products in attention, FFN weight multiplications) happens.
- The Unpacker reads tiles from SRAM circular buffers and feeds them to the Matrix Engine. The Packer writes results back to SRAM circular buffers.

### Vector Unit (SFPU — Special Function Processing Unit)

- Handles element-wise operations that the Matrix Engine cannot do.
- Operations: ReLU, GELU, SiLU, softmax, exp, sqrt, reciprocal, LayerNorm, etc.
- Works on vectors (one row of a tile at a time).
- Runs in parallel with the Matrix Engine when the operations allow it.

### The 5 Baby RISC-V Processors

These are small, in-order RISC-V cores. They do not do math — they only issue instructions to move data and control the compute units.

| Name | Role |
|---|---|
| **BRISC** (Brisc) | Reads data from GDDR6 or NoC, writes it into input circular buffers in SRAM |
| **NCRISC** (Ncrisc) | Reads results from output circular buffers and sends them out over the NoC |
| **TRISC0** | Controls the Unpacker — sequences tiles from SRAM to Matrix Engine |
| **TRISC1** | Controls the Math Engine — configures and triggers computation |
| **TRISC2** | Controls the Packer — sequences results from Matrix Engine back to SRAM |

All 5 run **simultaneously** and in a pipelined fashion:
- BRISC fills input CBs while TRISC0 drains them into the engine.
- TRISC2 fills output CBs while NCRISC drains them to the NoC.
- No core waits idle if the data is available.

### Circular Buffers (CB)

Circular buffers are the staging areas inside SRAM. They are like fixed-size ring buffers.

```
Circular Buffer operation:

  Producer (BRISC writes in)
         ↓
  [ slot0 | slot1 | slot2 | slot3 | slot4 | slot5 | slot6 | slot7 ]
                              ↑
                   Consumer (TRISC0 reads out)
                   
  When consumer catches up to producer → consumer stalls and waits
  When producer catches up to consumer → producer stalls and waits
  This is automatic synchronisation with no explicit locks
```

The size of each circular buffer and the number of tiles it holds is set by the compiler before execution. This is one place where the compiler's static knowledge of the model graph is critical.

---

## 2.2 Network-on-Chip (NoC) — the 2D Mesh

### What the NoC is

The NoC is an on-chip interconnect that connects every Tensix core to every other Tensix core. Instead of a shared bus (where only one core can send at a time), the NoC is a network of routers, one per core, that can all send simultaneously.

### Topology: 2D Torus

Tenstorrent uses a **2D Torus** topology. Imagine a grid where the edges wrap around.

```
2D Torus (simplified 4×4 example, Blackhole is 10×12):

  C00 ── C01 ── C02 ── C03
   │      │      │      │    (vertical links)
  C10 ── C11 ── C12 ── C13
   │      │      │      │
  C20 ── C21 ── C22 ── C23
   │      │      │      │
  C30 ── C31 ── C32 ── C33
   
  Plus wrap-around edges:
  C00 connects to C03 (left-right wrap)
  C00 connects to C30 (top-bottom wrap)
```

The wrap-around matters because it means any core can reach any other core in at most N/2 hops in either direction, reducing maximum hop count.

### Wormhole routing

Data on the NoC is broken into **flits** (flow control units — small fixed-size chunks). Wormhole routing means:
- The **header flit** travels through the network first and reserves a path.
- The remaining **body flits** follow the same path immediately.
- The path is held open (no re-routing mid-packet).
- This reduces per-flit routing overhead — only the header needs to be routed.

```
Wormhole packet in transit:

  Source ──► Router_A ──► Router_B ──► Router_C ──► Destination

  [HEADER flit] → chooses path, reserves output port at each router
  [BODY flit 1] → follows immediately
  [BODY flit 2] → follows
  [BODY flit 3] → follows
  [TAIL flit]   → releases reserved path
```

### Two NoC planes

Tenstorrent chips have **two independent NoC planes** (NoC0 and NoC1):
- NoC0 routes primarily in one direction pattern.
- NoC1 routes in the other direction pattern.
- Having two planes doubles effective NoC bandwidth and avoids deadlock conditions that can arise with bidirectional traffic on a single network.

### NoC bandwidth numbers (Wormhole)

- Each link (between adjacent cores): ~32 bytes/cycle
- At 1 GHz: ~32 GB/s per link
- Each core has 4 links (north, south, east, west) × 2 planes = 8 links

### What actually travels on the NoC

```
Data that uses the NoC:
- Weight tiles: GDDR6 controller core → destination Tensix core
- Activation tiles: Tensix core → next Tensix core (spatial pipeline)
- Partial sums: across cores during tensor-parallel matrix multiply
- Ethernet bridge: last core on one edge → Ethernet PHY → another chip
```

---

## 2.3 Blackhole Chip — Full Die Floorplan

### What is on the Blackhole die

Blackhole (released 2024) is Tenstorrent's current flagship AI accelerator chip.

```
Blackhole Die Floorplan (approximate, not to scale)

 ┌──────────────────────────────────────────────────────────────┐
 │                    Blackhole Die                              │
 │                                                               │
 │  [ETH] [ETH] [ETH] [ETH] [ETH] [ETH] [ETH] [ETH]  ← top    │
 │                                                               │
 │  ┌──────────────────────────────────────────────────────┐    │
 │  │         10 × 12 grid = 120 Tensix Cores              │    │
 │  │                                                        │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  │  [TX][TX][TX][TX][TX][TX][TX][TX][TX][TX]             │    │
 │  └──────────────────────────────────────────────────────┘    │
 │                                                               │
 │  [GDDR6 Ctrl] [GDDR6 Ctrl] [GDDR6 Ctrl] [GDDR6 Ctrl]        │
 │                                                               │
 │  [Big RISC-V ×16]    [PCIe Gen5]    [ARC (mgmt)]             │
 │                                                               │
 │  [ETH] [ETH] [ETH] [ETH] [ETH] [ETH] [ETH] [ETH]  ← bottom │
 └──────────────────────────────────────────────────────────────┘
```

Key components:

| Component | Count | Purpose |
|---|---|---|
| Tensix Cores | 120 | All AI math and data movement |
| Baby RISC-V per Tensix | 5 | Data movement and compute control |
| Big RISC-V cores | 16 | OS-level orchestration (host CPU bypass) |
| GDDR6 controllers | 8 | Interface to off-chip GDDR6 memory |
| Ethernet PHY blocks | 14 | 400 GbE for chip-to-chip scale-out |
| PCIe Gen5 | 1 | Host CPU connection |
| ARC management core | 1 | Power management, boot, housekeeping |

### The Big RISC-V cores (Blackhole-specific)

These are full-sized RISC-V cores (not the tiny baby RISC-V inside each Tensix). Their purpose is to:
- Run the dispatch software that sends kernels to the Tensix cores.
- Handle OS-level memory management (allocating GDDR6 buffers, etc.).
- Process PCIe transactions and Ethernet framing.
- **Bypass the host CPU** — on previous chips (Wormhole), the host CPU was involved in dispatching work. On Blackhole, the big RISC-V cores handle this autonomously.

This matters for latency: when you send a large LLM batch, the host CPU submits a single job. After that, the big RISC-V cores run the full execution without the host needing to be involved.

### GDDR6 configuration on Tenstorrent cards

| Card | GDDR6 Capacity | GDDR6 Bandwidth |
|---|---|---|
| Wormhole n150 | 12 GB | 288 GB/s |
| Wormhole n300 | 24 GB | 576 GB/s |
| Blackhole p100 | 16 GB | 448 GB/s |
| Blackhole p150 | 32 GB | 576 GB/s |

---

## 2.4 Distributed SRAM vs Central HBM

### The numbers side by side

```
NVIDIA H100 (HBM approach):
  ┌──────────────────────────────────────────────────────────┐
  │           132 SMs                                         │
  │  Each SM has 256 KB L1 = 132 × 256 KB = 33 MB L1 total  │
  │  Plus 50 MB shared L2 cache                              │
  │  Total on-chip SRAM: ~83 MB                              │
  └──────────────────────────────────────────────────────────┘
  Plus: 80 GB HBM3 at 3,350 GB/s

Tenstorrent Blackhole (distributed SRAM approach):
  ┌──────────────────────────────────────────────────────────┐
  │           120 Tensix cores                                │
  │  Each core has 1.5 MB dedicated SRAM                     │
  │  Total on-chip SRAM: 120 × 1.5 MB = 180 MB              │
  └──────────────────────────────────────────────────────────┘
  Plus: 32 GB GDDR6 at 576 GB/s
```

Blackhole has 2× more on-chip SRAM than H100. But it is distributed — each 1.5 MB chunk lives physically next to one Tensix core and is only directly accessible by that core.

### Why distributed SRAM is faster

```
H100 data path for one SM to compute:
  HBM → memory controller → crossbar → L2 cache → SM
  Latency: ~300–500 cycles, bandwidth shared with all 132 SMs

Blackhole data path for one Tensix to compute:
  GDDR6 → NoC → this core's SRAM → Matrix Engine
  SRAM is physically adjacent, ~1-5 cycles for the math unit to read it
  Or: previous core's SRAM → NoC → this core's SRAM (no GDDR6 at all)
```

### When GDDR6 is accessed vs not

```
Model weights:
  First inference: GDDR6 → NoC → Tensix SRAM  (loaded from GDDR6)
  Subsequent inferences: weights can stay in SRAM if they fit,
                         or be re-fetched from GDDR6

Activations (intermediate results):
  Tensix Core A computes → pushes result directly over NoC → Tensix Core B
  Activation never goes to GDDR6
  This is the "secret weapon" — GDDR6 bandwidth not consumed by activations

KV Cache:
  Stored in GDDR6 (too large for SRAM)
  Fetched per attention layer, per decode step
```

### Bandwidth the workload actually needs

During LLM decode (one token step):
- Tenstorrent's design: GDDR6 is only needed for weight tiles and KV cache. Activations flow over NoC (free, doesn't use GDDR6 bandwidth).
- NVIDIA's design: GDDR6/HBM is used for weights, KV cache, and all intermediate activations between operations.

Tenstorrent argues their GDDR6 bandwidth (~576 GB/s) is sufficient because they load far less through it.

---

## 2.5 Chip Generations — Grayskull, Wormhole, Blackhole

### Grayskull (2021)

First commercial Tenstorrent chip. Sold as e75 and e150 cards.

- 120 Tensix cores (same count as later chips but at lower performance)
- 120 MB of SRAM (1 MB per Tensix)
- No on-chip Ethernet (scale-out via PCIe)
- LPDDR4 memory interface
- TDP: 75W (e75) / 150W (e150)

Grayskull is discontinued but its software model (Metalium API) is the direct ancestor of what runs on Wormhole and Blackhole.

### Wormhole (2023)

The product that shipped to most early customers. Sold as n150 and n300 cards.

```
Wormhole key improvements over Grayskull:
- GDDR6 (vs LPDDR4) — higher bandwidth
- On-chip Ethernet PHY ×16 ports at 100 Gbps each
- 8×8 NoC grid (reduced from Grayskull's configuration)
- SRAM per Tensix: 1.5 MB
- 2× Wormhole chips on n300 (connected via Ethernet on-board)
- First chip where the compiler handles multi-chip as one grid
```

The `tt-metal` repository on GitHub has working code examples primarily targeting Wormhole. Most of the documentation at docs.tenstorrent.com is written with Wormhole as the reference chip.

### Blackhole (2024)

Current flagship.

```
Blackhole key improvements over Wormhole:
- 120 Tensix cores in 10×12 grid (vs 8×8)
- 16 big RISC-V cores for host-bypass dispatch
- Ethernet: 14 ports × 400 Gbps (vs 16 × 100 Gbps on Wormhole)
- PCIe Gen5 (vs Gen4 on Wormhole)
- GDDR6 up to 32 GB
- ARC management core improved
- FP8 hardware support added to Matrix Engine
```

### Reading the GitHub repos by chip

| Repository | What it contains | Which chips |
|---|---|---|
| `tt-metal` | Full hardware SDK (Metalium) + TTNN | Grayskull, Wormhole, Blackhole |
| `tt-metal/ttnn` | Tensor operation library | All chips |
| `tt-forge` | ML compiler (replaces TT-Buda) | Wormhole, Blackhole |
| `tt-llk` | Lowest-level kernel math ops | All chips (chip-specific subdirs) |
| `tt-lang` | New high-level programming language | Blackhole and forward |

In the `tt-metal` repo, chip-specific code lives under:
- `tt_metal/hw/firmware/src/` — RISC-V firmware for Tensix cores
- `tt_metal/hw/ckernels/` — compute kernels per chip variant
- `tt_metal/impl/dispatch/` — command dispatch logic

---

# Phase 3 — Dataflow Execution Model

## 3.1 Von Neumann vs Dataflow Architectures

### Von Neumann (what CPUs and GPUs use)

Von Neumann architecture has:
- A single shared memory space
- A processor that fetches instructions from memory, decodes them, executes them
- A program counter that advances sequentially

```
Von Neumann execution:

  Memory (instructions + data mixed together)
       │
  ┌────▼────┐
  │  Fetch  │ ← reads instruction at program counter address
  └────┬────┘
  ┌────▼────┐
  │ Decode  │ ← figures out what the instruction means
  └────┬────┘
  ┌────▼────┐
  │ Execute │ ← does the operation (possibly reading/writing memory)
  └────┬────┘
  ┌────▼────┐
  │Writeback│ ← stores result back to memory or register
  └─────────┘
  (repeat, incrementing program counter each time)
```

The bottleneck: the processor must read from and write to the shared memory for nearly every operation. Memory bandwidth is the limiting factor.

GPU warps use this model — they add parallelism (many SMs running many warps), but each thread still follows the fetch-decode-execute pattern with shared HBM as the backing store.

### Pure dataflow (reference concept)

In a pure dataflow architecture:
- There is no program counter.
- A computation node **fires** (executes) as soon as all its inputs are available.
- There is no concept of sequential instruction order.

```
Dataflow graph example (addition then multiply):

  [A] ──┐
        ├──► [ADD] ──► [C = A+B]
  [B] ──┘
                            │
  [D] ──────────────► [MUL] ──► [E = C*D]
  
  ADD fires as soon as A and B are ready.
  MUL fires as soon as C and D are ready (no waiting for anything else).
```

Pure dataflow chips were researched in the 1970s–80s but were difficult to build practically.

### Tenstorrent's position: explicit dataflow, spatial execution

Tenstorrent is not pure dataflow. It uses **explicit dataflow with static scheduling**:
- A compiler analyses the full neural network graph.
- The compiler decides, at compile time, which Tensix core runs which part of the model.
- The compiler decides, at compile time, the exact path each tensor takes across the NoC.
- The RISC-V cores execute these pre-decided movement instructions explicitly (not hardware-inferred).
- During runtime, no scheduling decisions are made. Everything is pre-planned.

```
Tenstorrent execution model:

  Compile time (once):
  PyTorch graph → TT-Forge compiler → physical schedule
  "Core 0 does layer 1, Core 1 does layer 2, data moves from 0→1 via NoC at path X"
  
  Runtime (every inference):
  Each RISC-V in each core executes its pre-assigned kernel
  No dynamic decisions, no cache guessing, no warp scheduling
  Fully deterministic
```

---

## 3.2 Spatial Pipelining — the Core Technique

### What spatial pipelining means

In a regular pipeline (time-multiplexed), one processor runs stage 1, then stage 2, then stage 3, one after another on the same piece of hardware.

In a **spatial** pipeline, different stages are physically mapped to different processor cores. Stage 1 always runs on Core A. Stage 2 always runs on Core B. Data physically moves from Core A to Core B.

```
Temporal pipeline (GPU-like):
  
  Time:  t1    t2    t3    t4    t5    t6
  Core:  [L1] [L2] [L3] [L1] [L2] [L3]   ← same hardware, sequential layers
  
  One token processed, then the next token starts from the beginning.
  HBM accessed between every layer (write result, read for next layer).

Spatial pipeline (Tenstorrent):
  
  Core A: [L1] [L1] [L1] [L1] ...  ← always runs Layer 1
  Core B: [L2] [L2] [L2] [L2] ...  ← always runs Layer 2
  Core C: [L3] [L3] [L3] [L3] ...  ← always runs Layer 3
  
  Token1: A→B→C (flows through physically)
  Token2: A→B→C (starts when Token1 leaves Layer 1)
  Token3: A→B→C (starts when Token2 leaves Layer 1)
  
  At steady state: Core A processes Token 3, Core B processes Token 2,
                   Core C processes Token 1 — all simultaneously.
```

### How activations flow between cores

```
Transformer layer execution across 3 Tensix cores:

GDDR6          Core 0             Core 1             Core 2
 │          ┌──────────┐       ┌──────────┐       ┌──────────┐
 │  W_QKV   │  SRAM    │       │  SRAM    │       │  SRAM    │
 ├─────────►│  (1.5MB) │       │  (1.5MB) │       │  (1.5MB) │
 │          │          │       │          │       │          │
 │  tokens  │ [Input]  │       │[Attn out]│       │ [FFN out]│
 ├─────────►│    ↓     │  NoC  │    ↓     │  NoC  │    ↓     │
 │          │ QKV proj │──────►│ Attention│──────►│   FFN    │
 │          │    ↓     │       │    ↓     │       │    ↓     │
 │  W_O     │ [Q,K,V]  │       │[Weighted │       │[Layer    │
 ├─────────►│          │       │  values] │       │  output] │
 │          └──────────┘       └──────────┘       └──────────┘
 │                                                      │
 │                                                      ▼ NoC
 │                                               (next layer / output)
```

The activation tensors (`[Q,K,V]`, `[Attn out]`) travel over the NoC directly into the next core's SRAM. They do not go to GDDR6.

### Circular buffers as the synchronisation mechanism

```
Between Core A (producer) and Core B (consumer):

Core A SRAM:          Core B SRAM:
┌──────────────────┐  ┌──────────────────┐
│ output CB:       │  │ input CB:        │
│ [tile0][tile1][] │  │ [tile0][tile1][] │
│     ↑            │  │            ↑    │
│  Packer writes   │  │  Unpacker reads │
└──────────────────┘  └──────────────────┘
           │ NoC transfer (NCRISC pushes, BRISC receives)
           └────────────────────────────────────►

Synchronisation:
  Core A's NCRISC: "I have a tile ready" → pushes over NoC
  Core B's BRISC: receives tile → writes to input CB → signals TRISC0
  If Core B's CB is full: Core A's NCRISC stalls and waits
  This is backpressure — the pipeline automatically throttles
```

---

## 3.3 Full LLM Forward Pass — Step by Step

This traces a single decode step of LLaMA-2 7B on a Wormhole n300 (a realistic example).

### Step 0: Model loading (one-time, at startup)

```
NVMe SSD (model files: .safetensors, ~13 GB)
    │
    │ PCIe 4.0 (read via host CPU, ~7 GB/s)
    ▼
Host CPU RAM (DDR5, ~32 GB)
    │
    │ PCIe 4.0 (DMA transfer to card, ~16 GB/s)
    ▼
GDDR6 on Wormhole n300 (24 GB)
    │
    │ (weights now in GDDR6, ready for inference)
    ▼
KV cache also allocated here (grows as sequence grows)
```

This happens once. Subsequent inferences read weights from GDDR6 (or from SRAM if they fit and have been pre-loaded).

### Step 1: Host dispatches a decode step

```
Host CPU:
  user_input = "What is the capital of France?"
  output = model.forward(token_ids, kv_cache)
  
  ↓ (via PCIe)
  
Blackhole big RISC-V cores:
  Receive job descriptor: "run decode step, token=X, kv_cache_ptr=Y"
  Look up pre-compiled kernel schedule from TT-Forge
  Dispatch kernel commands to all assigned Tensix cores simultaneously
```

### Step 2: Weight tiles fetched from GDDR6 into Tensix SRAM

```
For the first transformer layer:

GDDR6 (W_Q, W_K, W_V matrices live here)
    │
    │ NoC read request issued by BRISC of Core #0
    ▼
GDDR6 memory controller (attached core on the die edge)
    │
    │ NoC packet: weight tiles → Core #0
    ▼
Core #0 SRAM input circular buffer
    │
    │ TRISC0 signals: "tile available"
    ▼
Unpacker → Matrix Engine
```

This happens simultaneously for all transformer layers that have cores assigned to them (the compiler distributes layers across the core grid).

### Step 3: QKV projection computed

```
Core #0 (assigned to QKV projection of Layer 1):

  SRAM:
    [Input embedding tile] ← fetched from GDDR6 (the current token's embedding)
    [W_Q tile]             ← fetched from GDDR6
    [W_K tile]             ← fetched from GDDR6
    [W_V tile]             ← fetched from GDDR6
    
  Matrix Engine:
    embedding × W_Q → Q tile (32×32 result)
    embedding × W_K → K tile
    embedding × W_V → V tile
    
  Packer writes Q, K, V tiles to output circular buffer in SRAM
  NCRISC sends Q, K, V tiles over NoC to Core #1
  (Does NOT write back to GDDR6)
```

### Step 4: KV cache update and attention

```
Core #1 (assigned to attention of Layer 1):

  Receives Q, K, V for current token over NoC → SRAM
  
  From GDDR6 (this IS a GDDR6 access):
    K_cache for all previous tokens → SRAM  (can be many GB for long context)
    V_cache for all previous tokens → SRAM
    
  Appends current K, V to cache in GDDR6
  
  Matrix Engine:
    Q × K_cache^T → attention scores (scaled)
    
  Vector Unit (SFPU):
    softmax(attention scores) → attention weights
    
  Matrix Engine:
    attention weights × V_cache → context vector
    
  Result (context vector tiles) → output circular buffer → NoC → Core #2
```

### Step 5: FFN and remaining layers

```
Core #2 (FFN of Layer 1):
  Receives context vector over NoC
  Fetches W_FF1, W_FF2 from GDDR6
  
  Matrix Engine: context × W_FF1 → intermediate
  Vector Unit: SiLU(intermediate) → activated
  Matrix Engine: activated × W_FF2 → FFN output
  
  Result → NoC → Core #3 (Layer 2 QKV)
  (and so on through all 32 layers of LLaMA-2 7B)
```

### Step 6: Final output

```
Last layer → logits computation → softmax over vocabulary
→ top-k/top-p sampling → next token ID
→ sent via NoC → big RISC-V → PCIe → host CPU
→ host decodes token ID to text string
→ "Paris"
```

### Data that uses GDDR6 bandwidth vs what does not

```
Uses GDDR6 bandwidth:            Does NOT use GDDR6:
─────────────────────            ──────────────────────────
Weight matrices (every step)     Layer-to-layer activations
KV cache (growing per token)     Partial sums within a layer
Initial embedding lookup         Intermediate attention scores
Output logits storage            Most FFN intermediate values
```

This is why 576 GB/s GDDR6 can be sufficient: the activations — which in a GPU must also go through HBM — bypass GDDR6 entirely on Tenstorrent.

---

## 3.4 Explicit Memory Management — No Hardware Prefetcher

### What a hardware prefetcher does (CPU/GPU)

In CPUs and GPUs, a hardware prefetcher watches the pattern of memory accesses:

```
Hardware prefetcher:
  Access: address 0x1000
  Access: address 0x1040 (stride +64 bytes)
  Access: address 0x1080 (stride +64 bytes)
  
  Prefetcher detects stride pattern
  Automatically issues pre-read for 0x10C0 before it is requested
  Data arrives in L1 cache before the CPU asks for it
  
  If the prefetch was wrong: wasted bandwidth, no harm
  If the prefetch was right: zero latency access
```

The problem: the prefetcher is guessing. For irregular access patterns (sparse attention, dynamic sequence lengths), it guesses wrong and wastes bandwidth.

### Tenstorrent's approach: explicit DMA-style reads

The baby RISC-V (BRISC) inside each Tensix executes a kernel that says exactly:

```c
// Pseudocode of what a BRISC kernel does:
void brisc_main() {
    // Compiler has told us: weights are at GDDR6 address 0x8000_0000,
    // size 1024 tiles (32KB), destination is CB0 (circular buffer 0)
    
    while (processing) {
        // Wait for space in the circular buffer
        cb_wait_for_space(CB0, 1);
        
        // Issue explicit read: source address, destination CB, size
        noc_async_read(
            src_addr = GDDR6_BASE + weight_offset,
            dst_local_l1_addr = get_cb_write_ptr(CB0),
            size = TILE_SIZE
        );
        
        // Advance the circular buffer write pointer
        cb_push_back(CB0, 1);
        
        weight_offset += TILE_SIZE;
    }
}
```

Source reference: `tt_metal/hw/firmware/src/brisc.cc` and kernel examples in `tt_metal/kernels/dataflow/`

### Why this is better for AI workloads

```
Hardware prefetcher on GPU:              Explicit RISC-V on Tenstorrent:
─────────────────────────────────────    ─────────────────────────────────────
Guesses access pattern at runtime        Compiler knows exact access pattern
                                         at compile time (model is static)
                                         
Wastes bandwidth on wrong guesses        No wasted bandwidth — every load
                                         is precisely what is needed
                                         
Adds logic to detect patterns            No detection needed
(die area, power)                        
                                         
Latency hidden by having many            Latency hidden by pipelining:
in-flight warps (needs high             BRISC issues many async reads ahead
occupancy)                               of time, NoC handles them in flight

Cannot be programmed by user             Kernel developer has full control
```

### The trade-off: predictability required

The explicit model works because transformer models are **static graphs** — the same sequence of operations runs every time, with the same access patterns (for fixed sequence lengths). For workloads with truly unpredictable access patterns (e.g., graph neural networks with variable-length node neighbours), explicit prefetching is harder to write.

---

# Phase 4 — Compiler and Software Stack

## 4.1 Compiler Fundamentals (prerequisite)

### What a compiler does for AI (in general terms)

A compiler takes a high-level description of computation (a neural network in PyTorch) and converts it into instructions that hardware can execute. Between the high-level description and the hardware instructions, there are several **intermediate representations (IRs)**.

```
Compiler pipeline (general):

  Python/PyTorch model definition
          │
          ▼  (frontend: parse, type-check, trace)
  High-level IR (computation graph)
  "matmul(A, B) → add(result, C) → relu(result)"
          │
          ▼  (optimisation passes)
  Optimised IR (fused ops, constants folded, dead code removed)
  "fused_matmul_add_relu(A, B, C)"
          │
          ▼  (lowering passes)
  Low-level IR (hardware-specific)
  "load tile from addr X; feed to MXU; activate with SFPU"
          │
          ▼  (code generation)
  Executable kernels (RISC-V machine code + NoC routing tables)
```

### MLIR — the framework TT-Forge is built on

MLIR (Multi-Level Intermediate Representation) is a compiler infrastructure from Google/LLVM. It allows defining **dialects** — domain-specific extensions to the IR — and writing transformation passes that operate on them.

Key MLIR concepts:

**Operations (ops):**
```
// An MLIR operation looks like:
%result = "dialect.opname"(%input1, %input2) { attributes } : (type, type) -> type

// Example:
%y = "tosa.matmul"(%A, %B) {quantization_info = ...} : (tensor<32x64xf16>, tensor<64x128xf16>) -> tensor<32x128xf16>
```

**Dialects:** A set of operations that belong to a specific level of abstraction. For example:
- `tosa` dialect: portable operations (industry standard)
- `linalg` dialect: linear algebra ops
- `ttir` dialect: Tenstorrent's IR (custom)
- `ttnn` dialect: Tenstorrent's hardware-level ops

**Lowering passes:** A pass takes a graph in dialect A and converts it to dialect B. The full compiler is a series of lowering passes from abstract to concrete.

**SSA form (Static Single Assignment):**
Every variable is assigned exactly once. Instead of:
```
x = 5
x = x + 3  // x assigned twice
```
SSA writes:
```
x1 = 5
x2 = x1 + 3  // new name for each assignment
```
This makes dataflow analysis simpler for the compiler.

### Graph rewriting and op fusion

Before lowering, the compiler rewrites the graph to reduce the number of operations:

```
Before fusion:
  matmul(A, B) → C
  add(C, bias)  → D
  relu(D)       → E
  
  Three separate operations, each needing to write/read SRAM.

After fusion:
  matmul_add_relu(A, B, bias) → E
  
  One kernel, result only written once to SRAM.
```

Op fusion is critical on Tenstorrent because it reduces the number of NoC transfers between cores and reduces the number of SRAM write-reads.

---

## 4.2 TT-Forge — the ML Compiler

### What TT-Forge is

TT-Forge is Tenstorrent's MLIR-based compiler that takes a user's neural network (in PyTorch or JAX) and compiles it into the executable form that runs on Tensix cores.

Source: [github.com/tenstorrent/tt-forge](https://github.com/tenstorrent/tt-forge)

TT-Forge replaced the older **TT-Buda** compiler in 2024. TT-Buda still exists in the `pybuda` package but is in maintenance mode.

### Full compilation pipeline

```
User code (Python):
  import torch
  model = LlamaModel.from_pretrained("meta-llama/Llama-2-7b")
  compiled_model = tt_forge.compile(model, example_inputs)
  output = compiled_model(input_tokens)
  
The compilation steps:

Step 1: Frontend capture
  PyTorch model
  └──► torch.compile() backend hook (tt-forge registers as a backend)
  └──► or: TT-XLA (for JAX models) via XLA HLO graph
  └──► Result: StableHLO or TOSA dialect graph
  
Step 2: TTIR conversion
  TOSA/StableHLO ops
  └──► tt-forge lowering pass
  └──► TTIR dialect (Tenstorrent Intermediate Representation)
  
  TTIR represents operations at the tensor level, format-agnostic.
  Example TTIR op: ttir.matmul, ttir.add, ttir.softmax
  
Step 3: Optimisation on TTIR
  - Op fusion (matmul + add + relu → one op)
  - Constant folding (compute fixed values at compile time)
  - Dead code elimination
  - Layout propagation (decide row-major vs tile layout per tensor)
  
Step 4: TTNN lowering
  TTIR ops
  └──► TTNN dialect ops
  
  TTNN is hardware-aware. A ttnn.matmul op knows:
  - Which Tensix core(s) execute it
  - What tile size to use (32×32)
  - What data format (BFP8, FP16, etc.)
  - What SRAM circular buffers to use
  
Step 5: Core placement and scheduling
  - Compiler assigns each TTNN op to one or more physical Tensix cores
  - Compiler decides NoC routes for all inter-core data transfers
  - Compiler computes circular buffer sizes (enough to avoid stalls)
  - Output: a physical execution schedule
  
Step 6: Code generation
  - Compiler generates RISC-V C++ kernel source for each core
  - Kernels compiled to RISC-V machine code
  - NoC routing tables generated
  - Everything packaged as a binary for the TT-Metalium runtime
```

### How to use TT-Forge (basic example)

```python
# Source: tt-forge GitHub examples
import torch
import tt_forge

# Load a standard PyTorch model
model = torch.nn.Linear(128, 64)
model.eval()

# Compile for Tenstorrent
compiled_model = tt_forge.compile(
    model,
    sample_inputs=[torch.randn(1, 128)],
    device="tt"   # target: Tenstorrent device
)

# Run inference
with torch.no_grad():
    output = compiled_model(torch.randn(1, 128))
```

Behind the scenes: `tt_forge.compile()` runs all six steps above and returns a callable that, when invoked, dispatches the pre-compiled kernels to the Tensix cores.

### Key files in the tt-forge repository

```
tt-forge/
├── forge/                         # Main compiler source
│   ├── csrc/                      # C++ compiler core
│   │   ├── tt_torch_device/       # torch.compile backend registration
│   │   └── passes/                # MLIR transformation passes
│   ├── python_lib/                # Python frontend API
│   └── test/                      # Test cases (good for learning)
├── forge/csrc/passes/
│   ├── lower_to_ttir.cpp          # TOSA → TTIR lowering
│   ├── lower_to_ttnn.cpp          # TTIR → TTNN lowering
│   └── fuse_ops.cpp               # Op fusion pass
```

---

## 4.3 TTNN — the Tensor Operation Library

### What TTNN is

TTNN is the layer between the compiler's output and the actual hardware execution. It is both:
1. A Python/C++ API for writing hardware-aware tensor operations directly (for researchers and framework developers).
2. The target dialect in the TT-Forge compiler — compiled models become sequences of TTNN ops.

Source: [github.com/tenstorrent/tt-metal/tree/main/ttnn](https://github.com/tenstorrent/tt-metal/tree/main/ttnn)

### Tensor memory layout in TTNN

TTNN introduces a concept that does not exist in PyTorch: **explicit memory layout**.

```
Standard PyTorch tensor layout:
  A 4×4 matrix stored as 16 contiguous FP32 values in RAM.
  Row-major (C-style): [row0_col0, row0_col1, ..., row3_col3]
  
TTNN tensor layouts:

  ROW_MAJOR layout:
    Same as PyTorch. Used for irregular shapes, host ↔ device transfers.
  
  TILE layout (preferred for compute):
    Matrix is divided into 32×32 tiles.
    Each tile is stored contiguously in memory.
    [tile0_all_values | tile1_all_values | ...]
    
    Why tiles? The Matrix Engine processes 32×32 blocks at a time.
    Tile layout means a single contiguous read from SRAM gives the
    engine exactly one complete tile — no scatter/gather needed.
```

### Tensor memory location in TTNN

```python
# TTNN explicitly specifies where a tensor lives:

import ttnn

# Create tensor on device (in Tensix SRAM)
t = ttnn.from_torch(
    torch.randn(128, 128),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.L1_MEMORY_CONFIG  # L1 = Tensix SRAM
)

# Or: tensor stored in DRAM (GDDR6) and fetched on demand
t_dram = ttnn.from_torch(
    torch.randn(1024, 1024),
    dtype=ttnn.bfloat16,
    layout=ttnn.TILE_LAYOUT,
    device=device,
    memory_config=ttnn.DRAM_MEMORY_CONFIG  # DRAM = GDDR6
)
```

### Supported data types

| TTNN dtype | Bits per element | Description |
|---|---|---|
| `ttnn.float32` | 32 | Full precision (rarely used on-chip) |
| `ttnn.bfloat16` | 16 | Brain float (1 sign, 8 exp, 7 mantissa) |
| `ttnn.bfloat8_b` | 8 | Block float: shared exponent per 16 values |
| `ttnn.bfloat4_b` | 4 | Block float: 4-bit mantissa, most compressed |
| `ttnn.uint32` | 32 | Unsigned int (indices, etc.) |
| `ttnn.uint16` | 16 | Unsigned int |

BFP8 and BFP4 are Tenstorrent's **block floating point** formats. In a 32×32 tile, each row of 16 values shares a single exponent byte. This achieves near-FP16 accuracy at half the storage.

### Core TTNN operations

```python
# Matrix multiply
result = ttnn.matmul(tensor_a, tensor_b)

# Element-wise ops
result = ttnn.add(tensor_a, tensor_b)
result = ttnn.multiply(tensor_a, tensor_b)

# Activation functions
result = ttnn.softmax(tensor, dim=-1)
result = ttnn.gelu(tensor)
result = ttnn.silu(tensor)

# Attention (fused kernel)
result = ttnn.transformer.attention(q, k, v, attention_mask)

# Moving between memory locations
t_l1 = ttnn.to_memory_config(t_dram, ttnn.L1_MEMORY_CONFIG)
t_host = ttnn.to_torch(t_l1)  # back to PyTorch
```

### Sharded tensors — how large tensors fit in 1.5 MB SRAM

A weight matrix for LLaMA-2 7B has shape [4096, 4096] in FP16 = 32 MB. This does not fit in one core's 1.5 MB SRAM. TTNN handles this with **tensor sharding**:

```
W_Q matrix [4096, 4096] sharded across 32 Tensix cores:

  Core #0:  W_Q[0:128, :]     (128 × 4096 × 2 bytes = 1 MB)
  Core #1:  W_Q[128:256, :]   (1 MB)
  ...
  Core #31: W_Q[3968:4096, :] (1 MB)
  
Each core has its shard in SRAM and computes only its portion of the matmul.
Partial results combined over the NoC (allreduce).
```

TTNN shard types:
- `HEIGHT_SHARD`: split along rows
- `WIDTH_SHARD`: split along columns
- `BLOCK_SHARD`: split into 2D blocks

```python
# Create a height-sharded tensor
shard_spec = ttnn.ShardSpec(
    core_grid,
    shard_shape=[128, 4096],  # each core gets 128 rows
    orientation=ttnn.ShardOrientation.ROW_MAJOR
)
memory_config = ttnn.MemoryConfig(
    ttnn.TensorMemoryLayout.HEIGHT_INTERLEAVED,
    ttnn.BufferType.L1,
    shard_spec
)
```

---

## 4.4 TT-Metalium — Bare-Metal Programming SDK

### What TT-Metalium is

TT-Metalium is the lowest-level user-facing API. It lets you:
1. Allocate buffers in GDDR6 or Tensix SRAM directly.
2. Write C++ kernels that run on the baby RISC-V processors.
3. Specify exactly which cores run which kernels.
4. Define the circular buffers and their sizes.
5. Configure the NoC transfers.

Source: [github.com/tenstorrent/tt-metal](https://github.com/tenstorrent/tt-metal) (root level)

Everything above Metalium (TTNN, TT-Forge) compiles down to Metalium calls eventually.

### Three types of kernels

On each Tensix core, you write **three separate kernel files**, one for each group of RISC-V cores:

```
1. Compute kernel (compiled for TRISC0, TRISC1, TRISC2):
   Controls the Matrix Engine and SFPU.
   Written using Tenstorrent's compute API.
   
2. Data movement kernel — reader (compiled for BRISC):
   Reads data from GDDR6 or from another core over NoC.
   Writes data to input circular buffers in SRAM.
   
3. Data movement kernel — writer (compiled for NCRISC):
   Reads from output circular buffers in SRAM.
   Sends data to another core over NoC, or writes to GDDR6.
```

### A complete Metalium example: matrix multiply

This is simplified from the actual tt-metal examples at `tt_metal/programming_examples/matmul_single_core/`.

```cpp
// ── FILE 1: reader_kernel.cpp (runs on BRISC) ──────────────────────
// Reads weight matrix and input tensor from DRAM into CBs

#include "dataflow_api.h"

void kernel_main() {
    // Arguments passed from host at runtime
    uint32_t src0_addr  = get_arg_val<uint32_t>(0);  // input tensor in DRAM
    uint32_t src1_addr  = get_arg_val<uint32_t>(1);  // weight matrix in DRAM
    uint32_t num_tiles  = get_arg_val<uint32_t>(2);  // how many tiles to read
    
    // Set up DRAM address generators (handle tile striding automatically)
    InterleavedAddrGenFast<true> src0_gen = {
        .bank_base_address = src0_addr,
        .page_size = get_tile_size(cb_id_in0)  // 32×32 × 2 bytes = 2KB
    };
    
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Wait until circular buffer has space for one more tile
        cb_reserve_back(cb_id_in0, 1);
        
        // Get write pointer into the CB
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        
        // Issue async DRAM read
        noc_async_read_tile(i, src0_gen, l1_write_addr);
        
        // Wait for the read to complete
        noc_async_read_barrier();
        
        // Notify compute kernel that tile is ready
        cb_push_back(cb_id_in0, 1);
    }
}
```

```cpp
// ── FILE 2: compute_kernel.cpp (runs on TRISC0/1/2) ───────────────
// Performs matrix multiply tiles

#include "compute_kernel_api/matmul.h"

namespace NAMESPACE {
void MAIN {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);
    
    // Initialise the matrix multiply unit
    mm_init();
    
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Wait for input tiles to be ready in CBs
        cb_wait_front(cb_id_in0, 1);  // wait for 1 tile in CB0
        cb_wait_front(cb_id_in1, 1);  // wait for 1 tile in CB1
        
        // Reserve space in output CB
        cb_reserve_back(cb_id_out0, 1);
        
        // Acquire tiles from CBs for compute
        acquire_dst(tt::DstMode::Half);
        
        // Perform tile matrix multiply: CB0[0] × CB1[0] → dest register
        matmul_tiles(cb_id_in0, cb_id_in1, 0, 0, 0, false);
        
        // Pack result from dest register to output CB
        pack_tile(0, cb_id_out0);
        
        // Release dest register
        release_dst(tt::DstMode::Half);
        
        // Advance CB read pointers (consumed one tile from each input)
        cb_pop_front(cb_id_in0, 1);
        cb_pop_front(cb_id_in1, 1);
        
        // Advance CB write pointer (produced one tile in output)
        cb_push_back(cb_id_out0, 1);
    }
}
} // namespace NAMESPACE
```

```cpp
// ── FILE 3: writer_kernel.cpp (runs on NCRISC) ─────────────────────
// Reads output tiles from CB and writes to DRAM (or sends over NoC)

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr   = get_arg_val<uint32_t>(0);  // destination in DRAM
    uint32_t num_tiles  = get_arg_val<uint32_t>(1);
    
    InterleavedAddrGenFast<true> dst_gen = {
        .bank_base_address = dst_addr,
        .page_size = get_tile_size(cb_id_out0)
    };
    
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // Wait for compute kernel to produce a tile
        cb_wait_front(cb_id_out0, 1);
        
        uint32_t l1_read_addr = get_read_ptr(cb_id_out0);
        
        // Write tile to DRAM
        noc_async_write_tile(i, dst_gen, l1_read_addr);
        noc_async_write_barrier();
        
        cb_pop_front(cb_id_out0, 1);
    }
}
```

### Host-side setup code (Python with tt-metal)

```python
import ttnn
import tt_lib as ttl

device = ttnn.open_device(device_id=0)

# Allocate input and weight buffers in DRAM
input_tensor  = ttnn.from_torch(torch_input,  layout=ttnn.TILE_LAYOUT, device=device)
weight_tensor = ttnn.from_torch(torch_weight, layout=ttnn.TILE_LAYOUT, device=device)
output_tensor = ttnn.allocate_tensor_on_device(output_shape, ttnn.bfloat16, ttnn.TILE_LAYOUT, device)

# Define the program (which kernels run on which core)
program = ttl.CreateProgram()

core = ttl.CoreCoord(0, 0)  # Tensix core at grid position (0,0)

# Define circular buffers
cb_config_in0 = ttl.CircularBufferConfig(num_tiles=4, data_format=ttl.DataFormat.Float16_b)
cb_config_in1 = ttl.CircularBufferConfig(num_tiles=4, data_format=ttl.DataFormat.Float16_b)
cb_config_out = ttl.CircularBufferConfig(num_tiles=4, data_format=ttl.DataFormat.Float16_b)
cb_in0 = ttl.CreateCircularBuffer(program, core, cb_config_in0)
cb_in1 = ttl.CreateCircularBuffer(program, core, cb_config_in1)
cb_out = ttl.CreateCircularBuffer(program, core, cb_config_out)

# Compile and attach kernels
reader_kernel = ttl.CreateKernel(program, "reader_kernel.cpp", core, ttl.ReaderDataMovementConfig())
compute_kernel = ttl.CreateKernel(program, "compute_kernel.cpp", core, ttl.ComputeConfig())
writer_kernel  = ttl.CreateKernel(program, "writer_kernel.cpp", core, ttl.WriterDataMovementConfig())

# Set runtime arguments (addresses become known at runtime)
ttl.SetRuntimeArgs(program, reader_kernel, core, [input_tensor.buffer_address(), weight_tensor.buffer_address(), num_tiles])
ttl.SetRuntimeArgs(program, writer_kernel, core, [output_tensor.buffer_address(), num_tiles])

# Execute
ttl.LaunchProgram(device, program)
ttl.Synchronize(device)
```

This is the most direct path to the hardware — no compiler abstraction.

---

## 4.5 TT-LLK — Low-Level Kernels

### What TT-LLK is

TT-LLK (Low-Level Kernels) is the lowest software layer. It contains the math operation implementations that directly program the Matrix Engine and SFPU hardware.

Source: [github.com/tenstorrent/tt-llk](https://github.com/tenstorrent/tt-llk)

The kernels in tt-llk are not things you typically call directly. They are called by the TT-Metalium compute kernel API (`matmul_tiles`, `add_tiles`, etc.). But understanding them shows you exactly what the hardware does.

### What "Low-Level" means here

```
User's model (PyTorch)
    │
TT-Forge compiler (MLIR passes)
    │
TTNN tensor operations
    │
TT-Metalium compute kernel (matmul_tiles, pack_tile, etc.)
    │
TT-LLK                           ← HERE
    │
Matrix Engine hardware registers  ← actual silicon
```

### Repository structure

```
tt-llk/
├── common/
│   ├── inc/                     # Headers shared across all chips
│   │   ├── llk_math_matmul.h    # Matrix multiply LLK
│   │   ├── llk_math_eltwise_unary.h   # Element-wise ops (relu, gelu, etc.)
│   │   ├── llk_math_eltwise_binary.h  # Binary ops (add, mul)
│   │   ├── llk_pack.h           # Packer control
│   │   ├── llk_unpack_A.h       # Unpacker for A matrix
│   │   └── llk_unpack_AB.h      # Unpacker for A and B matrices
├── wormhole_b0/
│   └── inc/                     # Wormhole-specific implementations
└── blackhole/
    └── inc/                     # Blackhole-specific implementations
```

### What the Packer and Unpacker do

The **Unpacker** reads tiles from an SRAM circular buffer and formats them for the Matrix Engine's input registers.
The **Packer** takes the Matrix Engine's output from its destination registers and writes the result back to an SRAM circular buffer.

```
SRAM CB_in → Unpacker → Matrix Engine input regs
                              ↓
                        Math happens in MXU
                              ↓
              Packer ← Matrix Engine output (dest regs)
                ↓
             SRAM CB_out
```

LLK code controls:
- Tile format conversion (from storage format to compute format).
- Accumulate mode: should this result add to existing dest register values, or overwrite.
- The specific registers and ports on the Matrix Engine.

### SFPU (Special Function Processing Unit) ops

The SFPU is the vector unit inside each Tensix. LLK implementations for SFPU:

```cpp
// From tt-llk/common/inc/llk_math_eltwise_unary.h (simplified)

// Apply GELU activation to a tile currently in dest registers
template <EltwiseUnaryType unary_type>
inline void llk_math_eltwise_unary_sfpu_gelu(uint dst_index) {
    // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    // Implemented as a piecewise polynomial approximation in SFPU microcode
    sfpu_gelu<APPROX_MODE>(dst_index);
}

// ReLU is simpler
inline void llk_math_eltwise_unary_sfpu_relu(uint dst_index) {
    sfpu_relu(dst_index);  // max(0, x) — a single SFPU instruction
}
```

### When you need to look at TT-LLK

You look at tt-llk when:
- A TTNN operation is producing incorrect numerical results and you need to understand the math implementation.
- You are adding a new mathematical primitive not currently in TTNN.
- You are porting to a new Tenstorrent chip (you need to implement chip-specific LLK variations).
- You want to understand the actual hardware precision and rounding behaviour.

---

## 4.6 Deterministic Routing — How the Compiler Hardcodes Data Paths

### What deterministic routing means

When the TT-Forge compiler processes a model, it assigns:
1. Each operation → specific Tensix core(s)
2. Each tensor transfer → specific NoC path (sequence of router hops)
3. Each circular buffer → specific size (in tiles)
4. Execution order → fully determined

At runtime, nothing is decided dynamically. Every NoC packet follows the exact path the compiler chose.

### Why this is possible for transformer models

Transformer models (and most neural networks) have a fixed graph structure:
- The same sequence of operations happens every decode step.
- The shapes of every tensor are known at compile time (if the sequence length is fixed, or if dynamic shapes are handled by compiling multiple variants).
- The same data access pattern repeats every inference.

Because the graph is static, the compiler can solve a placement and routing problem once, at compile time, and reuse the solution for every inference.

### The NoC routing algorithm

```
Problem the compiler solves:
  
  Given:
    - 10×12 grid of Tensix cores
    - A directed graph of operations and data dependencies
    - Estimated compute time per op
    - NoC link bandwidth per link
  
  Find:
    - Assignment of ops to cores that minimises total execution time
    - Routing paths for all inter-core transfers that avoid congestion
    - Circular buffer sizes that prevent stalls without wasting SRAM
```

This is an NP-hard optimisation problem in general. TT-Forge uses heuristics and cost models:

**Placement:** Operations in a sequential pipeline are placed on adjacent cores to minimise NoC hop count. Parallel operations (tensor-parallel matmuls) are spread across a row or column of cores.

**Routing:** The compiler uses shortest-path routing on the 2D torus, with load balancing to spread traffic across both NoC planes.

**Buffer sizing:** The compiler computes a minimum safe buffer size as: `buffer_size = producer_throughput × consumer_latency`. Too small → producer stalls. Too large → wasted SRAM.

### What the output looks like (conceptual)

After compilation, the binary contains:
```
For each core in the 10×12 grid:
  - Reader RISC-V binary (compiled BRISC kernel)
  - Compute RISC-V binary (compiled TRISC kernel)
  - Writer RISC-V binary (compiled NCRISC kernel)
  - Runtime argument table (addresses of buffers in GDDR6/SRAM)
  - NoC routing configuration (which core to send output to, what address)
  - Circular buffer configuration (sizes, data formats)
```

All of this is loaded onto the chip once. Then every inference just says "go" — the cores execute their pre-loaded programs.

---

## 4.7 TT-Lang — the Future Programming Model

### What TT-Lang is

TT-Lang is a new programming language being developed by Tenstorrent to make explicit spatial programming easier and safer. It sits above TT-Metalium in the abstraction hierarchy.

Source: [github.com/tenstorrent/tt-lang](https://github.com/tenstorrent/tt-lang)

Status (as of 2024): active development, not yet production-ready.

### The problem TT-Metalium has

Writing Metalium code requires managing three separate kernel files per core, reasoning about circular buffer synchronisation manually, and thinking in terms of physical core coordinates. This is error-prone.

Example of what can go wrong in Metalium:
```cpp
// Bug: forgot to call cb_push_back after cb_reserve_back
// Compute kernel will wait forever for input that is never signaled as ready
cb_reserve_back(cb_id_out, 1);
// ... write to CB ...
// ← missing: cb_push_back(cb_id_out, 1);
// Consumer stalls permanently
```

### TT-Lang's approach

TT-Lang introduces:
- A single unified kernel language (no separate reader/compute/writer files)
- Safe circular buffer abstractions (the language ensures push/pop are balanced)
- A spatial type system: tensors have types that encode where they live (SRAM vs DRAM)
- Compiler-driven placement (you describe the computation, the compiler chooses which RISC-V handles each part)

```
// Conceptual TT-Lang syntax (not final):

@spatial_fn
fn matmul_layer(input: Tensor<SRAM>, weight: Tensor<DRAM>) -> Tensor<SRAM> {
    // Compiler decides this is a reader operation for BRISC
    let w_tile = dram_load(weight, tile_idx);
    
    // Compiler decides this is a compute operation for TRISC
    let result = matmul_tile(input, w_tile);
    
    // Compiler decides this is a writer operation for NCRISC
    return result;
}
```

The spatial type system and ownership model prevent the synchronisation bugs common in Metalium.

### Relationship to Metalium

TT-Lang compiles to Metalium — it generates reader/compute/writer kernel files. Think of TT-Lang as a safer, higher-level way to express what Metalium expresses. The underlying hardware interaction is identical.

---

# Phase 5 — Scale-Out: Multi-Chip and Cluster

## 5.1 Ethernet Scale-Out — NoC Beyond the Chip

### How the NoC extends over Ethernet

Each Tenstorrent chip has Ethernet PHY blocks at its edges. These act as NoC endpoints: when a RISC-V on Core #119 (the last core on the grid edge) sends a NoC packet to an address that maps to another chip, the packet goes through the Ethernet PHY, over a standard cable, and appears at the receiving chip's Ethernet PHY, then continues on that chip's NoC.

```
Chip A (10×12 Tensix grid)        Chip B (10×12 Tensix grid)
┌─────────────────────────┐       ┌─────────────────────────┐
│ C00 ─ C01 ─ ... ─ C09  │       │ C00 ─ C01 ─ ... ─ C09  │
│  │    │           │     │       │  │    │           │     │
│ C10 ─ C11 ─ ... ─ C19  │       │ C10 ─ C11 ─ ... ─ C19  │
│  │                │     │       │                   │     │
│ ... (middle rows) ...   │       │ ... (middle rows) ...   │
│  │                │     │       │                   │     │
│C110─ C111 ─ ...─C119   │       │C110─ C111 ─ ...─C119   │
│                   │     │       │ │                       │
│               [ETH PHY]│──────►│[ETH PHY]               │
│               400 GbE  │       │                         │
└─────────────────────────┘       └─────────────────────────┘

From the compiler's perspective: C119 of Chip A and C00 of Chip B
are just neighbours in a larger grid. The Ethernet hop is transparent.
```

### Why standard Ethernet, not NVLink

NVIDIA's NVLink:
- Proprietary protocol, high bandwidth (900 GB/s bidirectional for NVLink 4.0)
- Only connects a small fixed number of GPUs (up to 8 with NVSwitch)
- Requires dedicated NVSwitch hardware for larger clusters
- Expensive and inflexible

Tenstorrent's choice of Ethernet:
- Open standard, commodity switches and cables
- Can connect arbitrary numbers of chips using standard data-centre infrastructure
- 400 GbE × 14 ports per Blackhole chip = up to 5.6 Tb/s of inter-chip bandwidth
- Lower peak bandwidth per link than NVLink, but more ports and cheaper

Ethernet latency is higher than NVLink (~microseconds vs sub-microsecond). Tenstorrent addresses this by ensuring that only operations that can tolerate inter-chip latency (e.g., layer boundaries with large tensor transfers) cross chip boundaries.

### What the compiler does with multi-chip

The compiler receives the total cluster topology (how many chips, how they are connected). It:
1. Partitions the model graph across chips.
2. Assigns NoC routes that include Ethernet hops.
3. Treats the whole cluster as one logical NoC grid.
4. No user intervention needed — `tt_forge.compile(model, device=cluster)` handles it.

---

## 5.2 Galaxy System — the Server Rack as One Chip

### Galaxy hardware configuration

The Tenstorrent Galaxy is their reference server system. The current configuration uses Wormhole chips:

```
Galaxy server:
  32 × Wormhole n300 boards
  = 64 × Wormhole chips (each n300 has 2 chips)
  
  Chips connected in a 2D mesh over 100 GbE Ethernet
  Total SRAM: 64 × 120 × 1.5 MB = 11.52 GB
  Total GDDR6: 64 × 24 GB = 1.536 TB
  Total compute: 64 × ~236 TFLOP/s = ~15 PFLOP/s (FP16)
  
  The compiler sees this as a single 960-core grid
  (with Ethernet latency on the inter-chip links).
```

### Running LLaMA 70B on Galaxy

LLaMA-2 70B in FP16 requires 140 GB. It does not fit on one chip (24 GB GDDR6). On Galaxy:

```
Tensor parallelism (within a layer):
  The weight matrix W_Q [8192, 8192] is split across 8 chips.
  Each chip has W_Q_shard [8192, 1024].
  Each chip computes its portion: input × W_Q_shard → partial_Q
  AllReduce over Ethernet combines partial results.

Pipeline parallelism (across layers):
  Layers 1–8   → chips 1–8
  Layers 9–16  → chips 9–16
  Layers 17–24 → chips 17–24
  Layers 25–32 → chips 25–32
  ...
  
  Token 1 enters chip group 1, when it finishes and moves to chip group 2,
  Token 2 enters chip group 1 simultaneously.
  At steady state, all chip groups process different tokens simultaneously.
```

---

## 5.3 Model Parallelism Strategies

### Tensor Parallelism (TP)

Split individual operations across multiple cores or chips. Every core/chip participates in every layer.

```
W [4096, 4096] split across 4 chips:
  Chip 0: W_0 [4096, 1024]  → partial_output_0 [batch, 1024]
  Chip 1: W_1 [4096, 1024]  → partial_output_1 [batch, 1024]
  Chip 2: W_2 [4096, 1024]  → partial_output_2 [batch, 1024]
  Chip 3: W_3 [4096, 1024]  → partial_output_3 [batch, 1024]
  
  AllReduce (concatenate along dim 1): full_output [batch, 4096]
  
  Requires: inter-chip communication every layer
  Good for: models with large individual layers
  Bad for: high inter-chip latency
```

### Pipeline Parallelism (PP)

Split layers across chips. Each chip runs a subset of layers.

```
  LLaMA-2 70B: 80 layers
  4 chips: Chip 0 runs layers 1–20, Chip 1 runs 21–40, etc.
  
  Chip 0: processes token → sends activations to Chip 1 via Ethernet
  Chip 1: processes → sends to Chip 2
  Chip 2: processes → sends to Chip 3
  Chip 3: produces output
  
  Requires: inter-chip communication once per pipeline stage
  Good for: very large models that don't fit on one chip
  Bad for: high pipeline bubble (chip 0 idle while chips 2/3 compute)
```

### Sequence Parallelism (SP)

Split the input sequence across cores. Each core processes a portion of the sequence.

```
  Sequence length 4096 split across 4 cores:
  Core 0: tokens 0–1023
  Core 1: tokens 1024–2047
  Core 2: tokens 2048–3071
  Core 3: tokens 3072–4095
  
  Each core computes QKV for its tokens.
  For attention: cores share K and V (they all need the full sequence).
  
  Requires: AllGather of K, V across cores
  Good for: very long sequences (reduces per-core SRAM pressure)
```

### How the compiler automates this

TT-Forge's placement pass selects a combination of TP, PP, and SP automatically based on:
- Model size vs. available SRAM per chip
- Number of chips available
- Target batch size
- Inter-chip latency vs. intra-chip NoC latency

The user specifies the device cluster and the model. The compiler chooses the parallelism strategy.

---

# Phase 6 — Advanced Topics and Trade-offs

## 6.1 Quantisation and Data Formats

### Why quantisation matters on Tenstorrent

SRAM is the bottleneck resource. The Tensix core has 1.5 MB per core. Every format decision directly affects:
- How many weight tiles fit in SRAM (fewer GDDR6 accesses)
- Matrix Engine throughput (smaller formats = faster)
- Numerical accuracy (smaller formats = less precision)

### Block Floating Point (BFP) — Tenstorrent's key format

Standard floating point: each number has its own sign, exponent, and mantissa.

```
FP32 (standard float):
  [sign 1bit][exponent 8bit][mantissa 23bit] = 32 bits per number

BFP8 (block float 8):
  One tile = 32×32 = 1024 values
  Organised into blocks of 16 values (one row):
  [shared exponent 8bit][mantissa0 7bit][mantissa1 7bit]...[mantissa15 7bit]
  = 8 bits + (16 × 7 bits) / 16 = 8 bits per number average
  
  Key insight: within one row of a matrix tile, the values often have
  similar magnitudes (same exponent). The shared exponent costs only
  1/16th extra.
```

```
BFP4 (block float 4):
  Same idea, but 3-bit mantissa per value.
  = ~4 bits per number
  
  2× more values fit in SRAM vs BFP8.
  2× higher throughput on Matrix Engine.
  More numerical error — appropriate for weights of some layers, not all.
```

### Matrix Engine throughput by format

| Format | Bits/value | Relative throughput | Notes |
|---|---|---|---|
| FP32 | 32 | 1× | Rarely used for inference on-chip |
| FP16 / BF16 | 16 | 2× | Standard inference format |
| BFP8 | 8 | 4× | Good accuracy, 4× more values in SRAM |
| BFP4 | 4 | 8× | Acceptable for most weights, not all |

### Per-layer format assignment

Not all layers tolerate aggressive quantisation equally:

```
Layers that tolerate BFP4:    Layers that need BFP8 or FP16:
──────────────────────────    ────────────────────────────────────
Most FFN weight matrices       First and last embedding layers
Middle transformer layers      LayerNorm weights and biases
                               Q, K, V projection weights
                               Output projection (quality-sensitive)
```

TT-Forge can assign different formats per layer based on a sensitivity analysis pass or user-specified per-layer configuration.

### Activations vs weights

- **Weights** are static — they stay in BFP8/BFP4 for the entire inference.
- **Activations** (intermediate results) often need BFP8 minimum because they can have higher dynamic range than weights.

---

## 6.2 When Tenstorrent Loses — Honest Limits

### Case 1: Sparse models

LLMs with structured or unstructured sparsity have irregular memory access patterns. If 50% of weights are zero, ideally you skip those multiplications. But:

```
Dense matrix multiply (good for Tenstorrent):
  All tiles are loaded in a regular, predictable pattern.
  BRISC can be programmed to fetch exactly the tiles needed.
  
Sparse matrix multiply (harder):
  Which tiles are non-zero depends on the sparsity pattern.
  Pattern may not be known statically at compile time.
  Irregular NoC access patterns are harder to compile-time route.
  Hardware prefetcher (GPU) adapts at runtime; explicit RISC-V must be re-programmed.
```

NVIDIA GPUs handle structured sparsity (2:4 sparsity) in hardware with dedicated sparse Tensor Cores. Tenstorrent has no equivalent hardware unit.

### Case 2: Dynamic shapes

Transformers with variable sequence lengths:

```
Fixed sequence length (works well):
  Compile once for seq_len=2048.
  Compiler knows all tensor shapes → optimal circular buffer sizes → optimal routing.

Dynamic sequence length (harder):
  Must either:
    a) Compile multiple variants (seq_len=512, 1024, 2048, 4096) → high compile time
    b) Pad all inputs to maximum → wastes compute on padding tokens
    c) Use a dynamic shape runtime → loses static routing optimality
```

GPUs handle dynamic shapes natively because every CUDA kernel dispatches at runtime with size parameters. Tenstorrent's compile-time approach is fundamentally harder to adapt dynamically.

### Case 3: Training workloads

During training:
- You need to compute gradients (backward pass): more memory, more compute.
- You need an optimiser step (Adam, etc.): reads/writes large optimiser state.
- Gradient accumulation and all-reduce across chips: lots of inter-chip communication.
- Gradient checkpointing: deliberately writes activations to DRAM and recomputes them (the opposite of Tenstorrent's "keep in SRAM" philosophy).

For training, HBM's higher capacity (80–192 GB vs 24–32 GB GDDR6) means more of the model, gradients, and optimiser states fit on one device, reducing inter-chip communication.

**Current status:** TT-Forge/Metalium training support exists but is less mature than inference support.

### Case 4: Very small batch sizes with high model utilisation

Tenstorrent's spatial pipeline works best when many tokens flow through simultaneously (high throughput). For single-token, latency-critical inference:

```
Batch size 1, single token decode:
  The spatial pipeline is mostly empty.
  Core A finishes token 1 → Core B starts on token 1 → Core A idles.
  Only 1 core active at a time per pipeline stage.
  
  On GPU: all 132 SMs work on the same token (different parts of the model).
  Higher parallelism per token at low batch size.
```

For latency-optimised single-request inference, NVIDIA GPUs often beat Tenstorrent's throughput-optimised spatial pipeline.

### Case 5: SRAM capacity ceiling

With 1.5 MB per core, extremely large attention operations (very long context windows) require splitting the KV cache across many cores, which requires coordinated NoC communication across the full chip.

For context lengths > 128K tokens (modern long-context models), the KV cache dominates the GDDR6 capacity (e.g., at 128K tokens, LLaMA-2 7B KV cache ≈ 16 GB). This fits in 24 GB GDDR6 but just barely.

For 192 GB HBM on AMD MI300X, extremely long contexts are trivially handled.

---

## 6.3 Competitive Landscape

### NVIDIA H100 / H200 / B200

The market leader. Architecture: streaming multiprocessors (SMs), HBM, CUDA.

| | NVIDIA H100 SXM | Tenstorrent Blackhole p150 |
|---|---|---|
| Compute (FP16) | 989 TFLOP/s | 786 TFLOP/s |
| Memory BW | 3,350 GB/s (HBM3) | 576 GB/s (GDDR6) |
| Memory capacity | 80 GB | 32 GB |
| On-chip SRAM | ~83 MB | 180 MB |
| TDP | 700W | 300W |
| Retail price | ~$30,000 | ~$2,400–$3,200 |
| Software maturity | Mature (CUDA, cuDNN) | Growing (tt-metal, tt-forge) |

H100 wins: training, dynamic shapes, sparse models, software ecosystem.
Blackhole wins: inference throughput per dollar, power efficiency, SRAM capacity.

### AMD MI300X

The second-place competitor. Uses an MCM (multi-chip module) with CPU and GPU chiplets.

- 192 GB HBM3, 5,300 GB/s bandwidth (highest in the market)
- 1,307 TFLOP/s (FP16)
- Wins on memory capacity: 192 GB fits the largest models on one card
- Uses ROCm (HIP) for programming — broadly CUDA-compatible

### Groq LPU

Most architecturally similar to Tenstorrent. Also uses on-chip SRAM only — no DRAM on the chip.

```
Groq LPU TS-1:
  896 KB SRAM per slice × 2,304 slices = 2 GB total SRAM
  NO DRAM on chip — entire model must fit in SRAM
  If model doesn't fit in SRAM → multiple LPU chips needed
  
  Advantage: deterministic, ultra-low latency (no DRAM access ever)
  Disadvantage: 2 GB capacity; large models require many chips
```

Groq achieves very high tokens/second for models that fit in SRAM, but the architecture is rigid compared to Tenstorrent's SRAM + GDDR6 combination.

### Graphcore IPU (Intelligence Processing Unit)

- Uses Bulk Synchronous Parallel (BSP) execution model
- 900 MB on-chip SRAM (IPU Bow)
- No DRAM — all state must fit in SRAM or be streamed from host
- Programming model (Poplar/PopART) is proprietary and quite different from CUDA
- Acquired by SoftBank in 2023; slower ecosystem growth

### Cerebras WSE-3

The wafer-scale engine — the entire 300mm silicon wafer is one chip.

```
Cerebras WSE-3:
  900,000 AI cores on one 300mm wafer
  44 GB of SRAM distributed across the wafer
  No DRAM — entire model lives in on-chip SRAM
  
  For very large models that fit: extraordinary throughput
  Limitation: cannot currently use standard server infrastructure;
  requires Cerebras CS-3 system
```

### SambaNova SN40L

Uses a Reconfigurable Dataflow Unit (RDU) architecture.

- On-chip SRAM + HBM combination
- Proprietary dataflow programming model
- Positioned primarily for fine-tuning and training pipelines

---

## 6.4 Reading Primary Sources

### GitHub repository navigation

#### tt-metal (most important repo)

```
tt-metal/
├── tt_metal/                    # Core SDK (C++)
│   ├── hw/
│   │   ├── firmware/src/        # RISC-V firmware loaded onto each core
│   │   │   ├── brisc.cc         # BRISC main loop
│   │   │   ├── ncrisc.cc        # NCRISC main loop
│   │   │   └── trisc*.cc        # TRISC main loops
│   │   ├── ckernels/            # Chip-specific compute kernels
│   │   │   ├── wormhole_b0/     # Wormhole-specific
│   │   │   └── blackhole/       # Blackhole-specific
│   │   └── inc/
│   │       ├── dataflow_api.h   # NoC and CB APIs used in data movement kernels
│   │       └── compute_kernel_api/ # APIs for compute kernels
│   ├── impl/
│   │   ├── dispatch/            # Host-side kernel dispatch
│   │   └── buffers/             # Buffer management (SRAM, DRAM)
│   └── api/                     # Public C++ API headers
│
├── ttnn/                        # TTNN tensor library
│   ├── cpp/ttnn/operations/     # Individual op implementations
│   │   ├── matmul/              # Matrix multiply
│   │   ├── transformer/         # Fused transformer ops
│   │   └── reduction/           # Sum, mean, etc.
│   └── python_api/              # Python wrappers
│
└── tests/
    ├── tt_metal/                # Low-level tests (good for learning APIs)
    └── ttnn/                    # TTNN op tests
```

**Start reading:** `tt_metal/programming_examples/` — these are self-contained examples showing each concept step by step.

#### tt-forge

```
tt-forge/
├── forge/
│   ├── csrc/
│   │   ├── tt_torch_device/     # torch.compile backend
│   │   ├── passes/              # MLIR compiler passes
│   │   │   ├── lower_to_ttir.cpp
│   │   │   └── lower_to_ttnn.cpp
│   │   └── tt_device/           # Device management
│   └── test/
│       └── mlir/                # Compiler pass tests (show what each pass does)
```

#### tt-llk

```
tt-llk/
├── common/inc/
│   ├── llk_math_matmul.h        # Tile matrix multiply
│   ├── llk_math_eltwise_*.h     # Element-wise ops
│   ├── llk_pack*.h              # Packer control
│   └── llk_unpack*.h            # Unpacker control
└── wormhole_b0/                 # Chip-specific overrides
```

### Tenstorrent documentation

- Main docs: [docs.tenstorrent.com](https://docs.tenstorrent.com)
  - TT-Metalium programming guide
  - TTNN API reference
  - Architecture overviews per chip
- TT-Metal GitHub README has a getting-started guide with Docker setup.

### Academic papers to read alongside this

These papers describe the academic foundations of the architectural ideas in Tenstorrent:

| Paper | Why relevant |
|---|---|
| "Eyeriss: A Spatial Architecture for Energy-Efficient Dataflow for Convolutional Neural Networks" (Chen et al., 2016) | Foundational paper on spatial dataflow architectures |
| "MAESTRO: A Data-Centric Approach to Understand Reuse, Performance, and Hardware Cost of DNN Mappings" (Kwon et al., 2019) | Framework for analysing dataflow mappings |
| "SCNN: An Accelerator for Compressed-sparse Convolutional Neural Networks" (Parashar et al., 2017) | On handling sparse computation in spatial accelerators |
| "In-Datacenter Performance Analysis of a Tensor Processing Unit" (Jouppi et al., 2017, Google TPU) | Another spatial architecture; good contrast |
| "Efficiently Scaling Transformer Inference" (Pope et al., 2022, Google) | Analyses memory bandwidth bottlenecks in LLM inference — directly relevant to why Tenstorrent's architecture exists |

### Conference proceedings

- **Hot Chips** (hotchips.org): Annual symposium where Tenstorrent has presented chip designs. Search for "Tenstorrent" in Hot Chips 2021, 2023 proceedings.
- **ISSCC** (International Solid-State Circuits Conference): For silicon implementation details.
- **MLSys**: For software and compiler work.

---

# Quick Reference

## Memory speed cheat sheet

| Memory | Latency | Bandwidth (typical AI card) | Where |
|---|---|---|---|
| Tensix SRAM | ~1–5 ns | Terabytes/s (local) | On-chip, per core |
| H100 HBM3 | ~50 ns | 3,350 GB/s | On-package, chip+HBM |
| Blackhole GDDR6 | ~80 ns | 576 GB/s | On board, off chip |
| PCIe 5.0 x16 | ~1 µs | ~128 GB/s (bidirectional) | Host ↔ card |
| NVMe SSD | ~0.1 ms | ~14 GB/s (PCIe 5.0) | Storage |

## Tensix core role mapping

| RISC-V core | Name | Does |
|---|---|---|
| BRISC | "Brisc" | GDDR6/NoC → SRAM read |
| NCRISC | "Ncrisc" | SRAM → NoC/GDDR6 write |
| TRISC0 | Math 0 | Unpacker control |
| TRISC1 | Math 1 | Matrix Engine control |
| TRISC2 | Math 2 | Packer control |

## Software stack layers (top to bottom)

```
PyTorch / JAX model
      ↓
TT-Forge compiler (MLIR: TOSA → TTIR → TTNN dialects)
      ↓
TTNN tensor library (Python/C++ API)
      ↓
TT-Metalium SDK (kernel definition, CB management, NoC config)
      ↓
TT-LLK (math kernel implementations: matmul, SFPU ops)
      ↓
RISC-V firmware (BRISC, NCRISC, TRISC running on each Tensix)
      ↓
Hardware (Matrix Engine, SFPU, NoC, SRAM, GDDR6)
```

## Data format comparison

| Format | Bits | Values per 32×32 tile | Accuracy |
|---|---|---|---|
| FP32 | 32 | 1,024 values × 4 bytes = 4 KB | Highest |
| FP16/BF16 | 16 | 1,024 values × 2 bytes = 2 KB | High |
| BFP8 | ~8 | 1,024 values × 1 byte = 1 KB | Good |
| BFP4 | ~4 | 1,024 values × 0.5 byte = 512 B | Moderate |

## What each GitHub repo is for

| Repo | Use it when |
|---|---|
| tt-forge | Compiling PyTorch/JAX models to run on Tenstorrent hardware |
| tt-metal/ttnn | Writing TTNN ops, understanding tensor operations |
| tt-metal (root) | Writing bare-metal Metalium kernels for full hardware control |
| tt-llk | Understanding or modifying the lowest-level math implementations |
| tt-lang | Experimenting with the next-generation programming model |

## Chip generation summary

| Chip | Tensix | SRAM total | Ethernet | Memory | Released |
|---|---|---|---|---|---|
| Grayskull | 120 | 120 MB | None | LPDDR4 | 2021 |
| Wormhole | 80 | 120 MB | 16 × 100 GbE | GDDR6 12/24 GB | 2022 |
| Blackhole | 120 | 180 MB | 14 × 400 GbE | GDDR6 16/32 GB | 2024 |
