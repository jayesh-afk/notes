# Tenstorrent Hands-On Learning Guide
### Windows WSL Ubuntu Edition — Zero to Dirty Hands

> **Who this is for:** You have a new employer laptop with WSL Ubuntu already installed. You have zero Tenstorrent hardware. You want to get hands-on with every concept in the Tenstorrent Mastery doc — using software simulation and QEMU where real hardware isn't available.
>
> **Promise:** Every step is written so a beginner can follow it without guessing. If a command needs to be typed, you will see the exact command. If something might go wrong, there is a fix.

---

## Table of Contents

- [Part 0 — Understanding What We're Setting Up](#part-0--understanding-what-were-setting-up)
- [Part 1 — WSL Check and First-Time Setup](#part-1--wsl-check-and-first-time-setup)
- [Part 2 — Git Mastery (New Employer Laptop)](#part-2--git-mastery-new-employer-laptop)
- [Part 3 — System Prerequisites](#part-3--system-prerequisites)
- [Part 4 — Python Environment](#part-4--python-environment)
- [Part 5 — Cloning and Building tt-metal](#part-5--cloning-and-building-tt-metal)
- [Part 6 — Running tt-metal in Simulation Mode](#part-6--running-tt-metal-in-simulation-mode)
- [Part 7 — QEMU RISC-V: Understand Tensix Internals](#part-7--qemu-risc-v-understand-tensix-internals)
- [Part 8 — Lab 1: Memory Wall and Roofline (Phase 1)](#part-8--lab-1-memory-wall-and-roofline-phase-1)
- [Part 9 — Lab 2: Your First Metalium Kernel (Phase 3 + 4)](#part-9--lab-2-your-first-metalium-kernel-phase-3--4)
- [Part 10 — Lab 3: TTNN Tensor Operations (Phase 4)](#part-10--lab-3-ttnn-tensor-operations-phase-4)
- [Part 11 — Lab 4: Dataflow and Circular Buffers (Phase 3)](#part-11--lab-4-dataflow-and-circular-buffers-phase-3)
- [Part 12 — Lab 5: tt-forge with a PyTorch Model (Phase 4)](#part-12--lab-5-tt-forge-with-a-pytorch-model-phase-4)
- [Part 13 — Lab 6: Exploring the Repo Structure (Phase 4 + 6)](#part-13--lab-6-exploring-the-repo-structure-phase-4--6)
- [Part 14 — Git Workflow for Day-to-Day Learning](#part-14--git-workflow-for-day-to-day-learning)
- [Part 15 — Cheat Sheet and Quick Reference](#part-15--cheat-sheet-and-quick-reference)

---

## Part 0 — Understanding What We're Setting Up

### Why WSL (not native Windows, not a VM, not QEMU machine)?

Think of WSL (Windows Subsystem for Linux) like a secret Linux door inside your Windows laptop. When you open it, you get a real Ubuntu terminal. Programs running inside it see a real Linux filesystem and real Linux tools.

```
Your Windows Laptop
│
├── Windows side  (Word, Chrome, Explorer)
│
└── WSL2 Ubuntu side  ← THIS is where we do everything
    ├── /home/yourname/        ← your Linux home folder
    ├── /usr/bin/gcc           ← C compiler
    ├── /usr/bin/python3       ← Python
    └── /mnt/c/               ← you can see your Windows C: drive from here
```

**Why not native Windows?** Tenstorrent's entire software stack (tt-metal, tt-forge) is Linux-only. It will not build on Windows natively. Period.

**Why not QEMU for the whole machine?** QEMU would be slow and complex to set up. WSL2 is much faster and already on your machine.

**What is QEMU used for in this guide?** Only for Part 7 — where we use `qemu-riscv32` to actually run and inspect the tiny RISC-V programs (BRISC, NCRISC, TRISC) that run inside each Tensix core. That's a learning exercise, not the main workflow.

### What you can do WITHOUT physical Tenstorrent hardware

| What | How | Requires HW? |
|---|---|---|
| Read and understand all APIs | Read source code + docs | No |
| Build tt-metal from source | cmake + make | No |
| Run host-side TTNN operations | CPU fallback mode | No |
| Run programming examples (data movement) | Software simulation | No |
| Explore the NoC routing compiler | Build + inspect output | No |
| Run tt-forge with a PyTorch model | CPU mode | No |
| Run actual Tensix kernels at speed | Physical TT card | **Yes** |
| Benchmark real memory bandwidth | Physical TT card | **Yes** |

The guide is honest: a few things need real hardware. But 80% of the learning — all the concepts from the mastery doc — you can do in WSL right now.

---

## Part 1 — WSL Check and First-Time Setup

### Step 1.1 — Open your WSL terminal

On Windows:
1. Press the **Windows key** on your keyboard
2. Type `wsl` and press Enter
3. A black terminal window opens — that's Ubuntu Linux inside Windows

If it opens without errors, you're in. You'll see something like:

```
yourname@LAPTOP-ABC123:~$
```

The `~` means you're in your home folder. The `$` means you're a normal user (not admin).

### Step 1.2 — Check your WSL version (must be WSL2, not WSL1)

In the WSL terminal, type this and press Enter:

```bash
cat /proc/version
```

You should see something with `microsoft-standard-WSL2` in the output. Example:

```
Linux version 5.15.167.4-microsoft-standard-WSL2 ...
```

If it says `WSL2` anywhere — you're fine. ✅

If it says `WSL1` — open Windows PowerShell (not WSL) and run:

```powershell
wsl --set-version Ubuntu 2
```

### Step 1.3 — Check your Ubuntu version

In the WSL terminal:

```bash
lsb_release -a
```

You should see Ubuntu 20.04, 22.04, or 24.04. All of these work. Write down which one you have — you'll need this later.

### Step 1.4 — Make sure you can connect to the internet from WSL

```bash
ping -c 3 google.com
```

You should see lines like `64 bytes from ...`. Press Ctrl+C to stop. If this fails, WSL networking has a problem — tell your IT department you need WSL internet access.

### Step 1.5 — Update Ubuntu's package lists

Think of this like refreshing the app store — it downloads the latest list of what's available:

```bash
sudo apt update
```

`sudo` means "run as administrator". Ubuntu will ask for your WSL password. Type it (you won't see it as you type — that's normal) and press Enter.

Then upgrade any outdated packages:

```bash
sudo apt upgrade -y
```

The `-y` means "say yes to everything automatically". This may take a few minutes.

---

## Part 2 — Git Mastery (New Employer Laptop)

Git is the tool that tracks changes in code. Every professional developer uses it daily. Since this is a new employer laptop, we set it up fresh — cleanly.

### Step 2.1 — Install Git

```bash
sudo apt install git -y
```

Verify it installed:

```bash
git --version
```

You should see: `git version 2.x.x`

### Step 2.2 — Configure your identity

Git needs to know who you are. Use your work email (or personal if this is a personal project):

```bash
git config --global user.name "Your Full Name"
git config --global user.email "your.email@example.com"
```

Replace `Your Full Name` with your actual name and `your.email@example.com` with your email. Keep the quotes.

Set a comfortable default editor (nano is beginner-friendly):

```bash
git config --global core.editor nano
```

Set the default branch name to `main` (modern standard):

```bash
git config --global init.defaultBranch main
```

### Step 2.3 — Verify your Git config

```bash
git config --global --list
```

You'll see your name, email, editor. All good.

### Step 2.4 — Generate an SSH key for GitHub (one-time setup)

SSH keys let GitHub know it's really you — no password typing every time.

**Generate the key:**

```bash
ssh-keygen -t ed25519 -C "your.email@example.com"
```

It will ask three questions:
- `Enter file to save the key:` — just press **Enter** (accepts the default location)
- `Enter passphrase:` — press **Enter** (no passphrase, fine for learning)
- `Enter same passphrase again:` — press **Enter** again

**See your public key:**

```bash
cat ~/.ssh/id_ed25519.pub
```

You'll see a line starting with `ssh-ed25519 AAAA...`. Copy that entire line.

**Add it to GitHub:**
1. Go to https://github.com in your browser
2. Click your profile picture (top right) → **Settings**
3. Click **SSH and GPG keys** (left sidebar)
4. Click **New SSH key**
5. Title: `WSL Laptop`
6. Paste the line you copied
7. Click **Add SSH key**

**Test it:**

```bash
ssh -T git@github.com
```

You'll see: `Hi yourname! You've successfully authenticated...` ✅

### Step 2.5 — Create your learning repository

This is where you'll save all your notes, experiments, and exercises:

```bash
mkdir -p ~/tenstorrent-learning
cd ~/tenstorrent-learning
git init
```

Create a first file:

```bash
echo "# My Tenstorrent Learning Journal" > README.md
git add README.md
git commit -m "Initial commit: start learning journey"
```

**Git workflow you'll use every day:**

```bash
# 1. See what changed
git status

# 2. Add changed files to "staging" (like putting stuff in a box before shipping)
git add filename.py
# or add everything:
git add .

# 3. Commit (save a snapshot with a message)
git commit -m "Add lab 1 memory roofline notes"

# 4. If you've connected to GitHub, push (upload):
git push origin main
```

### Step 2.6 — Create a GitHub repo and connect it (optional but recommended)

1. Go to https://github.com → click **+** → **New repository**
2. Name it `tenstorrent-learning`
3. Keep it **Private** (employer laptop)
4. Click **Create repository**
5. GitHub shows you commands. Since you already have a local repo, use the SSH version:

```bash
git remote add origin git@github.com:YOUR_USERNAME/tenstorrent-learning.git
git branch -M main
git push -u origin main
```

Replace `YOUR_USERNAME` with your GitHub username.

### Step 2.7 — The git commands you need to know (cheat sheet)

```bash
git status              # What files changed?
git diff                # See the actual changes line by line
git add .               # Stage all changed files
git add path/to/file    # Stage one specific file
git commit -m "msg"     # Save a snapshot
git log --oneline       # See history (press q to quit)
git log --oneline -10   # See last 10 commits only

git branch              # List all branches
git branch my-feature   # Create a new branch
git checkout my-feature # Switch to that branch
git checkout main        # Switch back to main

git clone <url>         # Download a repository from GitHub
git pull                # Download latest changes from GitHub
git push                # Upload your commits to GitHub

git stash               # Temporarily hide your changes
git stash pop           # Bring them back
```

---

## Part 3 — System Prerequisites

This installs all the tools needed to build Tenstorrent software from source.

### Step 3.1 — Install essential build tools

```bash
sudo apt install -y \
    build-essential \
    cmake \
    ninja-build \
    pkg-config \
    git \
    curl \
    wget \
    unzip \
    zip \
    software-properties-common
```

`build-essential` includes `gcc`, `g++`, and `make` — the tools that turn C++ source code into programs you can run.

### Step 3.2 — Install Clang/LLVM (Tenstorrent uses Clang, not GCC)

```bash
wget https://apt.llvm.org/llvm.sh
chmod +x llvm.sh
sudo ./llvm.sh 17
```

Set Clang 17 as the default:

```bash
sudo update-alternatives --install /usr/bin/clang clang /usr/bin/clang-17 100
sudo update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-17 100
```

Verify:

```bash
clang --version
```

Should say `clang version 17.x.x`.

### Step 3.3 — Install QEMU for RISC-V (for Part 7)

QEMU is an emulator — it pretends to be different hardware so you can run RISC-V programs on your x86 laptop:

```bash
sudo apt install -y \
    qemu-user \
    qemu-user-static \
    gcc-riscv64-linux-gnu \
    g++-riscv64-linux-gnu \
    binutils-riscv64-linux-gnu
```

Verify QEMU RISC-V:

```bash
qemu-riscv64 --version
```

### Step 3.4 — Install Python build dependencies

```bash
sudo apt install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    libpython3-dev \
    python3-setuptools
```

### Step 3.5 — Install other libraries tt-metal needs

```bash
sudo apt install -y \
    libhwloc-dev \
    libnuma-dev \
    libboost-all-dev \
    libyaml-cpp-dev \
    libgtest-dev \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libncurses-dev
```

### Step 3.6 — Verify everything installed correctly

```bash
echo "=== Checking tools ==="
gcc --version | head -1
clang --version | head -1
cmake --version | head -1
python3 --version
qemu-riscv64 --version | head -1
echo "=== All checks done ==="
```

All should print version numbers, not errors.

---

## Part 4 — Python Environment

We use **virtual environments** — isolated Python sandboxes. This means packages for Tenstorrent don't interfere with anything else on your machine.

### Step 4.1 — Create a virtual environment

```bash
cd ~
python3 -m venv tt-env
```

This creates a folder `~/tt-env/` with an isolated Python inside it.

### Step 4.2 — Activate the virtual environment

You must activate it every time you open a new terminal:

```bash
source ~/tt-env/bin/activate
```

Your prompt changes to show `(tt-env)` at the start:

```
(tt-env) yourname@LAPTOP:~$
```

**IMPORTANT:** Every time you open a new WSL terminal window and want to work on Tenstorrent stuff, run `source ~/tt-env/bin/activate` first.

### Step 4.3 — Install basic Python packages we'll use for labs

```bash
pip install --upgrade pip
pip install numpy matplotlib jupyter torch torchvision tqdm
```

This takes a few minutes. Let it run.

### Step 4.4 — Make activation automatic (optional but convenient)

Add it to your `.bashrc` so it activates automatically every time you open WSL:

```bash
echo 'source ~/tt-env/bin/activate' >> ~/.bashrc
source ~/.bashrc
```

---

## Part 5 — Cloning and Building tt-metal

`tt-metal` is the main Tenstorrent repository. It contains:
- The Metalium SDK (bare-metal kernel programming)
- TTNN (tensor library)
- Programming examples
- The software simulation layer

### Step 5.1 — Clone the repository

```bash
cd ~
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
```

This downloads the entire repository. It may take a few minutes (it's large).

### Step 5.2 — Understand the folder structure before building

```bash
ls -la
```

Key folders you'll explore:

```
tt-metal/
├── tt_metal/              ← The core C++ SDK
│   ├── hw/
│   │   ├── firmware/      ← RISC-V firmware (BRISC, NCRISC, TRISC code)
│   │   └── ckernels/      ← Compute kernels per chip type
│   ├── impl/              ← Dispatch, buffer management
│   └── api/               ← Public headers you write against
├── ttnn/                  ← Tensor library (higher-level than Metalium)
│   ├── cpp/               ← C++ op implementations
│   └── python_api/        ← Python wrappers
├── tests/                 ← Tests = great learning examples
│   ├── tt_metal/          ← Low-level tests
│   └── ttnn/              ← TTNN tests
└── programming_examples/  ← START HERE for learning
```

### Step 5.3 — Initialize git submodules

tt-metal depends on other repositories (submodules). Fetch them:

```bash
git submodule update --init --recursive
```

This may take several minutes.

### Step 5.4 — Set up the Python package in editable mode

```bash
source ~/tt-env/bin/activate
pip install -e ".[dev]"
```

If this fails with a missing package error, install that package and retry. Common fix:

```bash
pip install setuptools wheel
pip install -e ".[dev]"
```

### Step 5.5 — Build tt-metal

```bash
cmake -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DTT_METAL_BUILD_TESTS=OFF \
    -DENABLE_TRACY=OFF
```

Then build:

```bash
cmake --build build --parallel $(nproc)
```

`$(nproc)` automatically uses all your CPU cores to build faster. This will take **10–30 minutes** depending on your laptop.

While it builds, grab a coffee — when it finishes you'll see something like `[100%] Built target tt_metal`.

### Step 5.6 — Set required environment variables

tt-metal needs to know where it lives. Add these to your `.bashrc`:

```bash
cat >> ~/.bashrc << 'EOF'

# Tenstorrent tt-metal environment
export TT_METAL_HOME=$HOME/tt-metal
export PYTHONPATH=$HOME/tt-metal:$HOME/tt-metal/ttnn:$PYTHONPATH
export LD_LIBRARY_PATH=$HOME/tt-metal/build/lib:$LD_LIBRARY_PATH
EOF
source ~/.bashrc
```

### Step 5.7 — Verify the build

```bash
python3 -c "import ttnn; print('TTNN imported successfully')"
```

If this prints the success message — your build works. ✅

If it throws an error like `No module named ttnn`, check that your `PYTHONPATH` is set:

```bash
echo $PYTHONPATH
```

Should contain the tt-metal paths.

---

## Part 6 — Running tt-metal in Simulation Mode

Without physical hardware, tt-metal can run in a **CPU simulation mode** where operations execute on your laptop's CPU instead of a real Tensix core. This is perfect for learning the APIs.

### Step 6.1 — What simulation mode does

When you run tt-metal without a physical TT card:
- Device operations fall back to CPU execution
- You can still write kernels, define circular buffers, describe data movement
- The API surface is identical to what you'd use on real hardware
- You won't get real benchmark numbers, but you learn the programming model

### Step 6.2 — Run your first tt-metal example

Look at the programming examples:

```bash
ls ~/tt-metal/tt_metal/programming_examples/
```

Run the "hello world" example:

```bash
cd ~/tt-metal
source ~/tt-env/bin/activate
python3 tt_metal/programming_examples/hello_world_datatypes_example/hello_world_datatypes.py
```

If it runs without crashing — even if it prints "no device found" — that's expected. The key is it ran.

### Step 6.3 — Run TTNN in CPU mode

TTNN operations can fall back to CPU. Try this:

```bash
python3 - << 'EOF'
import torch
import ttnn

# Create a simple tensor using PyTorch first
a = torch.randn(32, 32)
b = torch.randn(32, 32)

# Do matrix multiply in PyTorch (as reference)
result_torch = torch.mm(a, b)
print(f"PyTorch matmul output shape: {result_torch.shape}")
print(f"PyTorch result (first row): {result_torch[0, :4]}")

print("\nTTNN imported OK. On real hardware, this would run on Tensix cores.")
print("In simulation (no TT card), use PyTorch as the execution backend.")
EOF
```

---

## Part 7 — QEMU RISC-V: Understand Tensix Internals

This is the unique part of this guide. Each Tensix core contains 5 RISC-V processors (BRISC, NCRISC, TRISC0, TRISC1, TRISC2). We use QEMU to actually run RISC-V code so you feel what those processors do.

### Step 7.1 — Write a simple RISC-V program that mimics BRISC behavior

BRISC's job: read data from memory (GDDR6 via NoC) into the local SRAM circular buffer.

We'll simulate this concept with a real RISC-V C program:

```bash
mkdir -p ~/tenstorrent-learning/labs/risc-v-sim
cd ~/tenstorrent-learning/labs/risc-v-sim
```

Create the file:

```bash
cat > brisc_sim.c << 'EOF'
/*
 * brisc_sim.c
 * Simulates what the BRISC (Builder RISC-V) core does inside a Tensix:
 * - It reads a block of data (weights/activations) from a "source" (GDDR6)
 * - It writes that data into the Circular Buffer (CB) in local SRAM
 * - It signals TRISC that new data is ready
 *
 * In real hardware: BRISC issues NoC read commands.
 * Here: we simulate with C arrays.
 */

#include <stdio.h>
#include <stdint.h>
#include <string.h>

/* === Simulated hardware resources === */

/* GDDR6 "off-chip" memory - imagine this is 32GB far away */
#define GDDR6_SIZE 1024
static float gddr6_memory[GDDR6_SIZE];

/* Local SRAM Circular Buffer - this is the 1.5MB per-Tensix SRAM */
/* Circular buffers have slots; each slot holds one "tile" (32x32 = 1024 values) */
#define TILE_SIZE    1024        /* 32x32 tile */
#define CB_NUM_SLOTS 4           /* circular buffer depth */
#define CB_SIZE      (TILE_SIZE * CB_NUM_SLOTS)
static float cb_sram[CB_SIZE];  /* This is in LOCAL SRAM - fast access */

/* CB read/write pointers */
static int cb_write_ptr = 0;    /* BRISC writes here */
static int cb_read_ptr  = 0;    /* TRISC reads from here */
static int cb_fill_count = 0;   /* how many slots are filled */

/* === Circular Buffer operations === */

/* BRISC calls this: "I have a tile ready in SRAM, advance write pointer" */
void cb_push_back(int num_tiles) {
    cb_write_ptr = (cb_write_ptr + num_tiles) % CB_NUM_SLOTS;
    cb_fill_count += num_tiles;
    printf("[BRISC] CB push: write_ptr=%d, fill_count=%d\n", cb_write_ptr, cb_fill_count);
}

/* TRISC calls this: "I'm done with a tile, advance read pointer" */
void cb_pop_front(int num_tiles) {
    cb_read_ptr = (cb_read_ptr + num_tiles) % CB_NUM_SLOTS;
    cb_fill_count -= num_tiles;
    printf("[TRISC] CB pop: read_ptr=%d, fill_count=%d\n", cb_read_ptr, cb_fill_count);
}

int cb_has_data(void) {
    return cb_fill_count > 0;
}

int cb_has_space(void) {
    return cb_fill_count < CB_NUM_SLOTS;
}

/* === BRISC: Data Movement Core === */
/* BRISC reads tiles from GDDR6 and writes them into the CB (SRAM) */
void brisc_load_tile_from_gddr6(int tile_idx, int gddr6_offset) {
    /* In real hardware: BRISC issues a NoC read command to the GDDR6 controller.
     * The data travels: GDDR6 -> NoC -> BRISC -> CB (SRAM)
     * Here: we just memcpy from our fake gddr6 array */
    
    int cb_slot = tile_idx % CB_NUM_SLOTS;
    float* cb_dest = &cb_sram[cb_slot * TILE_SIZE];
    float* gddr6_src = &gddr6_memory[gddr6_offset];
    
    printf("[BRISC] Loading tile %d from GDDR6 offset %d into CB slot %d\n",
           tile_idx, gddr6_offset, cb_slot);
    
    memcpy(cb_dest, gddr6_src, TILE_SIZE * sizeof(float));
}

/* === TRISC: Compute Core === */
/* TRISC reads from CB (SRAM), sends to Matrix Engine, writes result */
float trisc_compute_tile_sum(int cb_slot) {
    /* In real hardware: TRISC unpacks tile from CB, routes to Matrix Engine.
     * Here: we just sum the values to prove we read the data. */
    float* tile_data = &cb_sram[cb_slot * TILE_SIZE];
    float sum = 0.0f;
    for (int i = 0; i < TILE_SIZE; i++) {
        sum += tile_data[i];
    }
    return sum;
}

int main(void) {
    printf("=== Tensix Core Simulation ===\n");
    printf("Demonstrating: BRISC data movement + TRISC compute\n\n");
    
    /* Initialize fake GDDR6 memory with known values */
    printf("[INIT] Filling fake GDDR6 memory with test data...\n");
    for (int i = 0; i < GDDR6_SIZE; i++) {
        gddr6_memory[i] = (float)i * 0.001f;
    }
    
    int num_tiles = 6; /* We'll process 6 tiles through the pipeline */
    
    printf("\n[START] Processing %d tiles through BRISC->CB->TRISC pipeline\n\n", num_tiles);
    
    /* Simulate the BRISC/TRISC interleaved execution:
     * - BRISC loads tiles from GDDR6 into the CB
     * - TRISC reads from CB and computes
     * - The CB acts as a double-buffer between them
     * This is SPATIAL PIPELINING in action! */
    
    int brisc_tile = 0;
    int trisc_tile = 0;
    
    while (trisc_tile < num_tiles) {
        /* BRISC: fill the CB while there's space and more tiles to load */
        while (cb_has_space() && brisc_tile < num_tiles) {
            brisc_load_tile_from_gddr6(brisc_tile, brisc_tile * TILE_SIZE);
            cb_push_back(1);
            brisc_tile++;
        }
        
        /* TRISC: process tiles that are ready */
        while (cb_has_data() && trisc_tile < num_tiles) {
            int slot = trisc_tile % CB_NUM_SLOTS;
            float result = trisc_compute_tile_sum(slot);
            printf("[TRISC] Tile %d sum = %.2f (tile processed by Matrix Engine)\n",
                   trisc_tile, result);
            cb_pop_front(1);
            trisc_tile++;
        }
    }
    
    printf("\n=== Pipeline complete. %d tiles processed. ===\n", num_tiles);
    printf("\nKey insight: BRISC and TRISC run concurrently on separate RISC-V cores.\n");
    printf("While TRISC computes tile N, BRISC is already loading tile N+1.\n");
    printf("The Circular Buffer (CB) is the handoff point in SRAM.\n");
    
    return 0;
}
EOF
```

### Step 7.2 — Cross-compile for RISC-V 64-bit

```bash
riscv64-linux-gnu-gcc -O2 -o brisc_sim brisc_sim.c
```

### Step 7.3 — Run the RISC-V binary on QEMU

```bash
qemu-riscv64 ./brisc_sim
```

You should see the BRISC loading tiles and TRISC processing them — all running as actual RISC-V machine code on QEMU!

### Step 7.4 — Inspect the RISC-V assembly (see what the CPU actually does)

```bash
riscv64-linux-gnu-objdump -d brisc_sim | head -80
```

This shows you the actual RISC-V assembly instructions. The Tensix RISC-V cores run instructions just like this.

### Step 7.5 — Save your work to git

```bash
cd ~/tenstorrent-learning
git add labs/risc-v-sim/
git commit -m "Lab: BRISC/TRISC simulation in RISC-V via QEMU"
```

---

## Part 8 — Lab 1: Memory Wall and Roofline (Phase 1)

This lab makes the theoretical concepts from Phase 1 of the mastery doc concrete. You'll compute the roofline model and see exactly why LLM inference is memory-bound.

### Step 8.1 — Create the lab

```bash
mkdir -p ~/tenstorrent-learning/labs/lab1-memory-roofline
cd ~/tenstorrent-learning/labs/lab1-memory-roofline
```

```bash
cat > roofline_analysis.py << 'EOF'
"""
Lab 1: Memory Wall and Roofline Model Analysis
Covers: Phase 1.1, 1.2, 1.3, 1.4 of the mastery doc

This script lets you feel the memory wall by:
1. Computing arithmetic intensity for different LLM operations
2. Plotting the Roofline model for H100 vs Blackhole
3. Showing why LLM decode is memory-bound
"""

import numpy as np
import matplotlib.pyplot as plt
import time

print("=" * 60)
print("Lab 1: Memory Wall & Roofline Model")
print("=" * 60)

# ============================================================
# Part A: Hardware Specs (from the mastery doc)
# ============================================================

hardware = {
    "H100 SXM (HBM3)": {
        "peak_flops_tflops": 989,      # FP16 TFLOP/s
        "peak_bw_GBs": 3350,           # GB/s
        "memory_GB": 80,
        "memory_type": "HBM3",
        "approx_price_usd": 30000,
        "color": "#76b900",            # NVIDIA green
    },
    "Blackhole (GDDR6)": {
        "peak_flops_tflops": 786,
        "peak_bw_GBs": 576,
        "memory_GB": 32,
        "memory_type": "GDDR6",
        "approx_price_usd": 5000,
        "color": "#e55b2b",            # Tenstorrent orange
    },
    "AMD MI300X (HBM3e)": {
        "peak_flops_tflops": 1307,
        "peak_bw_GBs": 5300,
        "memory_GB": 192,
        "memory_type": "HBM3e",
        "approx_price_usd": 15000,
        "color": "#ed1c24",            # AMD red
    },
}

print("\n--- Hardware Specs ---")
for name, hw in hardware.items():
    ridge = hw["peak_flops_tflops"] * 1000 / hw["peak_bw_GBs"]  # FLOP/byte
    print(f"\n{name}")
    print(f"  Peak compute:    {hw['peak_flops_tflops']} TFLOP/s (FP16)")
    print(f"  Peak bandwidth:  {hw['peak_bw_GBs']} GB/s")
    print(f"  Memory:          {hw['memory_GB']} GB {hw['memory_type']}")
    print(f"  Ridge point:     {ridge:.1f} FLOP/byte")
    print(f"  Price (approx):  ~${hw['approx_price_usd']:,}")

# ============================================================
# Part B: Arithmetic Intensity of LLM Operations
# ============================================================

print("\n\n--- Arithmetic Intensity of LLM Operations ---")
print("(Higher = more compute per byte = better hardware utilization)")

def matmul_arithmetic_intensity(M, N, K, bytes_per_element=2):
    """
    Matrix multiply: C = A @ B
    A is (M, K), B is (K, N), C is (M, N)
    FLOPs = 2 * M * N * K  (multiply-accumulate = 2 ops)
    Bytes = (M*K + K*N + M*N) * bytes_per_element
    """
    flops = 2 * M * N * K
    bytes_accessed = (M * K + K * N + M * N) * bytes_per_element
    return flops / bytes_accessed, flops, bytes_accessed

# LLM scenarios
scenarios = [
    # (name, M, N, K, description)
    ("Prefill: QKV proj (batch=32, seq=1024)", 32*1024, 4096, 4096, "High reuse, compute-bound"),
    ("Prefill: QKV proj (batch=1, seq=512)",   1*512,   4096, 4096, "Lower batch, still OK"),
    ("Decode: single token",                   1,       4096, 4096, "Loads full weight for 1 token!"),
    ("Small matmul (batch=8)",                 8,       512,  512,  "Common in small models"),
    ("Training: large batch",                  512,     4096, 4096, "Training is compute-bound"),
]

for name, M, N, K, note in scenarios:
    intensity, flops, bytes_acc = matmul_arithmetic_intensity(M, N, K)
    print(f"\n{name}")
    print(f"  Size: ({M}, {N}, {K})  |  Note: {note}")
    print(f"  FLOPs: {flops/1e9:.2f} GFLOPs")
    print(f"  Bytes accessed: {bytes_acc/1e9:.3f} GB")
    print(f"  Arithmetic intensity: {intensity:.1f} FLOP/byte")
    
    # Check which region of the roofline we're in
    for hw_name, hw in hardware.items():
        ridge = hw["peak_flops_tflops"] * 1000 / hw["peak_bw_GBs"]
        region = "MEMORY-BOUND ⚠️" if intensity < ridge else "compute-bound ✅"
        print(f"    → {hw_name}: {region} (ridge at {ridge:.0f} FLOP/byte)")

# ============================================================
# Part C: Plot the Roofline
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("Roofline Model: Where LLM Operations Fall", fontsize=14, fontweight='bold')

for ax_idx, (hw_name, hw) in enumerate(list(hardware.items())[:2]):
    ax = axes[ax_idx]
    
    peak_flops = hw["peak_flops_tflops"] * 1000  # Convert to GFLOP/s
    peak_bw = hw["peak_bw_GBs"]
    ridge = peak_flops / peak_bw  # FLOP/byte
    
    # Arithmetic intensity axis (x)
    x = np.logspace(-2, 4, 1000)  # 0.01 to 10000 FLOP/byte
    
    # Roofline: min of (bandwidth * intensity, peak_flops)
    roofline = np.minimum(peak_bw * x, peak_flops)
    
    ax.loglog(x, roofline, color=hw["color"], linewidth=3, label="Roofline")
    ax.axvline(x=ridge, color=hw["color"], linestyle='--', alpha=0.5, label=f"Ridge point ({ridge:.0f} FLOP/byte)")
    
    # Plot the LLM scenarios
    scenario_colors = ['red', 'orange', 'darkred', 'blue', 'green']
    for (name, M, N, K, note), color in zip(scenarios, scenario_colors):
        intensity, flops, bytes_acc = matmul_arithmetic_intensity(M, N, K)
        # Achievable perf = min(intensity * bw, peak_flops)
        achievable = min(intensity * peak_bw, peak_flops)
        ax.scatter(intensity, achievable, s=100, color=color, zorder=5)
        # Short label
        short_name = name.split(":")[0] if ":" in name else name[:20]
        ax.annotate(short_name, (intensity, achievable),
                   textcoords="offset points", xytext=(5, 5), fontsize=7)
    
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=11)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=11)
    ax.set_title(f"{hw_name}\n{hw['memory_GB']} GB {hw['memory_type']} | "
                 f"{hw['peak_flops_tflops']} TFLOP/s | ${hw['approx_price_usd']:,}",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig("roofline_model.png", dpi=150, bbox_inches='tight')
print(f"\n\nRoofline chart saved to: roofline_model.png")

# ============================================================
# Part D: Measure ACTUAL arithmetic intensity on your CPU
# (This makes the concept real — you're measuring it)
# ============================================================

print("\n\n--- Measuring ACTUAL arithmetic intensity on your CPU ---")

def measure_matmul(M, N, K):
    A = np.random.randn(M, K).astype(np.float32)
    B = np.random.randn(K, N).astype(np.float32)
    
    # Warmup
    _ = A @ B
    
    # Measure
    t_start = time.perf_counter()
    for _ in range(10):
        C = A @ B
    t_end = time.perf_counter()
    
    elapsed = (t_end - t_start) / 10  # per iteration
    flops = 2 * M * N * K
    gflops = flops / elapsed / 1e9
    return gflops, elapsed

print("\nRunning matrix multiplications on your CPU (NumPy/BLAS):")
test_cases = [
    (1, 4096, 4096, "Decode (1 token, huge weight)"),
    (32, 4096, 4096, "Small batch prefill"),
    (512, 4096, 4096, "Large batch prefill"),
]
for M, N, K, label in test_cases:
    gflops, time_s = measure_matmul(M, N, K)
    intensity, _, _ = matmul_arithmetic_intensity(M, N, K)
    print(f"\n{label} — ({M}x{N}x{K})")
    print(f"  Arithmetic intensity: {intensity:.1f} FLOP/byte")
    print(f"  CPU throughput:       {gflops:.1f} GFLOP/s")
    print(f"  Time per matmul:      {time_s*1000:.2f} ms")

print("\n\nKey takeaway:")
print("  'Decode' (M=1) has intensity ~1 FLOP/byte — that's 1000x below the ridge")
print("  point of H100. So H100's 3350 GB/s HBM barely helps — it's still bottlenecked")
print("  on bandwidth. This is exactly the memory wall problem from the mastery doc.")
EOF
```

### Step 8.2 — Run the lab

```bash
source ~/tt-env/bin/activate
python3 roofline_analysis.py
```

You'll see the analysis printed and a `roofline_model.png` chart saved. Open it from Windows Explorer (your WSL files are at `\\wsl$\Ubuntu\home\yourname\`).

### Step 8.3 — Save to git

```bash
cd ~/tenstorrent-learning
git add labs/lab1-memory-roofline/
git commit -m "Lab 1: Roofline model analysis with real CPU measurements"
```

---

## Part 9 — Lab 2: Your First Metalium Kernel (Phase 3 + 4)

This lab teaches you to write kernels in the Metalium style — the programming model from Phase 4.4 of the mastery doc.

### Step 9.1 — Study the real programming examples first

Read the actual tt-metal examples:

```bash
ls ~/tt-metal/tt_metal/programming_examples/
cat ~/tt-metal/tt_metal/programming_examples/hello_world_datatypes_example/hello_world_datatypes.py
```

### Step 9.2 — Write a commented kernel that explains the concepts

```bash
mkdir -p ~/tenstorrent-learning/labs/lab2-metalium-kernel
cd ~/tenstorrent-learning/labs/lab2-metalium-kernel
```

```bash
cat > kernel_anatomy.md << 'EOF'
# Anatomy of a Metalium Kernel

## What a kernel does
A kernel is a small program that runs on ONE Tensix core.
Each Tensix has 5 RISC-V processors. You write 3 separate programs:
1. **Data Movement IN** (runs on BRISC) - reads from GDDR6/NoC into CB
2. **Compute** (runs on TRISC) - reads from CB, does math, writes to CB
3. **Data Movement OUT** (runs on NCRISC) - reads from CB, writes to GDDR6/NoC

## The Circular Buffer (CB) - the key concept
- Lives in the Tensix's 1.5MB SRAM
- BRISC writes to it (pushes)
- TRISC reads from it (pops)
- The two programs run CONCURRENTLY - pipeline!

## Kernel file structure (real tt-metal code):

```
my_kernel/
├── kernels/
│   ├── dataflow/
│   │   ├── reader_kernel.cpp   # BRISC kernel (reads from DRAM)
│   │   └── writer_kernel.cpp   # NCRISC kernel (writes to DRAM)
│   └── compute/
│       └── eltwise_kernel.cpp  # TRISC kernel (does the math)
└── my_kernel.py                # Host code (Python, runs on your CPU)
```

## Host code responsibilities:
- Create a Device (the TT chip)
- Allocate buffers in GDDR6 (input/output data)
- Define Circular Buffers (SRAM slots)
- Compile and dispatch the 3 kernels
- Wait for completion
- Read results back
EOF
```

### Step 9.3 — Write a real Metalium-style host program

```bash
cat > metalium_study.py << 'EOF'
"""
Metalium API Study
This program shows the EXACT API calls used in tt-metal
and explains what each one does. Even without a TT card,
reading this teaches you the programming model.
"""

# In a real tt-metal program, you'd import:
# import ttnn
# But we'll study the API structure with annotations

print("=" * 60)
print("Metalium Programming Model - API Study")
print("=" * 60)

# ============================================================
# STEP 1: Create a device
# ============================================================
print("""
STEP 1: Open the device
-----------------------
In real Metalium code:

    device = ttnn.open_device(device_id=0)
    # device_id=0 means "first TT card in the system"
    # This initializes the hardware, loads firmware onto each RISC-V core,
    # sets up the dispatch queues.

Without hardware: device represents a CPU-simulation device.
""")

# ============================================================
# STEP 2: Create circular buffers
# ============================================================
print("""
STEP 2: Define Circular Buffers (the SRAM handoff zones)
---------------------------------------------------------
Each CB is a slot in the 1.5MB SRAM of one Tensix core.
BRISC writes to it, TRISC reads from it.

In real Metalium:

    # CB index 0 = input buffer (BRISC fills this)
    cb_input_config = ttnn.CircularBufferConfig(
        num_pages=2,              # 2 tiles in the buffer (double-buffering)
        page_size=tile_size_bytes, # one 32x32 tile
    ).set_page_size(CB_IN_ID, tile_size_bytes)

    cb_input = ttnn.CreateCircularBuffer(program, core, cb_input_config)

Why 2 pages (double-buffering)?
    While TRISC is computing on tile N (from slot 0),
    BRISC is loading tile N+1 into slot 1.
    Then TRISC moves to slot 1, BRISC fills slot 0 again.
    This hides the GDDR6 load latency!
""")

# ============================================================
# STEP 3: Define kernels
# ============================================================
print("""
STEP 3: Define the three kernel programs
-----------------------------------------
Each kernel is a C++ file that gets compiled to RISC-V machine code:

    # Reader kernel (runs on BRISC - data movement in)
    reader_kernel = ttnn.CreateKernel(
        program,
        "kernels/dataflow/reader_kernel.cpp",  # your C++ file
        core_spec,
        ttnn.DataMovementConfig(
            processor=ttnn.DataMovementProcessor.RISCV_0,  # BRISC
            noc=ttnn.NOC.RISCV_0_default,
        )
    )

    # Writer kernel (runs on NCRISC - data movement out)
    writer_kernel = ttnn.CreateKernel(
        program,
        "kernels/dataflow/writer_kernel.cpp",
        core_spec,
        ttnn.DataMovementConfig(
            processor=ttnn.DataMovementProcessor.RISCV_1,  # NCRISC
            noc=ttnn.NOC.RISCV_1_default,
        )
    )

    # Compute kernel (runs on TRISC0/1/2 - the math)
    compute_kernel = ttnn.CreateKernel(
        program,
        "kernels/compute/eltwise_kernel.cpp",
        core_spec,
        ttnn.ComputeConfig(
            math_fidelity=ttnn.MathFidelity.HiFi4,
            fp32_dest_acc_en=False,
            math_approx_mode=False,
        )
    )
""")

# ============================================================
# STEP 4: Inside a reader kernel (BRISC code, written in C++)
# ============================================================
print("""
STEP 4: Inside the reader kernel (C++ runs on BRISC RISC-V core)
-----------------------------------------------------------------
File: kernels/dataflow/reader_kernel.cpp

void kernel_main() {
    // Get runtime args passed from host
    uint32_t src_addr = get_arg_val<uint32_t>(0);  // GDDR6 address
    uint32_t num_tiles = get_arg_val<uint32_t>(1); // how many tiles to process
    
    // Set up the GDDR6 source
    InterleavedAddrGenFast<true> src_addr_gen = {
        .bank_base_address = src_addr,
        .page_size = TILE_SIZE,
    };
    
    // Process tiles one by one
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // WAIT until CB slot is available (TRISC must have consumed previous tile)
        cb_reserve_back(cb_id, 1);  // blocks until space available
        
        // Get pointer to the writable slot in SRAM
        uint32_t l1_write_ptr = get_write_ptr(cb_id);
        
        // Issue NoC READ: copies data from GDDR6 to SRAM
        // This is the KEY operation: GDDR6 -> NoC -> local SRAM
        noc_async_read_tile(i, src_addr_gen, l1_write_ptr);
        noc_async_read_barrier();  // wait for DMA to finish
        
        // Signal TRISC: "data is ready in this slot"
        cb_push_back(cb_id, 1);
    }
}

KEY INSIGHT: cb_reserve_back() blocks until TRISC has consumed.
             cb_push_back() signals TRISC that new data is ready.
             These are the handshake primitives for the pipeline.
""")

# ============================================================
# STEP 5: Inside a compute kernel (TRISC code)
# ============================================================
print("""
STEP 5: Inside the compute kernel (C++ runs on TRISC RISC-V cores)
-------------------------------------------------------------------
File: kernels/compute/eltwise_kernel.cpp

void MAIN {
    uint32_t num_tiles = get_compile_time_arg_val(0);
    
    unary_op_init_common(cb_in, cb_out);  // initialize Math Engine
    
    for (uint32_t i = 0; i < num_tiles; ++i) {
        // WAIT for BRISC to deliver a tile
        cb_wait_front(cb_in, 1);  // blocks until BRISC pushes
        
        // Acquire the Math Engine's compute tile
        acquire_dst(tt::DstMode::Half);
        
        // UNPACK: move tile from CB (SRAM) to Math Engine's registers
        unpack_tilize_init(cb_in, 1);
        unpack_tilize_block(cb_in, 1);
        
        // COMPUTE: run the math operation (here: gelu activation)
        gelu_tile_init();
        gelu_tile(0);
        
        // PACK: move result from Math Engine registers back to output CB
        pack_tile(0, cb_out);
        
        // Release Math Engine
        release_dst(tt::DstMode::Half);
        
        // Signal reader that the input slot is free
        cb_pop_front(cb_in, 1);
        // Signal writer that output slot is filled
        cb_push_back(cb_out, 1);
    }
}

KEY INSIGHT: Three sub-operations (UNPACK -> COMPUTE -> PACK)
             map to three TRISC processors (TRISC0, TRISC1, TRISC2)
             These run concurrently within one Tensix!
""")

print("""
SUMMARY: The data flow through one Tensix core:
================================================

GDDR6/NoC
    |
    | noc_async_read_tile()
    ↓
[Circular Buffer IN - SRAM]   ← BRISC writes here
    |
    | unpack_tilize_block()
    ↓
[Math Engine registers]        ← TRISC0 unpacks here
    |
    | gelu_tile() / matmul_tile()
    ↓
[Destination registers]        ← TRISC1 runs math here
    |
    | pack_tile()
    ↓
[Circular Buffer OUT - SRAM]  ← TRISC2 packs here
    |
    | noc_async_write_tile()
    ↓
GDDR6/NoC (next core or output)
""")
EOF
```

### Step 9.4 — Run it and study the output

```bash
python3 metalium_study.py
```

### Step 9.5 — Read a real programming example from the repo

```bash
# Read the actual hello world example from tt-metal
cat ~/tt-metal/tt_metal/programming_examples/hello_world_datatypes_example/hello_world_datatypes.py
```

Look for: `CreateCircularBuffer`, `CreateKernel`, `SetRuntimeArgs`, `LaunchProgram`. These match what you just studied.

---

## Part 10 — Lab 3: TTNN Tensor Operations (Phase 4)

TTNN is the Python tensor library — the layer above Metalium. This is what most users interact with.

### Step 10.1 — Study TTNN data types and layouts

```bash
mkdir -p ~/tenstorrent-learning/labs/lab3-ttnn
cd ~/tenstorrent-learning/labs/lab3-ttnn
```

```bash
cat > ttnn_study.py << 'EOF'
"""
Lab 3: TTNN Tensor Operations Study
Covers: Phase 4.3 — TTNN tensor library

TTNN wraps Metalium kernels in a PyTorch-like API.
Key concepts:
- Tensor layout: ROW_MAJOR vs TILE (32x32 tiles)
- Data format: BFP8, BF16, FP32
- Device memory: DRAM vs L1 (SRAM)
- Multi-core distribution
"""

import torch
import numpy as np

print("=" * 60)
print("Lab 3: TTNN Tensor Operations")
print("=" * 60)

# ============================================================
# Part A: Understand tile layout
# ============================================================
print("""
A. TILE LAYOUT — Why Tenstorrent uses 32x32 tiles
==================================================

Row-major (normal layout, used by CPU/GPU):
Data is stored row by row in memory.
For a 4x4 matrix:

  [0, 1, 2, 3,   4, 5, 6, 7,   8, 9, 10, 11,   12, 13, 14, 15]
   ├────row 0────┤ ├────row 1───┤ ├──────row 2──────┤ ├───row 3──┤

Tile layout (Tenstorrent):
Data is stored as 32x32 blocks. Each tile is 1024 values.

  [tile_0_row0, tile_0_row1, ..., tile_0_row31,  ← tile 0 of the matrix
   tile_1_row0, tile_1_row1, ..., tile_1_row31,  ← tile 1
   ...]

Why tiles? The Matrix Engine in each Tensix does EXACTLY one
32x32 matrix multiply per cycle. Tile layout means the data
is already arranged for the hardware to consume directly —
no reshuffling needed.
""")

# Demonstrate tile layout effect on memory access patterns
print("B. Memory access pattern comparison:")

size = 128  # 128x128 matrix (4x4 tiles of 32x32)
A = np.random.randn(size, size).astype(np.float32)

print(f"\nMatrix size: {size}x{size} = {size*size} elements")
print(f"Tile size: 32x32 = 1024 elements per tile")
print(f"Number of tiles: {(size//32)**2} tiles (4x4 grid of tiles)")
print(f"Memory for one tile: {1024 * 4} bytes = 4 KB (FP32)")
print(f"Memory for one tile: {1024 * 2} bytes = 2 KB (BF16)")
print(f"Memory for one tile: {1024 * 1} bytes = 1 KB (BFP8)")

# ============================================================
# Part B: Data format comparison (BFP8 vs BF16 vs FP32)
# ============================================================
print("""
C. Data Formats — What BFP8, BF16, FP32 mean
=============================================
""")

formats = {
    "FP32": {"bits": 32, "mantissa": 23, "exponent": 8, "description": "Full precision, max accuracy"},
    "BF16": {"bits": 16, "mantissa": 7, "exponent": 8, "description": "Same exp as FP32, halved mantissa"},
    "FP16": {"bits": 16, "mantissa": 10, "exponent": 5, "description": "IEEE half-precision"},
    "BFP8": {"bits": 8,  "mantissa": 2, "exponent": 8, "description": "Tenstorrent's block float 8 (shared exp per tile)"},
    "BFP4": {"bits": 4,  "mantissa": 0, "exponent": 4, "description": "Most aggressive compression"},
}

tile_values = 1024  # 32x32 tile

print(f"{'Format':<8} {'Bits':<6} {'Bytes/tile':<12} {'Tiles/MB':<12} {'Notes'}")
print("-" * 70)
for name, fmt in formats.items():
    bytes_per_tile = tile_values * fmt["bits"] // 8
    tiles_per_mb = 1024 * 1024 // bytes_per_tile
    print(f"{name:<8} {fmt['bits']:<6} {bytes_per_tile:<12,} {tiles_per_mb:<12} {fmt['description']}")

print("""
BFP8 special note:
  BFP8 stores ONE shared exponent per 32x32 tile, then 8-bit mantissa per value.
  This means: if one value in the tile is huge, all values share that exponent scale.
  This is why BFP8 works well for weight matrices (values are similar in magnitude
  within a tile) but would be risky for activation layers (wider range).
""")

# ============================================================
# Part C: Multi-core distribution
# ============================================================
print("""
D. Multi-core tensor distribution on the NoC grid
=================================================
When you run a large matmul (e.g., 4096x4096),
TTNN splits it across multiple Tensix cores automatically.

Example: Splitting a 128x128 matrix across a 4x4 core grid

    Core (0,0): rows 0-31    Core (0,1): rows 0-31
    Core (1,0): rows 32-63   Core (1,1): rows 32-63
    Core (2,0): rows 64-95   Core (2,1): rows 64-95
    Core (3,0): rows 96-127  Core (3,1): rows 96-127

Each core gets its own slice to compute — no communication needed
for embarrassingly parallel ops. For matmul, partial results
are combined over the NoC.
""")

# ============================================================
# Part D: TTNN-style operations using PyTorch (as proxy)
# ============================================================
print("""
E. TTNN-style operations (shown in PyTorch as fallback)
=======================================================
""")

# In real TTNN with hardware, you'd write:
#   device = ttnn.open_device(0)
#   t = ttnn.from_torch(tensor, device=device, layout=ttnn.TILE_LAYOUT)
#   result = ttnn.matmul(t, t)
# 
# Here we show the same operations in PyTorch:

# Create tiles-aligned tensors (multiples of 32)
batch_size = 32   # one LLM decode batch
seq_len = 32      # minimum tile size
hidden = 4096     # typical LLM hidden dimension
ffn_hidden = hidden * 4

print(f"LLM Decode Simulation:")
print(f"  batch_size = {batch_size}")
print(f"  seq_len = {seq_len}")
print(f"  hidden = {hidden}")
print(f"  ffn_hidden = {ffn_hidden}")

# Simulate one transformer FFN layer
x = torch.randn(batch_size, seq_len, hidden)  # input activations
W1 = torch.randn(hidden, ffn_hidden)           # FFN weight 1
W2 = torch.randn(ffn_hidden, hidden)           # FFN weight 2

print(f"\nFFN Layer 1 matmul:")
print(f"  input:  {list(x.shape)}")
print(f"  weight: {list(W1.shape)}")

t0 = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
import time
start = time.perf_counter()

# FFN computation
h = x @ W1                    # [batch, seq, ffn_hidden]
h = torch.nn.functional.gelu(h)  # activation
out = h @ W2                  # [batch, seq, hidden]

end = time.perf_counter()

print(f"  output: {list(out.shape)}")

# Count FLOPs
flops_layer1 = 2 * batch_size * seq_len * hidden * ffn_hidden
flops_layer2 = 2 * batch_size * seq_len * ffn_hidden * hidden
total_flops = flops_layer1 + flops_layer2

bytes_W1 = hidden * ffn_hidden * 2  # BF16
bytes_W2 = ffn_hidden * hidden * 2

print(f"\n  FLOPs: {total_flops/1e9:.2f} GFLOPs")
print(f"  Weights loaded: {(bytes_W1+bytes_W2)/1e9:.3f} GB (BF16)")
print(f"  Arithmetic intensity: {total_flops/(bytes_W1+bytes_W2):.1f} FLOP/byte")
print(f"  CPU time: {(end-start)*1000:.2f} ms")

print("""
In real TTNN on hardware, the sequence would be:
  ttnn.linear(x, W1) → runs reader_kernel (GDDR6→CB) + compute_kernel (TRISC) + writer_kernel
  ttnn.gelu(h)        → element-wise SFPU operation (TRISC directly)
  ttnn.linear(h, W2)  → same as W1 step
  
  The weight matrices (W1, W2) are loaded from GDDR6 into per-core SRAM once,
  then re-used if batch size > 1.
""")
EOF
```

```bash
python3 ttnn_study.py
```

---

## Part 11 — Lab 4: Dataflow and Circular Buffers (Phase 3)

This lab makes the dataflow execution model concrete.

```bash
mkdir -p ~/tenstorrent-learning/labs/lab4-dataflow
cd ~/tenstorrent-learning/labs/lab4-dataflow
```

```bash
cat > spatial_pipeline_sim.py << 'EOF'
"""
Lab 4: Spatial Pipelining Simulation
Covers: Phase 3.1, 3.2, 3.3 from the mastery doc

Simulates a 3-stage spatial pipeline:
Stage 0 (Core 0): Load weights from GDDR6 → do QKV projection
Stage 1 (Core 1): Receive activations → do Attention
Stage 2 (Core 2): Receive activations → do FFN

In real Tenstorrent:
- Each stage runs on different Tensix cores
- Data flows over the NoC between cores (not through GDDR6!)
- All stages run SIMULTANEOUSLY (pipelined)
"""

import threading
import queue
import time
import random
import numpy as np

print("=" * 60)
print("Lab 4: Spatial Pipelining")
print("=" * 60)

# ============================================================
# A. What spatial pipelining IS
# ============================================================
print("""
A. Spatial vs Temporal Pipelining
===================================

TEMPORAL (GPU approach):
  Token 1:  [QKV]→[ATT]→[FFN]  ... done, then
  Token 2:             [QKV]→[ATT]→[FFN]  ... done, then
  Token 3:                         [QKV]→[ATT]→[FFN]
  
  Each token goes through all stages sequentially.
  Cores sit idle when not in their stage.

SPATIAL (Tenstorrent approach):
  Core 0: [QKV-T1] → [QKV-T2] → [QKV-T3] → ...  always busy
  Core 1:         → [ATT-T1] → [ATT-T2] → [ATT-T3] → ...  always busy
  Core 2:                  → [FFN-T1] → [FFN-T2] → [FFN-T3] → ...  always busy
  
  All cores busy simultaneously! Data flows like water through pipes.
""")

# ============================================================
# B. Simulate with Python threads (each thread = one Tensix core)
# ============================================================

def make_noc_channel(name, maxsize=2):
    """Simulates the NoC connection between two cores.
    maxsize=2 simulates the double-buffer in the circular buffer (CB)."""
    q = queue.Queue(maxsize=maxsize)
    q.name = name
    return q

NUM_TOKENS = 8

# NoC channels between cores (simulate CB SRAM slots)
gddr6_to_core0 = make_noc_channel("GDDR6→Core0", maxsize=2)
core0_to_core1 = make_noc_channel("Core0→Core1", maxsize=2)
core1_to_core2 = make_noc_channel("Core1→Core2", maxsize=2)
output_channel = queue.Queue()

# Timing log
timing_log = []
timing_lock = threading.Lock()

def log_event(stage, token, event, t):
    with timing_lock:
        timing_log.append((stage, token, event, t))

def stage0_qkv_projection(token_id, weights_in_sram):
    """
    Core 0: BRISC loads weights from GDDR6.
    TRISC does Q, K, V matrix multiplications.
    NCRISC sends result over NoC to Core 1.
    
    Simulated with random delay.
    """
    t0 = time.perf_counter()
    log_event("QKV", token_id, "start", t0)
    
    # Simulate weight fetch from GDDR6 (slow — 80ns latency)
    time.sleep(0.01)  # 10ms simulated GDDR6 access
    
    # Simulate Q, K, V matrix multiply
    time.sleep(0.02)  # 20ms simulated compute
    
    # Result: activation tensor (simulated as array)
    qkv_activation = np.random.randn(4096).astype(np.float32)
    
    t1 = time.perf_counter()
    log_event("QKV", token_id, "end", t1)
    return qkv_activation

def stage1_attention(token_id, qkv_activation, kv_cache):
    """
    Core 1: Receive QKV activations over NoC.
    Compute Q @ K^T, softmax, @ V.
    Send result to Core 2.
    """
    t0 = time.perf_counter()
    log_event("ATT", token_id, "start", t0)
    
    # Simulate attention computation
    time.sleep(0.015)  # attention is cheaper (activations stay in SRAM)
    
    att_output = np.random.randn(4096).astype(np.float32)
    
    t1 = time.perf_counter()
    log_event("ATT", token_id, "end", t1)
    return att_output

def stage2_ffn(token_id, att_output):
    """
    Core 2: Receive attention output over NoC.
    Compute FFN (2 matmuls + GELU).
    Write final output.
    """
    t0 = time.perf_counter()
    log_event("FFN", token_id, "start", t0)
    
    # Simulate FFN computation (bigger matrices)
    time.sleep(0.03)
    
    ffn_output = np.random.randn(4096).astype(np.float32)
    
    t1 = time.perf_counter()
    log_event("FFN", token_id, "end", t1)
    return ffn_output

# Worker functions (each runs on a separate thread = separate Tensix core)

def worker_core0():
    """Core 0: QKV stage"""
    weights = np.random.randn(4096, 4096).astype(np.float32)  # fake weights in SRAM
    for token_id in range(NUM_TOKENS):
        result = stage0_qkv_projection(token_id, weights)
        core0_to_core1.put((token_id, result))  # push over "NoC"
    core0_to_core1.put(None)  # sentinel

def worker_core1():
    """Core 1: Attention stage"""
    kv_cache = {}
    while True:
        item = core0_to_core1.get()  # wait for data from Core 0 over "NoC"
        if item is None:
            core1_to_core2.put(None)
            break
        token_id, qkv = item
        result = stage1_attention(token_id, qkv, kv_cache)
        core1_to_core2.put((token_id, result))

def worker_core2():
    """Core 2: FFN stage"""
    while True:
        item = core1_to_core2.get()  # wait for data from Core 1 over "NoC"
        if item is None:
            break
        token_id, att_out = item
        result = stage2_ffn(token_id, att_out)
        output_channel.put((token_id, result))

print("\nB. Running spatial pipeline (3 cores, 8 tokens)...")
print("   Each core runs as a separate thread (simulating a Tensix core)\n")

start_time = time.perf_counter()

# Start all cores simultaneously (just like real hardware)
threads = [
    threading.Thread(target=worker_core0, name="Core-0-QKV"),
    threading.Thread(target=worker_core1, name="Core-1-ATT"),
    threading.Thread(target=worker_core2, name="Core-2-FFN"),
]
for t in threads:
    t.start()
for t in threads:
    t.join()

end_time = time.perf_counter()
total_pipeline_time = end_time - start_time

# ============================================================
# C. Compare to sequential execution
# ============================================================
print("C. Comparing pipeline vs sequential execution:")

# Sequential: one token at a time, each stage must finish before next
sequential_time = NUM_TOKENS * (0.01 + 0.02 + 0.015 + 0.03)

print(f"\n   Sequential (GPU temporal approach):")
print(f"   {NUM_TOKENS} tokens × (QKV + ATT + FFN) = {sequential_time:.3f}s estimated")

print(f"\n   Pipeline (Tenstorrent spatial approach):")
print(f"   All cores running simultaneously = {total_pipeline_time:.3f}s actual")

speedup = sequential_time / total_pipeline_time
print(f"\n   Speedup: {speedup:.1f}x")
print(f"   (Theoretical max for 3 stages = 3x)")

# Show the timeline
print(f"\nD. Timeline visualization (showing pipeline overlap):")
print("   Each row = one Tensix core, each column = time")
print()

# Sort events for display
events_by_token = {}
for stage, token, event, t in sorted(timing_log, key=lambda x: x[3]):
    key = (stage, token)
    if key not in events_by_token:
        events_by_token[key] = {}
    events_by_token[key][event] = t - start_time

print(f"   {'Stage':<8} {'Token':<8} {'Start':>8} {'End':>8} {'Duration':>10}")
print("   " + "-" * 45)
for (stage, token), times in sorted(events_by_token.items()):
    start = times.get('start', 0)
    end = times.get('end', 0)
    print(f"   {stage:<8} {token:<8} {start:>8.3f}s {end:>8.3f}s {(end-start):>8.3f}s")

print(f"""
E. Key insight from the timeline:
   Notice how all 3 stages overlap in time (different tokens).
   While Core-2 processes token 0 (FFN),
   Core-1 is doing attention on token 1,
   and Core-0 is doing QKV on token 2.
   
   This is spatial pipelining: each Tensix core is ALWAYS busy.
   No idle waiting. This is why Tenstorrent is efficient at inference.
""")
EOF
```

```bash
python3 spatial_pipeline_sim.py
```

---

## Part 12 — Lab 5: tt-forge with a PyTorch Model (Phase 4)

tt-forge is the ML compiler that converts PyTorch models to run on Tenstorrent hardware.

### Step 12.1 — Clone tt-forge

```bash
cd ~
git clone https://github.com/tenstorrent/tt-forge.git
cd tt-forge
```

### Step 12.2 — Install tt-forge

```bash
source ~/tt-env/bin/activate
pip install -e ".[dev]" 2>/dev/null || pip install -r requirements.txt 2>/dev/null || echo "Check tt-forge README for install steps"
```

### Step 12.3 — Study the MLIR compilation pipeline

```bash
mkdir -p ~/tenstorrent-learning/labs/lab5-ttforge
cd ~/tenstorrent-learning/labs/lab5-ttforge
```

```bash
cat > ttforge_study.py << 'EOF'
"""
Lab 5: tt-forge Compiler Study
Covers: Phase 4.2 — TT-Forge ML Compiler

tt-forge takes your PyTorch model and compiles it
through several MLIR passes:

PyTorch model
    ↓ torch.compile() backend
TOSA dialect (generic tensor ops)
    ↓ lower_to_ttir pass
TTIR dialect (TT intermediate representation)
    ↓ lower_to_ttnn pass
TTNN operations (maps to TTNN library calls)
    ↓ codegen
Kernel dispatch (Metalium programs)
"""

import torch
import torch.nn as nn

print("=" * 60)
print("Lab 5: tt-forge ML Compiler Pipeline Study")
print("=" * 60)

# ============================================================
# A. What torch.compile does for TT
# ============================================================
print("""
A. How tt-forge hooks into PyTorch
====================================
tt-forge registers itself as a torch.compile backend:

    import forge

    model = MyModel()
    
    # This line is ALL the user writes.
    # tt-forge takes over the compilation.
    compiled_model = torch.compile(model, backend="tt")
    
    output = compiled_model(input)

Behind the scenes, tt-forge:
  1. Captures the computation graph via torch.fx
  2. Lowers to TOSA (generic ML ops in MLIR)
  3. Applies TTIR passes (Tenstorrent-specific optimizations)
  4. Lowers to TTNN calls
  5. Dispatches to Metalium kernels on hardware
""")

# ============================================================
# B. Define a simple model and trace its operations
# ============================================================

class SimpleLLMBlock(nn.Module):
    """
    Represents ONE transformer decoder layer.
    This is what tt-forge would compile and run on a Tensix grid.
    """
    def __init__(self, hidden_dim=256, num_heads=4, ffn_dim=1024):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Attention
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # FFN
        self.ffn1 = nn.Linear(hidden_dim, ffn_dim, bias=False)
        self.ffn2 = nn.Linear(ffn_dim, hidden_dim, bias=False)
        
        # Norms
        self.attn_norm = nn.LayerNorm(hidden_dim)
        self.ffn_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x):
        # Attention
        residual = x
        x = self.attn_norm(x)
        B, S, H = x.shape
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        
        # Scaled dot-product attention
        Q = Q.view(B, S, self.num_heads, H // self.num_heads).transpose(1, 2)
        K = K.view(B, S, self.num_heads, H // self.num_heads).transpose(1, 2)
        V = V.view(B, S, self.num_heads, H // self.num_heads).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (H // self.num_heads) ** 0.5
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, S, H)
        x = residual + self.out_proj(out)
        
        # FFN
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn1(x)
        x = torch.nn.functional.gelu(x)
        x = self.ffn2(x)
        x = residual + x
        
        return x

model = SimpleLLMBlock(hidden_dim=256, num_heads=4, ffn_dim=1024)
model.eval()

print("B. Model defined: SimpleLLMBlock")
print(f"   Parameters: {sum(p.numel() for p in model.parameters()):,}")
print(f"   Weight bytes (FP32): {sum(p.numel() for p in model.parameters()) * 4:,}")
print(f"   Weight bytes (BF16): {sum(p.numel() for p in model.parameters()) * 2:,}")

# ============================================================
# C. Trace the ops using torch.fx
# ============================================================
print("\nC. Tracing with torch.fx (what tt-forge sees):")
x = torch.randn(1, 32, 256)  # batch=1, seq=32, hidden=256

# Symbolic trace (what the compiler does)
try:
    from torch.fx import symbolic_trace
    traced = symbolic_trace(model)
    
    ops = []
    for node in traced.graph.nodes:
        if node.op not in ('placeholder', 'output', 'get_attr'):
            ops.append(f"    {node.op}: {node.target}")
    
    print(f"\n   Found {len(ops)} operations in the graph:")
    for op in ops[:20]:  # show first 20
        print(op)
    if len(ops) > 20:
        print(f"   ... and {len(ops)-20} more ops")
        
    print("""
   Each of these ops becomes an MLIR operation in TOSA dialect.
   Then tt-forge's passes lower each one to a TTNN call.
   Then TTNN dispatches Metalium kernels onto the Tensix grid.
""")
except Exception as e:
    print(f"   (Symbolic trace not available for this model: {e})")

# ============================================================
# D. Measure model weight size vs memory hierarchy
# ============================================================
print("\nD. Weight size analysis for different model scales:")

models = [
    ("Tiny (our example)",   256,  256*4,  1,  4),
    ("Small (GPT-2 like)",   768,  768*4,  12, 12),
    ("Medium (7B LLM like)", 4096, 4096*4, 32, 32),
    ("Large (70B LLM like)", 8192, 8192*4, 64, 64),
]

print(f"\n{'Model':<28} {'Params':<12} {'BF16 Size':<12} {'Fits in?'}")
print("-" * 70)

blackhole_gddr6_gb = 32
blackhole_sram_mb = 180

for name, hidden, ffn, layers, heads in models:
    # Rough parameter count
    params_per_layer = (
        3 * hidden * hidden +  # QKV
        hidden * hidden +      # out_proj
        hidden * ffn +         # ffn1
        ffn * hidden           # ffn2
    )
    total_params = params_per_layer * layers
    bf16_gb = total_params * 2 / 1e9
    
    if bf16_gb < blackhole_sram_mb / 1024:
        fits = f"Tensix SRAM ({blackhole_sram_mb}MB)"
    elif bf16_gb < blackhole_gddr6_gb:
        fits = f"GDDR6 ({blackhole_gddr6_gb}GB)"
    else:
        fits = f"Too large! Need multi-chip"
    
    print(f"{name:<28} {total_params:>10,} {bf16_gb:>8.1f} GB   {fits}")

print(f"""
Key insight:
  Tiny models can potentially fit in the distributed SRAM (180MB).
  If all weights stay in SRAM — GDDR6 is never accessed during decode!
  That's the ultimate Tenstorrent scenario: infinite SRAM bandwidth.
  
  Most real LLMs need GDDR6 (32GB on Blackhole), and large ones need multi-chip.
""")
EOF
```

```bash
python3 ttforge_study.py
```

---

## Part 13 — Lab 6: Exploring the Repo Structure (Phase 4 + 6)

This lab teaches you to navigate the tt-metal codebase — a skill you need to read primary sources (Phase 6.4 of the mastery doc).

### Step 13.1 — Interactive repo explorer

```bash
mkdir -p ~/tenstorrent-learning/labs/lab6-repo-explorer
cd ~/tenstorrent-learning/labs/lab6-repo-explorer
```

```bash
cat > explore_ttmetal.sh << 'SCRIPT'
#!/bin/bash
# Guided tour of the tt-metal repository
# Run: bash explore_ttmetal.sh

TT_HOME=$HOME/tt-metal

echo "=============================================="
echo "tt-metal Repository Explorer"
echo "=============================================="

echo ""
echo "1. RISC-V Firmware (the code that runs on BRISC/NCRISC/TRISC)"
echo "   Location: tt_metal/hw/firmware/src/"
ls $TT_HOME/tt_metal/hw/firmware/src/*.cc 2>/dev/null | head -10
echo ""
echo "   → brisc.cc is the main loop for the BRISC (data movement in) processor"
echo "   → ncrisc.cc is the main loop for the NCRISC (data movement out) processor"
echo "   → trisc*.cc are the compute processor loops"

echo ""
echo "2. Chip-specific compute kernels"
echo "   Location: tt_metal/hw/ckernels/"
ls $TT_HOME/tt_metal/hw/ckernels/ 2>/dev/null
echo "   → Each subdirectory has kernels tuned for that chip generation"

echo ""
echo "3. Programming Examples (best place to start learning)"
echo "   Location: tt_metal/programming_examples/"
ls $TT_HOME/tt_metal/programming_examples/ 2>/dev/null
echo ""
echo "   Try reading these in order:"
echo "   1. hello_world_datatypes_example"
echo "   2. loopback"
echo "   3. eltwise_binary"
echo "   4. matmul_single_core"
echo "   5. matmul_multi_core"

echo ""
echo "4. TTNN Operations"
echo "   Location: ttnn/cpp/ttnn/operations/"
ls $TT_HOME/ttnn/cpp/ttnn/operations/ 2>/dev/null | head -20

echo ""
echo "5. tt-LLK (lowest level math kernels)"
echo "   This is what TRISC actually calls for matrix multiply:"
find $TT_HOME -name "llk_math_matmul*" 2>/dev/null | head -5

echo ""
echo "6. NoC API (what BRISC uses to read/write over the mesh)"
find $TT_HOME -name "dataflow_api.h" 2>/dev/null | head -3
echo "   This header defines: noc_async_read, noc_async_write, cb_push_back, cb_wait_front"

echo ""
echo "7. Tests (learn by example)"
echo "   tt_metal tests:"
ls $TT_HOME/tests/tt_metal/ 2>/dev/null | head -10

echo ""
echo "Done! Next steps:"
echo "  cat $TT_HOME/tt_metal/programming_examples/hello_world_datatypes_example/hello_world_datatypes.py"
echo "  cat $TT_HOME/tt_metal/hw/firmware/src/brisc.cc | head -100"
SCRIPT
chmod +x explore_ttmetal.sh
bash explore_ttmetal.sh
```

### Step 13.2 — Read real firmware code

```bash
# Read the actual BRISC firmware (this is the code that runs on Tensix RISC-V cores)
head -80 ~/tt-metal/tt_metal/hw/firmware/src/brisc.cc 2>/dev/null || \
  echo "File not at that path - try: find ~/tt-metal -name 'brisc.cc' 2>/dev/null"
```

```bash
# Find and read the dataflow API (the NoC commands BRISC uses)
find ~/tt-metal -name "dataflow_api.h" 2>/dev/null | head -1 | xargs head -100
```

---

## Part 14 — Git Workflow for Day-to-Day Learning

This is the git workflow you'll use every day as you go through the mastery doc.

### The daily learning loop

```bash
# Morning: start learning session
cd ~/tenstorrent-learning
source ~/tt-env/bin/activate

# Check what you were doing
git log --oneline -5

# Create a branch for today's topic
git checkout -b phase3-dataflow-notes

# ... work, experiment, write notes ...

# End of session: save everything
git add .
git commit -m "Phase 3: dataflow model notes and pipeline simulation"
git checkout main
git merge phase3-dataflow-notes
git push origin main
```

### When you want to experiment without breaking things

```bash
# Start a new branch for the experiment
git checkout -b experiment/quantum-buffer-sizing

# Do whatever you want
# Write code, break things, fix things

# If it works: merge it
git checkout main
git merge experiment/quantum-buffer-sizing

# If it doesn't work: just delete the branch, no harm done
git checkout main
git branch -D experiment/quantum-buffer-sizing
```

### Track your learning progress

```bash
# Create a learning log
cat >> ~/tenstorrent-learning/LEARNING_LOG.md << 'EOF'

## Session: [today's date]

### Topics covered:
- [ ] Phase 1.1: Memory wall problem
- [ ] Phase 1.2: Memory hierarchy
- [ ] Phase 2.1: Tensix core
- [ ] Phase 3.2: Spatial pipelining
- [ ] Phase 4.3: TTNN API
- [ ] Lab: RISC-V QEMU simulation
- [ ] Lab: Roofline model

### Notes:
(add your insights here)

### Questions to follow up:
(add things you didn't understand)
EOF

git add LEARNING_LOG.md
git commit -m "Log: learning session [date]"
```

---

## Part 15 — Cheat Sheet and Quick Reference

### Tenstorrent Stack (bottom to top)

```
Hardware Layer
├── Matrix Engine (32x32 tile matmul)
├── SFPU (element-wise: GELU, softmax, exp)
├── SRAM (1.5MB per Tensix)
└── RISC-V cores: BRISC, NCRISC, TRISC0/1/2

TT-LLK (Low Level Kernels)
├── llk_math_matmul.h    ← TRISC1 calls this for matrix multiply
├── llk_unpack*.h        ← TRISC0 controls the unpacker
└── llk_pack*.h          ← TRISC2 controls the packer

TT-Metalium (SDK)
├── dataflow_api.h       ← noc_async_read/write, cb_push_back/pop
├── compute_kernel_api/  ← matmul_tile, gelu_tile, pack_tile
└── CreateKernel, CreateCircularBuffer, LaunchProgram

TTNN (Tensor Library)
├── ttnn.matmul()        ← dispatches matmul kernels to core grid
├── ttnn.gelu()          ← SFPU element-wise
└── ttnn.from_torch()    ← converts PyTorch tensor, tiles it, moves to device

TT-Forge (ML Compiler)
├── torch.compile(model, backend="tt")
├── MLIR passes: TOSA → TTIR → TTNN
└── Automatic kernel dispatch

User Level (you)
└── PyTorch model → torch.compile → runs on Tensix
```

### Key vocabulary quick lookup

| Term | Simple definition |
|---|---|
| Tensix | One compute unit. Has 5 RISC-V cores + Math Engine + 1.5MB SRAM |
| BRISC | The RISC-V that reads data from GDDR6 into SRAM |
| NCRISC | The RISC-V that writes data from SRAM out to GDDR6 |
| TRISC | The 3 RISC-V that control unpack→compute→pack |
| CB | Circular Buffer — SRAM slots for handoff between BRISC and TRISC |
| NoC | Network-on-Chip — the 2D mesh connecting all Tensix cores |
| Tile | 32×32 block of values — the atomic unit of computation |
| GDDR6 | The off-chip DRAM (32GB on Blackhole, 576 GB/s) |
| HBM | What NVIDIA uses instead of GDDR6. Faster but 5× more expensive |
| LLK | Lowest-level kernel: the actual RISC-V instruction sequences |
| TTNN | Python tensor library, like PyTorch for Tenstorrent |
| Forge | ML compiler. Converts PyTorch graphs to TTNN calls |
| Spatial pipelining | Running different tokens through different cores simultaneously |

### Essential file locations in tt-metal

```bash
# RISC-V firmware (what actually runs on BRISC/NCRISC/TRISC)
~/tt-metal/tt_metal/hw/firmware/src/brisc.cc
~/tt-metal/tt_metal/hw/firmware/src/ncrisc.cc

# NoC API (the commands for reading/writing over the mesh)
~/tt-metal/tt_metal/hw/inc/dataflow_api.h

# Compute kernel API (matmul_tile, gelu_tile, etc.)
~/tt-metal/tt_metal/hw/inc/compute_kernel_api/

# Programming examples (best way to learn)
~/tt-metal/tt_metal/programming_examples/

# TTNN tensor operations
~/tt-metal/ttnn/cpp/ttnn/operations/

# LLK math kernels
~/tt-metal/tt_metal/hw/ckernels/
```

### Commands you'll use daily

```bash
# Activate environment
source ~/tt-env/bin/activate

# Navigate to your work
cd ~/tenstorrent-learning

# Run a Python lab
python3 labs/lab1-memory-roofline/roofline_analysis.py

# Build tt-metal after code changes
cd ~/tt-metal && cmake --build build --parallel $(nproc)

# Run a RISC-V program on QEMU
qemu-riscv64 ./my_riscv_program

# Cross-compile C to RISC-V
riscv64-linux-gnu-gcc -O2 -o output input.c

# Git: save your work
git add . && git commit -m "notes: studied NoC routing"
```

### Learning path mapped to mastery doc phases

| Mastery Doc Phase | Lab to do | Key concept to verify |
|---|---|---|
| Phase 1 (Memory wall) | Lab 1: Roofline | Can you explain ridge point? |
| Phase 1.4 (GPU baseline) | Part 9 Study | Can you draw GPU vs TT data flow? |
| Phase 2 (Tensix core) | Part 7 QEMU | Did you run RISC-V code on QEMU? |
| Phase 3 (Dataflow) | Lab 4 Pipeline | Does your pipeline show speedup? |
| Phase 4.4 (Metalium) | Lab 2 Kernel | Can you explain CB push/pop? |
| Phase 4.3 (TTNN) | Lab 3 Tensors | Can you explain tile layout? |
| Phase 4.2 (Forge) | Lab 5 Forge | Can you name the MLIR passes? |
| Phase 6.4 (Sources) | Lab 6 Explore | Did you read brisc.cc? |

---

## Appendix A — Troubleshooting

### "command not found: clang"
```bash
sudo apt install clang-17
which clang-17
sudo ln -s /usr/bin/clang-17 /usr/bin/clang
```

### "No module named ttnn"
```bash
source ~/tt-env/bin/activate
echo $PYTHONPATH   # must contain ~/tt-metal paths
export PYTHONPATH=$HOME/tt-metal:$HOME/tt-metal/ttnn:$PYTHONPATH
```

### WSL can't access the internet
Open Windows PowerShell (not WSL):
```powershell
wsl --shutdown
wsl
```
Then try `ping google.com` again.

### Build fails with "out of memory"
Reduce parallel jobs:
```bash
cmake --build build --parallel 2   # use only 2 cores instead of all
```

### Can't push to GitHub
```bash
# Re-add your SSH key
ssh -T git@github.com
# If it fails, re-run Part 2.4
```

---

## Appendix B — Next Steps After This Guide

Once you've completed all labs:

1. **Run the real tt-metal programming examples** in order (hello_world → eltwise → matmul_single_core → matmul_multi_core)

2. **Write your own kernel** — pick one small operation (e.g., vector addition) and write reader + compute + writer kernels from scratch

3. **Study tt-llk** — read `llk_math_matmul.h` and understand how the Matrix Engine is programmed at the LLK level

4. **Read the Hot Chips papers** — search hotchips.org for "Tenstorrent 2023" — the architecture slides make the mastery doc come alive visually

5. **If you get hardware access** — Tenstorrent Wormhole developer kits exist. Your employer may have them. The same code you wrote here runs directly on real hardware.

---

*Guide version 1.0 — Covers tt-metal as of 2024–2025. Check the GitHub repo for API changes.*
