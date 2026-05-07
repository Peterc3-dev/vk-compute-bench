# vk-compute-bench

Vulkan compute benchmark suite for AMD integrated GPUs, targeting Radeon 890M (RDNA 3.5).

## Features

- Memory bandwidth: device-local buffer copy, host-to-device write, device-to-host read
- SAXPY (a*x+y): measures FP32 compute throughput in GFLOPS
- Parallel reduction: workgroup-level tree reduction, reports GElem/s
- Matrix multiply (1024x1024): tiled compute shader, GFLOPS with peak % estimate
- GPU timestamp queries for accurate kernel timing (no CPU-side timing noise)
- UMA-aware: detects when GpuOnly alloc falls back to shared memory and adjusts reporting
- Results printed as a formatted table with throughput and peak % columns
- Reference peaks: ~8.6 TFLOPS FP32, ~89.6 GB/s memory bandwidth (LPDDR5X-7500)

## Install

```
cargo build --release
```

Requires Vulkan drivers (RADV) and `glslangValidator` for shader compilation at build time.

## Usage

```
./vk-compute-bench
```

Runs all benchmarks sequentially (memory, SAXPY, reduction, matmul) and prints results.

---

Built with Rust + ash + gpu-allocator.
