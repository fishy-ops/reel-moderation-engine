# Aurora benchmark

Benchmark date: 2026-09-15. These figures establish engineering throughput only;
they are not accuracy claims.

## Setup

- Host: Aurora R8, Intel i7-9700, 64 GB memory, NVIDIA RTX 2070 Super 8 GB
- Input: generated 60-second, 1280×720, 30 FPS non-sensitive test pattern
- Sampling: every-frame 96-pixel thumbnail scan, 2 FPS baseline inference
- Detector: NudeNet 3.4.2, ONNX Runtime
- Container limit: 4 CPU cores, 12 GB memory

## Results

| Execution provider | Engine time | Throughput |
| --- | ---: | ---: |
| CPU | 33.53 seconds | 1.79× real time |
| CUDA | 5.13 seconds | 11.70× real time |

The CUDA run decoded all 1,800 frames, analyzed 120 full-resolution candidates,
reported `nudenet/CUDAExecutionProvider`, and returned `APPROVED`. End-to-end wall
time including container startup was 7.20 seconds (8.33× real time).

The GPU worker is the deployed default override on Aurora. Startup requires CUDA;
it raises an error instead of silently falling back to CPU if the runtime is mismatched.
