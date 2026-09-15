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

## Operator five-video micro-regression

This is a development regression, not a general accuracy estimate. The five clips were
used during iteration and contain only one negative example, so the metrics must not be
presented as held-out or representative performance.

| Configuration | Recall | Precision | Review | Auto-block | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| NudeNet baseline | 75% (3/4) | 75% (3/4) | 3 | 1 | not comparable |
| Review-only two-model ensemble | 100% (4/4) | 80% (4/5) | 4 | 1 | 1.12× real time |

The ensemble processed 75.53 seconds of media in 67.40 seconds. It recovered the known
primary-model miss as a review, preserved the correct automatic block, and retained the
known revealing-clothing false positive as review. The next accuracy gate requires many
more representative negative examples and a genuinely held-out split.
