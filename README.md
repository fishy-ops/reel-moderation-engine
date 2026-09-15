# Reel Moderation Engine

An adaptive, asynchronous video moderation prototype designed to catch brief and
evasive visual-policy violations: one-frame overlays, rapid exposure, mirrors,
small reflections, occlusion, and clothing transitions.

The project is intentionally gender-neutral. It detects policy-relevant visual
evidence rather than targeting a demographic or style of dress.

## Why ordinary keyframe sampling fails

Sampling one frame per second can miss a one- or two-frame insertion completely.
This engine decodes every frame at thumbnail resolution, measures global scene
change, localized overlay change, and luminance flashes, then opens a dense analysis
burst around suspicious moments. A fine-grained detector examines the selected
full-resolution frames and overlapping crops, which helps recover small reflections.

## Decision policy

- No evidence: `APPROVED`
- Ambiguous or isolated evidence: `FLAG_FOR_HUMAN_REVIEW`
- Repeated strong evidence or an extreme single-frame detection: `AUTO_BLOCK`

Thresholds are explicit configuration, not hard-coded product truth. They must be
calibrated against a representative, consented evaluation set before production use.

## Local CLI

Python 3.11–3.13 and FFmpeg are required.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[nudenet]'
reel-moderate moderate /path/to/video.mp4 --scratch-dir /dev/shm/reel-moderation
```

The `dry-run` backend exercises decoding and sampling without classifying content:

```bash
reel-moderate moderate sample.mp4 --backend dry-run
```

## API and worker

The provided Compose deployment starts:

- FastAPI on port `8088`
- one resource-limited RQ worker
- an isolated Redis instance
- a tmpfs scratch area so decoded detector inputs are never persisted

The API binds to localhost by default. Set `API_BIND_ADDRESS` to Aurora's Tailscale
address only when direct tailnet access is needed; do not expose it to the public web.

On Aurora:

```bash
docker compose up --build -d
curl -F 'file=@sample.mp4' http://localhost:8088/api/v1/moderate
curl http://localhost:8088/api/v1/status/TASK_ID
```

After verifying NVIDIA container support, use the GPU worker override:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build -d
```

The report's `backend` field records the ONNX execution provider actually selected;
do not assume GPU acceleration merely because a container can see the device.

## Privacy and data handling

- Raw videos, frames, datasets, weights, reports, and model caches are Git-ignored.
- Detector scratch files live in tmpfs and are deleted after every frame.
- Reports contain timestamps, labels, boxes, and scores, not evidence images.
- Uploaded videos are retained by default until an operator applies a reviewed
  retention policy; opt-in automatic deletion is available through configuration.
- Public data must have clear research/commercial rights and confirmed-adult subjects.
- Do not scrape social platforms or acquire content of unknown provenance or age.
- Never use this system to infer identity, gender, attractiveness, or other unrelated traits.

See [data governance](docs/data-governance.md), the
[dataset register](docs/dataset-register.md), the [threat model](docs/threat-model.md),
and the [evaluation plan](docs/evaluation.md).

Aurora's current safe synthetic benchmark is recorded in
[the benchmark report](docs/benchmark.md). The evaluation CLI consumes a labeled
JSONL manifest in place and produces aggregate and per-slice metrics without copying
media into the repository.

## Current status

This is a research prototype, not a production moderation authority. NudeNet provides
the initial fine-grained backend; the scorer interface is replaceable. The next model
milestone is a calibrated fast first-stage classifier trained on licensed data and
adversarial transformations, followed by bias and context evaluation.
