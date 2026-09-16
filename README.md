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

`/health` checks the API and Redis path. `/ready` additionally requires a registered
worker and available queue capacity. Uploads are rejected before any bytes are stored
when no worker is available or the configured backlog is full. After upload, a
15-second local FFmpeg preflight rejects unreadable containers, missing video streams,
durations over five minutes, and resolutions above 8K before queue admission.

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

The GPU worker keeps one model ensemble warm across jobs. The first task after a
restart pays model-load time; later tasks reuse the loaded models while resetting all
per-video temporal state.

The GPU override also enables a SigLIP 2 plus Falconsai verifier ensemble in `review`
mode. It runs on a periodic baseline, adaptive visual-change triggers, and
primary-model candidates, so it can observe complete primary-model misses. Audit mode
records signals without changing the verdict; review mode may only raise a candidate
to human review at the video level and can neither lower risk, invent an exact evidence
timestamp, nor auto-block by itself. Switch to full
`enforce` mode only after a held-out evaluation establishes safe thresholds. See the
[phase-two architecture](docs/architecture-v2.md).

## Feedback, fingerprints, and challenge tests

```bash
reel-moderate fingerprint /data/incoming/video.mp4 -o /data/reports/video.fingerprint.json
reel-moderate compare-fingerprints first.fingerprint.json second.fingerprint.json
reel-moderate record-feedback /data/reports/video.json \
  --ground-truth violation --reason "brief exposure" --event-start 2.0 --event-end 3.0
reel-moderate feedback-summary /data/reports/feedback.sqlite
reel-moderate generate-challenge /data/incoming/approved.mp4 \
  --variant reflection -o /data/challenges/approved-reflection.mp4
```

Challenge generation is restricted to Aurora's configured derived-media directory.
It must only be used with operator-approved, lawfully obtained source media.

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
high-recall proposals; two independent classifiers provide audit signals. The next model
milestone is calibration against a larger legally sourced held-out set, followed by
bias, context, adversarial, and appeal-outcome evaluation.
