# Evaluation plan

## Primary targets

- At least 95% recall on the curated evasive-challenge set
- Less than 5% false-positive rate on representative ordinary video
- Explicit measurement for one-frame and two-frame events at 24, 30, and 60 FPS
- Faster than 5× real time on the target Aurora worker after optimization

These are engineering gates, not claims about current performance.

## Test slices

Evaluate each category independently:

1. Direct, sustained exposure
2. One-frame and two-frame overlays
3. Rapid step-out/step-behind occlusion sequences
4. Small mirror and secondary-reflection regions
5. Cropped, rotated, compressed, blurred, and low-light evidence
6. Swimwear, exercise, shirtless people, breastfeeding, medical, and artistic contexts
7. Diverse skin tones, body types, camera quality, lighting, and gender presentation
8. Videos with no people, hard cuts, strobe effects, and rapid editing

The localized-change threshold begins at `0.12`, selected to catch a compressed
one-frame fixture whose measured local change was `0.188`. This is a provisional
high-recall setting; false-positive and throughput testing must precede release.

## Metrics

Report frame-, event-, and video-level precision/recall; review rate; auto-block error
rate; detection latency; decoded/analyzed-frame ratio; and wall-clock throughput.
Never tune and report against the same split. Hash inputs to prevent split leakage.

## Reproducible manifest runner

Keep media on Aurora and create a JSONL manifest beside it. Relative video paths are
resolved from the manifest directory; the runner reads them in place and never copies
them into the repository.

```json
{"id":"safe-001","video":"safe/safe-001.mp4","expected_violation":false,"split":"test","slices":["ordinary"]}
{"id":"flash-001","video":"challenge/flash-001.mp4","expected_violation":true,"events":[{"start_seconds":2.0,"end_seconds":2.1}],"split":"test","slices":["one-frame","30fps"]}
```

Run the held-out split on Aurora:

```bash
reel-moderate evaluate /data/eval/manifest.jsonl \
  --output /data/reports/evaluation-test.json \
  --split test --execution-provider cuda \
  --scratch-dir /moderation-scratch
```

The output contains only input hashes, expected labels and event bounds, verdicts,
aggregate metrics, slice metrics, and throughput. A review verdict counts as a positive
prediction when computing the high-recall gate; report review and auto-block counts
separately. Event-labeled cases also report event-localization recall, first-detection
latency, on-target timestamps, off-target timestamps, and timestamp precision. A
video-level semantic review with no primary evidence remains a video true positive but
is correctly counted as not localized.

Compare a candidate report with its approved baseline before deployment. By default,
the gate allows no recall or false-positive-rate regression, checks every shared slice,
requires exactly the same labeled cases, and allows at most a 10% throughput loss:

```bash
reel-moderate compare-evaluations \
  /data/reports/evaluation-approved.json \
  /data/reports/evaluation-candidate.json
```

The command emits a media-free JSON comparison and exits nonzero when the candidate
regresses. Each moderation report also records the pipeline version, a deterministic
fingerprint of the installed pipeline source, and a canonical SHA-256 fingerprint of
all decision-affecting configuration.
