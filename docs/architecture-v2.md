# Phase-two moderation architecture

The phase-two worker is a cascade. The independent verifier runs on a throttled
baseline as well as visual-change and primary-detector triggers. That path preserves
the ability to recover a complete primary-model miss while fitting Aurora's shared GPU.

```text
video
  -> SHA-256 + perceptual frame fingerprint
  -> every-frame thumbnail/change scan
  -> NudeNet full-frame and overlapping-region proposals
  -> SigLIP 2 context + Falconsai binary NSFW verifier ensemble
  -> temporal region tracks and evidence clustering
  -> configurable policy decision
  -> review feedback and report-priority store
```

The adaptive candidate budget applies only to change-triggered extras. Uniform baseline
samples are never dropped, even if an adversarial strobe exhausts the adaptive budget.

## Semantic verifier modes

`audit` records semantic probabilities and decisions but leaves the NudeNet risk
unchanged. Reports also include event-level agreement/disagreement counts and temporal
region tracks. This is the default until a held-out set establishes calibration.

`review` is the deployed high-recall mode. Temporally adjacent agreement may raise an
otherwise approved video to human review, but it does not change frame risk or invent
an evidence timestamp. Covered-clothing disagreement cannot lower primary risk.
`enforce` may adjust frame risk in both directions; it remains disabled. In every mode,
viewer reports only prioritize review and never auto-block content.

The verifier evaluates candidate regions in memory. Hugging Face model files live in
Aurora's model volume. Semantic regions from one detector batch are scored together to
reduce GPU launch overhead. Both model revisions are pinned, recorded in the report
configuration, and loaded offline from Aurora's cache. No decoded frames are retained.

The initial operator calibration showed why one-way review assistance matters: SigLIP 2 correctly
recognized sustained covered-clothing evidence in the known false positive, but also
emitted covered/uncertain signals on a confirmed violation. A separately trained
Apache-licensed binary image classifier is therefore recorded alongside it for model
agreement. Neither independent signal is yet allowed to override verdicts.

## Region tracking

Exposed detections in the same crop are linked using temporal gaps and bounding-box
intersection-over-union. Reports contain only track label, crop name, timestamps, hit
count, and maximum confidence. Region coordinates are not copied into the feedback
database.

## Perceptual fingerprints

The fingerprint CLI samples low-resolution grayscale frames and emits 64-bit difference
hashes. Comparison tolerates modest transcoding, trimming, and horizontal mirroring by
matching each hash to its nearest direct or mirror-derived candidate. This is an
initial near-duplicate layer, not a replacement for
Meta's PDQ/TMK+PDQF or another audited industry hash implementation.

## Reviewer and report feedback

Reviewer feedback stores hashes, predicted verdict, ground truth, optional event
timestamps, reason, reviewer identifier, pipeline source fingerprint, and decision
configuration fingerprint in SQLite. It stores no images. Outcomes can be grouped by
exact configuration so feedback from different deployments is not mixed. Report
priority uses report rate, trusted-report fraction, and recent velocity, capped below
the automatic-enforcement threshold to resist brigading.

## Adversarial challenge generation

The generator supports mirror, compression, blur, secondary-reflection, and occlusion
variants. It refuses to run unless `MODERATION_DERIVED_MEDIA_ROOT` is explicitly set
and the output resolves inside that root. On Aurora this is `/data/challenges`.
