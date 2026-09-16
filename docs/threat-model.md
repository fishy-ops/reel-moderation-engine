# Threat model

## In scope

- One- or two-frame insertions
- Small mirrors, nested reflections, and picture-in-picture regions
- Foreground occlusion and rapid clothing transitions
- Cropping, rotation, blur, compression, low light, and unusual aspect ratios
- Malformed containers, oversized uploads, path traversal, and queue exhaustion
- False positives affecting legitimate fashion, health, art, or cultural content
- Leakage of raw uploads or extracted detector evidence

## Controls in the prototype

- Every frame receives a cheap thumbnail-level discontinuity check.
- Suspicious changes open dense temporal bursts and overlapping detector crops.
- Analyzed frames are never included in reports.
- Detector scratch images live in tmpfs and are immediately deleted.
- Input paths are constrained to the configured data directory.
- Upload size and container suffixes are bounded.
- Worker readiness and bounded queue capacity are checked before upload bytes are stored.
- A time-bounded, local-protocol FFmpeg probe rejects malformed, overlong, and
  extreme-resolution inputs before queue admission.
- Service roots and the model cache are read-only, transient upload/parser storage is
  tmpfs, Linux capabilities are dropped, and privilege escalation is disabled.
- Opaque API upload copies are deleted after either successful or failed processing.
- The API binds to localhost and Redis has no host port.
- Containers run without root and have CPU/memory limits.
- Ambiguous evidence routes to human review.

## Required before public deployment

- Authentication, authorization, rate limiting, malware/container validation, and TLS
- A CSAM-specific safety service and legally reviewed reporting process
- A written retention/deletion policy and an appeals workflow
- Dataset/model provenance review and demographic/context bias evaluation
- Calibrated thresholds with versioned model and policy identifiers
- Encrypted storage, restricted audit logs, monitoring, and incident response
