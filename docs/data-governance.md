# Data governance

## Allowed sources

Every dataset must have documented provenance, usage rights, subject consent where
applicable, and an adult-only guarantee for explicit examples. Record the source,
license, version, checksum, acquisition date, permitted uses, and deletion procedure
in a local manifest that is excluded from Git.

Benign evaluation material may include ordinary fashion, swimwear, exercise,
breastfeeding, medical, documentary, and artistic contexts when legally obtained.
These contexts are essential for measuring false positives, but private or sensitive
health imagery is not appropriate without a separate governance process.

## Prohibited sources

- CSAM or any explicit content involving a minor
- Explicit content where age is unknown or ambiguous
- Non-consensual intimate imagery
- Scraped private accounts or platform content without permission
- Leaked, doxxed, or unlawfully obtained media

If age or consent is uncertain, exclude the item. Do not ask a model to determine
whether an explicit subject is an adult as a substitute for provenance.

## Storage controls

- Store all media only under Aurora's dedicated ZFS dataset.
- Keep model scratch data in RAM-backed tmpfs.
- Delete each API upload copy after its job succeeds or fails; retain only the
  metadata-only report unless a documented evaluation-retention rule applies.
- Opportunistically remove abandoned task-ID upload copies after 24 hours without
  scanning or modifying operator-controlled source collections.
- Never copy media or model caches to the MacBook control workspace.
- Keep dataset and evidence paths out of logs and reports.
- Encrypt backups and limit Tailscale/Unix permissions to project operators.
- Define retention before ingesting any non-synthetic evaluation collection.
