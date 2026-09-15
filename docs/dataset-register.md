# Dataset and model register

No explicit-content dataset has been downloaded. A license badge alone is not enough:
the project also requires traceable provenance, consent or a defensible lawful basis,
and verified-adult subjects for every explicit photographic example.

| Asset | Intended use | Status | Reason |
| --- | --- | --- | --- |
| NudeNet 3.4.2 bundled weights | Prototype anatomical detector | Prototype only | Redistributable package is available, but training-data documentation is insufficient for a production claim. |
| Google SigLIP 2 base patch16-224 | Candidate semantic exposed-vs-covered verifier | Approved for audit evaluation | Google-published model files and code are Apache-2.0; zero-shot policy calibration is still required before enforcement. |
| Falconsai `nsfw_image_detection` | Independent binary verifier | Approved for audit evaluation | The published weights are Apache-2.0; training-set provenance and demographic evaluation remain insufficient for enforcement. |
| Cashbox NSFW Image Level v6 | Possible calibration data | Rejected for now | The card labels content 18+, but source sites are undisclosed and consent/adult-verification evidence is absent. |
| UnsafeSceneDetection | Possible research evaluation | Hold | Gated and non-commercial; access requires identity/affiliation terms and explicit operator acceptance. |
| Public Domain 12M artistic subset | Contextual false-positive evaluation | Candidate | Public-domain provenance is documented and non-artistic photographic nudity was filtered; use only a small license-verified subset. |
| Generated shapes and overlays | Sampling regression tests | Approved | Contains no people or explicit content and tests temporal/codec behavior only. |
| Five operator-provided private clips | Regression evaluation only | Approved, restricted | Operator confirmed lawful acquisition and supplied video-level/event labels; media stays on Aurora and is excluded from source control. |

Before acquiring a held or candidate asset, add its exact revision, terms, checksum,
age/consent basis, retention rule, and deletion procedure to the local restricted
manifest. Model approval and dataset approval are separate decisions.
