# Hardware plan

## Aurora: primary compute and storage

- RTX 2070 Super with 8 GB VRAM
- 64 GB system memory
- Approximately 7.1 TB free on the ZFS storage pool
- Docker and FFmpeg already available

Use a dedicated `/tank/archive/reel-moderation` tree. Keep the API and worker inside
resource-limited containers so existing photo services retain capacity. The initial
worker uses no more than four CPU cores and 12 GB RAM. GPU scheduling will be added
only after baseline profiling proves it improves end-to-end throughput.

## iMac: optional staging/backup

The iMac has roughly 917 GB free but an obsolete training GPU. It can hold encrypted
backups or non-sensitive build artifacts, not act as the main inference node.

## Raspberry Pi 4: optional monitoring

The Pi has 4 GB RAM and limited storage. It is suitable for health checks and queue
telemetry only.

## MacBook Pro: control plane only

The MacBook workspace may contain source code, tests, and documentation. It must not
contain datasets, model weights, extracted frames, evidence images, or model caches.
