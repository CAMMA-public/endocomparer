#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
GPUS=${GPUS:-2}
CONFIG=${CONFIG:-configs/endocomparer_qwen35_9b.yaml}
FRAME_CACHE=${FRAME_CACHE:-cache/frames}
PYTHON=${PYTHON:-python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false FORCE_QWENVL_VIDEO_READER=decord
export FLA_TILELANG=${FLA_TILELANG:-0}                       # Triton path for the Gated-DeltaNet backward
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
# One Triton cache per rank: concurrent first-time compilation in one directory races on network file systems.
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PWD/cache/triton}
exec "$PYTHON" -m torch.distributed.run --standalone --nproc_per_node="$GPUS" --no-python \
    bash -c 'export TRITON_CACHE_DIR="$TRITON_CACHE_DIR/rank${LOCAL_RANK:-0}"; mkdir -p "$TRITON_CACHE_DIR"; exec "$@"' _ \
    "$PYTHON" -m endocomparer.train --config "$CONFIG" --grad-accum "$((16 / GPUS))" --frame-cache "$FRAME_CACHE" "$@"
