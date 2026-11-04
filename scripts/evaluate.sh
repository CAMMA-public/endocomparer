#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG=${CONFIG:-configs/eval_default.yaml}
FRAME_CACHE=${FRAME_CACHE:-cache/frames}
PYTHON=${PYTHON:-python}
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false FORCE_QWENVL_VIDEO_READER=decord FLA_TILELANG=${FLA_TILELANG:-0}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$PWD/cache/triton/infer}
mkdir -p "$TRITON_CACHE_DIR"
exec "$PYTHON" -m endocomparer.infer --config "$CONFIG" --frame-cache "$FRAME_CACHE" "$@"
