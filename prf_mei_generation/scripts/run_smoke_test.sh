#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/hanfeig/venvs/mei-cu128/bin/python}"
BASE_SITE="${BASE_SITE:-/home/hanfeig/conda-envs/mei/lib/python3.11/site-packages}"

export LD_LIBRARY_PATH="${BASE_SITE}/cusparselt/lib:${BASE_SITE}/nvidia/nccl/lib:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" maximize_prf_activation.py \
  --task_config configs/clip_rn50_v1_contrast_smoke.json \
  --task_id 0
