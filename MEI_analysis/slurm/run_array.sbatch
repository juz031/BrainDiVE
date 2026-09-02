#!/bin/bash
#SBATCH --job-name=mei-feature

set -euo pipefail

: "${MEI_CONFIG:?MEI_CONFIG is required}"
: "${MEI_FEATURE:?MEI_FEATURE is required}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"

CODE_ROOT=${MEI_CODE_ROOT:-/home/junruz/BrainDiVE/MEI_analysis}
PYTHON=${MEI_PYTHON:-/home/junruz/.local/envs/env2025/bin/python}

# Keep each array element inside its Slurm CPU allocation. PyTorch/BLAS may
# otherwise detect every core on the node and oversubscribe neighboring jobs.
THREADS=${SLURM_CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=${MEI_NUM_THREADS:-${THREADS}}
export MKL_NUM_THREADS=${MEI_NUM_THREADS:-${THREADS}}
export OPENBLAS_NUM_THREADS=${MEI_NUM_THREADS:-${THREADS}}
export NUMEXPR_NUM_THREADS=${MEI_NUM_THREADS:-${THREADS}}

cd "${CODE_ROOT}"
echo "job=${SLURM_JOB_ID} array=${SLURM_ARRAY_TASK_ID} feature=${MEI_FEATURE} host=$(hostname) threads=${OMP_NUM_THREADS}"
if [[ "${MEI_FEATURE}" == "basic" ]]; then
    DEVICE=cpu
else
    DEVICE=${MEI_DEVICE:-cuda}
    if [[ "${DEVICE}" == "cuda" ]]; then
        nvidia-smi --query-gpu=name,memory.total --format=csv
    fi
fi

EXTRA=()
if [[ "${MEI_FORCE:-0}" == "1" ]]; then
    EXTRA+=(--force)
fi
if [[ -n "${MEI_BATCH_SIZE:-}" ]]; then
    EXTRA+=(--batch-size "${MEI_BATCH_SIZE}")
fi

"${PYTHON}" -u run_pipeline.py --config "${MEI_CONFIG}" extract \
    --feature "${MEI_FEATURE}" \
    --task-id "${SLURM_ARRAY_TASK_ID}" \
    --device "${DEVICE}" \
    "${EXTRA[@]}"
