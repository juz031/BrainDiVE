#!/bin/bash
#SBATCH --job-name=mei-stage

set -euo pipefail

: "${MEI_CONFIG:?MEI_CONFIG is required}"
: "${MEI_STAGE:?MEI_STAGE is required}"

CODE_ROOT=${MEI_CODE_ROOT:-/home/junruz/BrainDiVE/MEI_analysis}
PYTHON=${MEI_PYTHON:-/home/junruz/.local/envs/env2025/bin/python}
cd "${CODE_ROOT}"
echo "job=${SLURM_JOB_ID} stage=${MEI_STAGE} host=$(hostname)"

case "${MEI_STAGE}" in
    prepare|aggregate|statistics|figures|report)
        "${PYTHON}" -u run_pipeline.py --config "${MEI_CONFIG}" "${MEI_STAGE}"
        ;;
    validate)
        "${PYTHON}" -u run_pipeline.py --config "${MEI_CONFIG}" validate --require-analysis
        ;;
    *)
        echo "Unsupported stage: ${MEI_STAGE}" >&2
        exit 2
        ;;
esac
