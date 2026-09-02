#!/bin/bash
set -euo pipefail

CODE_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
CONFIG="${CODE_ROOT}/config/analysis_s1.yaml"
PYTHON=${MEI_PYTHON:-/home/junruz/.local/envs/env2025/bin/python}
DRY_RUN=0
FORCE=0
MAX_CONCURRENT=""

usage() {
    echo "Usage: $0 [--config PATH] [--max-concurrent N] [--dry-run] [--force]"
}

while (($#)); do
    case "$1" in
        --config) CONFIG=$2; shift 2 ;;
        --max-concurrent) MAX_CONCURRENT=$2; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

CONFIG=$(readlink -f "${CONFIG}")
"${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" plan
OUTPUT_ROOT=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value paths.output_root)
TOTAL=$(($(wc -l < "${OUTPUT_ROOT}/manifests/work_units.csv") - 1))
if ((TOTAL < 1)); then echo "No work units" >&2; exit 1; fi
LAST=$((TOTAL - 1))
if [[ -z "${MAX_CONCURRENT}" ]]; then
    MAX_CONCURRENT=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value slurm.array_max_concurrent)
fi
CPU_PARTITION=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value slurm.cpu_partition)
GPU_PARTITION=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value slurm.gpu_partition)
ACCOUNT=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value slurm.account)
LOG_DIR="${OUTPUT_ROOT}/logs"
mkdir -p "${LOG_DIR}"

submit() {
    if ((DRY_RUN)); then
        printf 'DRY-RUN:' >&2; printf ' %q' sbatch "$@" >&2; printf '\n' >&2
        echo "$((100000 + RANDOM))"
    else
        sbatch --parsable "$@"
    fi
}

account_args=()
if [[ -n "${ACCOUNT}" ]]; then account_args+=(--account "${ACCOUNT}"); fi
exports="ALL,MEI_CONFIG=${CONFIG},MEI_CODE_ROOT=${CODE_ROOT},MEI_PYTHON=${PYTHON},MEI_FORCE=${FORCE}"

prepare_id=$(submit "${account_args[@]}" --partition="${CPU_PARTITION}" --job-name=mei-prepare --time=04:00:00 --mem=32G --cpus-per-task=8 \
    --output="${LOG_DIR}/prepare-%j.out" --export="${exports},MEI_STAGE=prepare" "${CODE_ROOT}/slurm/run_stage.sbatch")

feature_job() {
    local feature=$1
    local time memory cpus gpus
    time=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value "slurm.${feature}.time")
    memory=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value "slurm.${feature}.memory")
    cpus=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value "slurm.${feature}.cpus")
    gpus=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value "slurm.${feature}.gpus")
    local partition
    partition=$("${PYTHON}" "${CODE_ROOT}/run_pipeline.py" --config "${CONFIG}" config-value "slurm.${feature}.partition")
    local args=("${account_args[@]}" --partition="${partition}" --dependency="afterok:${prepare_id}" --job-name="mei-${feature}" \
        --array="0-${LAST}%${MAX_CONCURRENT}" --time="${time}" --mem="${memory}" --cpus-per-task="${cpus}" \
        --output="${LOG_DIR}/${feature}-%A_%a.out" --export="${exports},MEI_FEATURE=${feature}")
    if ((gpus > 0)); then args+=(--gres="gpu:${gpus}"); fi
    submit "${args[@]}" "${CODE_ROOT}/slurm/run_array.sbatch"
}

basic_id=$(feature_job basic)
gabor_id=$(feature_job gabor)
curvature_id=$(feature_job curvature)
feature_dependency="afterok:${basic_id}:${gabor_id}:${curvature_id}"

stage_job() {
    local stage=$1 dependency=$2
    submit "${account_args[@]}" --partition="${CPU_PARTITION}" --dependency="afterok:${dependency}" --job-name="mei-${stage}" \
        --time=04:00:00 --mem=32G --cpus-per-task=8 \
        --output="${LOG_DIR}/${stage}-%j.out" --export="${exports},MEI_STAGE=${stage}" \
        "${CODE_ROOT}/slurm/run_stage.sbatch"
}

aggregate_id=$(submit "${account_args[@]}" --partition="${CPU_PARTITION}" --dependency="${feature_dependency}" --job-name=mei-aggregate \
    --time=04:00:00 --mem=64G --cpus-per-task=8 --output="${LOG_DIR}/aggregate-%j.out" \
    --export="${exports},MEI_STAGE=aggregate" "${CODE_ROOT}/slurm/run_stage.sbatch")
statistics_id=$(stage_job statistics "${aggregate_id}")
figures_id=$(stage_job figures "${statistics_id}")
report_id=$(stage_job report "${figures_id}")
validate_id=$(stage_job validate "${report_id}")

printf 'Submitted pipeline\nprepare=%s\nbasic=%s\ngabor=%s\ncurvature=%s\naggregate=%s\nstatistics=%s\nfigures=%s\nreport=%s\nvalidate=%s\n' \
    "${prepare_id}" "${basic_id}" "${gabor_id}" "${curvature_id}" "${aggregate_id}" \
    "${statistics_id}" "${figures_id}" "${report_id}" "${validate_id}"
