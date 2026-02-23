#!/bin/bash
# Run SDPO or GRPO training locally without SLURM.
#
# Usage:
#   ./run_local.sh [OPTIONS]
#
# Options:
#   -m, --method       sdpo|grpo              (default: sdpo)
#   -d, --dataset      dataset path           (default: datasets/sciknoweval/chemistry)
#   -M, --model        model name/path        (default: Qwen/Qwen3-8B)
#   -g, --gpus         number of GPUs         (default: auto-detect)
#   -n, --name         experiment name suffix  (default: local)
#   --no-wandb         disable W&B logging
#   --dry-run          print the command without executing
#   -h, --help         show this help
#
# Examples:
#   ./run_local.sh                                          # SDPO on chemistry, Qwen3-8B
#   ./run_local.sh -m grpo -d datasets/tooluse              # GRPO on tooluse
#   ./run_local.sh -g 1 --no-wandb                          # single GPU, no W&B
#   ./run_local.sh -M allenai/Olmo-3-7B-Instruct            # different model
#   ./run_local.sh --dry-run                                # show command only

set -e

# =============================================================================
# DEFAULTS
# =============================================================================

METHOD="sdpo"
DATASET="datasets/sciknoweval/chemistry"
MODEL="Qwen/Qwen3-8B"
GPUS=""
NAME_SUFFIX="local"
USE_WANDB=true
DRY_RUN=false

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =============================================================================
# PARSE ARGS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--method)    METHOD="$2";       shift 2;;
        -d|--dataset)   DATASET="$2";      shift 2;;
        -M|--model)     MODEL="$2";        shift 2;;
        -g|--gpus)      GPUS="$2";         shift 2;;
        -n|--name)      NAME_SUFFIX="$2";  shift 2;;
        --no-wandb)     USE_WANDB=false;   shift;;
        --dry-run)      DRY_RUN=true;      shift;;
        -h|--help)
            sed -n '2,/^$/p' "$0" | grep '^#' | sed 's/^# \?//'
            exit 0;;
        *) echo "Unknown option: $1"; exit 1;;
    esac
done

# =============================================================================
# AUTO-DETECT GPUs
# =============================================================================

if [ -z "$GPUS" ]; then
    if command -v nvidia-smi &>/dev/null; then
        GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
    else
        GPUS=1
    fi
fi

# =============================================================================
# VALIDATE
# =============================================================================

if [[ "$METHOD" != "sdpo" && "$METHOD" != "grpo" ]]; then
    echo "Error: method must be 'sdpo' or 'grpo', got '$METHOD'"
    exit 1
fi

DATASET_PATH="$PROJECT_ROOT/$DATASET"
if [ ! -f "$DATASET_PATH/train.parquet" ]; then
    echo "Error: $DATASET_PATH/train.parquet not found."
    echo "Run ./preprocess_all.sh first."
    exit 1
fi

# =============================================================================
# BUILD CONFIG
# =============================================================================

MODEL_TAG=$(echo "$MODEL" | tr '/' '-')

if [ "$METHOD" = "sdpo" ]; then
    CONFIG_NAME="sdpo"
    EXP_NAME="SDPO-${MODEL_TAG}-${NAME_SUFFIX}"
    METHOD_ARGS="\
        actor_rollout_ref.actor.ppo_mini_batch_size=32 \
        actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
        actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
        actor_rollout_ref.actor.self_distillation.alpha=0.5 \
        actor_rollout_ref.actor.self_distillation.include_environment_feedback=False \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        algorithm.rollout_correction.rollout_is=token"
else
    CONFIG_NAME="baseline_grpo"
    EXP_NAME="GRPO-${MODEL_TAG}-${NAME_SUFFIX}"
    METHOD_ARGS="\
        actor_rollout_ref.actor.ppo_mini_batch_size=8 \
        actor_rollout_ref.actor.optim.lr=1e-5 \
        actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
        algorithm.rollout_correction.rollout_is=token"
fi

COMMON_ARGS="\
    data.train_batch_size=32 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.model.path=$MODEL \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    trainer.n_gpus_per_node=$GPUS \
    trainer.group_name=${METHOD^^}-local"

# Disable W&B if requested
if [ "$USE_WANDB" = false ]; then
    COMMON_ARGS="$COMMON_ARGS trainer.logger=[console]"
fi

# =============================================================================
# ENVIRONMENT
# =============================================================================

export SDPO_DIR="$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"
export TASK="$DATASET"
export EXPERIMENT="$EXP_NAME"
export VLLM_USE_V1=1
export N_GPUS_PER_NODE="$GPUS"
unset VLLM_ATTENTION_BACKEND

# =============================================================================
# RUN
# =============================================================================

CMD="python -m verl.trainer.main_ppo --config-name $CONFIG_NAME $COMMON_ARGS $METHOD_ARGS"

echo "================================================================"
echo "  Method:     $METHOD"
echo "  Config:     $CONFIG_NAME"
echo "  Dataset:    $DATASET"
echo "  Model:      $MODEL"
echo "  GPUs:       $GPUS"
echo "  Experiment: $EXP_NAME"
echo "  W&B:        $USE_WANDB"
echo "================================================================"

if [ "$DRY_RUN" = true ]; then
    echo ""
    echo "Command (dry run):"
    echo "$CMD"
else
    exec $CMD
fi
