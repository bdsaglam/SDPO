#!/bin/bash

# Usage: ./run_arc_agi_sdpo.sh [experiment_name_suffix]

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="arc_agi"
DATA_PATH="datasets/arc_agi/2024"
MODEL_PATH="Qwen/Qwen3-8B"
export N_GPUS_PER_NODE=4

# Hyperparameters
TRAIN_BATCH_SIZE=16
ROLLOUT_N=8
LR=1e-5
SUCCESS_THRESHOLD=1.0
ALPHA=0.5

# Allow overriding experiment name suffix
SUFFIX=${1:-"arc_agi_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH
export USER=${USER:-$(whoami)}

# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="ARC-SDPO-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_N}-lr${LR}-threshold${SUCCESS_THRESHOLD}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=ARC-SDPO \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.actor.self_distillation.success_reward_threshold=$SUCCESS_THRESHOLD \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=8"

echo "----------------------------------------------------------------"
echo "Starting ARC-AGI SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "GPUs: $N_GPUS_PER_NODE"
echo "Analysis: enabled (Gemini)"
echo "Success threshold: $SUCCESS_THRESHOLD (exact match only)"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
