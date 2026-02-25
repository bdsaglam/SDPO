#!/bin/bash

# Usage: ./run_arc_agi_sdpo.sh [experiment_name_suffix]
#
# Environment overrides:
#   MAX_TURNS=4      — max REPL turns (default 8, set 0 for single-turn)

# =============================================================================
# CONFIGURATION
# =============================================================================

CONFIG_NAME="arc_agi"

# Default to ARC-AGI 2024 dataset (with analysis hints)
DATA_PATH="datasets/arc_agi/2024-with-hints"

# Hyperparameters
TRAIN_BATCH_SIZE=16
ROLLOUT_N=8
LR=1e-5
SUCCESS_THRESHOLD=1.0
DONTS_REPROMPT_ON_SELF_SUCCESS=True
ALPHA=0.5
MODEL_PATH="Qwen/Qwen2.5-7B-Instruct"
export N_GPUS_PER_NODE=4

MAX_TURNS=${MAX_TURNS:-8}

# Allow overriding experiment name suffix
SUFFIX=${1:-"arc_agi_sdpo"}

# =============================================================================
# SETUP
# =============================================================================

# Get the directory where this script is located
export PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export SDPO_DIR="$PROJECT_ROOT"
export PYTHONPATH=$PROJECT_ROOT:$PYTHONPATH

# Define USER for Hydra config (required by user.yaml)
export USER=${USER:-$(whoami)}

# Direct file logger output into outputs/ instead of project root
export VERL_FILE_LOGGER_ROOT="$PROJECT_ROOT/outputs/logs"


# =============================================================================
# EXECUTION
# =============================================================================

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="ARC-SDPO-MT${MAX_TURNS}-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_N}-lr${LR}-threshold${SUCCESS_THRESHOLD}-${MODEL_NAME}-${SUFFIX}"

ARGS="data.train_batch_size=$TRAIN_BATCH_SIZE \
trainer.group_name=ARC-SDPO \
actor_rollout_ref.rollout.n=$ROLLOUT_N \
actor_rollout_ref.model.path=$MODEL_PATH \
actor_rollout_ref.actor.optim.lr=$LR \
actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
actor_rollout_ref.actor.self_distillation.success_reward_threshold=$SUCCESS_THRESHOLD \
actor_rollout_ref.actor.self_distillation.alpha=$ALPHA \
actor_rollout_ref.actor.self_distillation.distillation_topk=100 \
actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=${DONTS_REPROMPT_ON_SELF_SUCCESS} \
actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
actor_rollout_ref.actor.use_dynamic_bsz=true \
actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
actor_rollout_ref.actor.entropy_checkpointing=true \
actor_rollout_ref.actor.entropy_from_logits_with_chunking=true \
actor_rollout_ref.ref.fsdp_config.param_offload=true \
actor_rollout_ref.model.enable_activation_offload=true \
actor_rollout_ref.rollout.enforce_eager=true \
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
algorithm.rollout_correction.rollout_is=token \
actor_rollout_ref.rollout.val_kwargs.n=8 \
trainer.logger=['console','file','wandb'] \
trainer.rollout_data_dir=rollouts \
trainer.validation_data_dir=val_rollouts"

if [ "$MAX_TURNS" -gt 0 ]; then
    ARGS="$ARGS \
data.return_raw_chat=True \
data.max_response_length=24576 \
max_model_len=32768 \
actor_rollout_ref.rollout.multi_turn.enable=True \
actor_rollout_ref.rollout.multi_turn.max_assistant_turns=${MAX_TURNS} \
actor_rollout_ref.rollout.multi_turn.max_user_turns=${MAX_TURNS} \
actor_rollout_ref.rollout.multi_turn.interaction_config_path=config/arc_agi_interaction_config.yaml \
actor_rollout_ref.rollout.multi_turn.max_tool_response_length=4096"
fi

echo "----------------------------------------------------------------"
echo "Starting ARC-AGI SDPO Training"
echo "Experiment: $EXP_NAME"
echo "Data: $DATA_PATH"
echo "Model: $MODEL_PATH"
echo "GPUs: $N_GPUS_PER_NODE"
echo "Max turns: $MAX_TURNS (0 = single-turn)"
echo "Hints: baked into dataset"
echo "Success threshold: $SUCCESS_THRESHOLD (exact match only)"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" $ARGS
