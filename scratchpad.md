# Setup

```sh
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt

pip install -e .

pip install flash-attn --no-build-isolation

pip install -r requirements_sglang.txt

pip install vllm

pip install word2number latex2sympy2 math-verify[antlr4_9_3]==0.8.0

./load_all.sh

./preprocess_all.sh
```


# Containerized usage

docker container stop sdpo
docker rm sdpo

docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
  --cap-add=SYS_ADMIN --entrypoint="" \
  --env-file .env \
  -v .:/workspace/sdpo --name sdpo \
  -v ./.dspy_cache:/root/.dspy_cache \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  verlai/verl:vllm011.latest sleep infinity

docker start sdpo

docker exec -it sdpo bash

# --- Inside the container ---

# 1. Install verl from your mounted source
cd /workspace/sdpo
pip3 install --no-deps -e .
pip3 install math-verify

# 2. Prepare the GSM8K dataset
python3 examples/data_preprocess/gsm8k.py --local_save_dir ~/data/gsm8k

# 3. Run PPO training (single-GPU minimal example)
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
  data.train_files=$HOME/data/gsm8k/train.parquet \
  data.val_files=$HOME/data/gsm8k/test.parquet \
  data.train_batch_size=256 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  actor_rollout_ref.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
  critic.optim.lr=1e-5 \
  critic.model.path=Qwen/Qwen2.5-0.5B-Instruct \
  critic.ppo_micro_batch_size_per_gpu=4 \
  algorithm.kl_ctrl.kl_coef=0.001 \
  trainer.logger=console \
  trainer.val_before_train=False \
  trainer.n_gpus_per_node=1 \
  trainer.nnodes=1 \
  trainer.save_freq=10 \
  trainer.test_freq=10 \
  trainer.total_epochs=15 2>&1 | tee verl_demo.log


# ARC-AGI SDPO

Uses `docker compose` with `Dockerfile.sdpo` (extends `verlai/verl:vllm011.latest`, pre-installs `dspy` and `math-verify`) and `entrypoint.sh` (runs `pip install --no-deps -e .` on startup).

```sh
# One-time build (re-run only if Dockerfile.sdpo changes)
docker compose build

# Start container
docker compose up -d

# Shell into it — ready to go, no manual pip installs needed
docker compose exec sdpo bash

# Stop & clean up
docker compose down
```

Inside the container:

## Data

Raw ARC-AGI data lives in `datasets/arc_agi/` organized by year:
- `datasets/arc_agi/2024/raw/` — ARC Prize 2024 (400 training, 400 evaluation)
- `datasets/arc_agi/2025/raw/` — ARC Prize 2025 (placeholder)
- `datasets/arc_agi/dummy/raw/` — 3-task dummy set for testing

```sh
# Prepare dummy dataset for testing
python3 data/prepare_arc_agi.py --data_folder datasets/arc_agi/dummy

# Prepare 2024 dataset
python3 data/prepare_arc_agi.py --data_folder datasets/arc_agi/2024

# Prepare 2024 dataset with hints
python3 data/prepare_arc_agi.py --data_folder datasets/arc_agi/2024-with-hints

# Prepare 2025 dataset
python3 data/prepare_arc_agi.py --data_folder datasets/arc_agi/2025

```

## Training (4 GPUs, Qwen3-8B)

```sh
./run_arc_agi_sdpo.sh
```

## Architecture

- **Model:** Qwen3-8B, single-turn REPL (reasoning + code + SUBMIT)
- **Reward:** Balanced mode: `(2.0*exact_match + 0.5*cell_acc + 0.3*shape + 0.1*format) / 2.9`
- **SDPO threshold:** 1.0 (only exact matches become peer solutions)
- **Analysis:** Gemini (`gemini/gemini-3-flash-preview`) via DSPy, injected as hint before task grids
- **Config:** `verl/trainer/config/arc_agi.yaml` (extends `sdpo.yaml`)
  - 8K prompt, 16K response, 24K total context

## Key files

| File | Purpose |
|------|---------|
| `data/format/arc_agi.py` | Dataset loader (SDPO schema from ARC JSON) |
| `data/prepare_arc_agi.py` | Standalone data prep script |
| `data/analysis/arc_agi_analyzer.py` | Gemini DSPy analyzer |
| `verl/utils/reward_score/feedback/arc_agi.py` | Reward function + feedback |
| `verl/utils/reward_score/feedback/subprocess_interpreter.py` | Subprocess code execution |
| `verl/trainer/hooks/arc_analysis.py` | Pre-rollout Gemini hint injection |
| `verl/trainer/config/arc_agi.yaml` | Hydra config |
| `run_arc_agi_sdpo.sh` | Launch script |

## Disable analysis hints

```sh
./run_arc_agi_sdpo.sh 
```

Or override via CLI args in the launch script.

## Verification checklist

1. `python3 data/prepare_arc_agi.py` → produces `datasets/arc_agi/2024/{train,test}.{json,parquet}`
2. Dry run: `trainer.val_only=True trainer.total_epochs=1`
3. Short training: 2-3 epochs, check W&B for `reward/mean`, `teacher_kl`, `entropy`
4. Check `self_distillation/success_sample_fraction` > 0 (some exact matches → peer solutions)


# Miscellaneous
rsync -avP 144.122.52.26:~/.cache/huggingface/hub/ ~/.cache/huggingface/hub/
