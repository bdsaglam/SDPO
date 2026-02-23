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

# Recreate with --entrypoint="" so sleep infinity actually runs
docker rm sdpo
docker create --runtime=nvidia --gpus all --net=host --shm-size="10g" \
  --cap-add=SYS_ADMIN --entrypoint="" \
  --env-file .env \
  -v .:/workspace/sdpo --name sdpo \
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


# Experiments