module load python/3.11
source /home/s/sunyh/torch-env2/bin/activate
cd /home/s/sunyh/codes/Bagel-main

torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  train/pretrain_unified_navit.py \
  --dataset_config_file ./data/configs/example.yaml \
  --results-dir "/scratch/s/sunyh/bagel-exp" \
  --checkpoint-dir "/scratch/s/sunyh/bagel-exp/result" \
  --model_path "/scratch/s/sunyh/BAGEL-7B-MoT" \
  --layer_module Qwen2MoTDecoderLayer \
  --resume-from "" \
  --finetune_from_hf True \
  --auto_resume True \
  --resume-model-only True \
  --finetune-from-ema True \
  --log_every 1 \
  --lr 2e-5 \
  --expected_num_tokens 4000 \
  --max_num_tokens 4000 \
  --max_num_tokens_per_sample 4000 \
  --total_steps 3000 \
  --wandb_offline True