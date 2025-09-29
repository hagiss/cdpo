export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="hagiss/mvv_full"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

export ADAPTER_INIT="tmp-sd15-cdpo-500steps-500beta-ipadapter-only-simultaneous/checkpoint-500"

# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=16 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=8 \
#   --max_train_steps 500 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
#   --learning_rate=1e-8 --scale_lr \
#   --cache_dir="/data3/mvv_full/" \
#   --checkpointing_steps 100 \
#   --beta_dpo 5000 \
#   --csft \
#   --cond_adapter_init=$ADAPTER_INIT \
#   --cond_projector_type="mlp" \
#   --report_to="wandb" \
#   --output_dir="tmp-sd15-csft-500steps-pretrained-mlp-nopool"

# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=16 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=8 \
#   --max_train_steps 2000 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
#   --learning_rate=1e-8 --scale_lr \
#   --cache_dir="/data3/mvv_full/" \
#   --checkpointing_steps 500 \
#   --beta_dpo 500 \
#   --report_to="wandb" \
#   --output_dir="tmp-sd15-dpo-beta500-2000steps"

# IP adapter
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=16 \
  --max_train_steps 2000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/data3/mvv_full/" \
  --checkpointing_steps 500 \
  --beta_dpo 500 \
  --cdpo \
  --ip_adapter_ckpt=$ADAPTER_INIT \
  --ip_adapter \
  --report_to="wandb" \
  --simultaneous_conditioning \
  --output_dir="tmp-sd15-cdpo-2000steps-500beta-ipadapter-dpotrained-simultaneous"