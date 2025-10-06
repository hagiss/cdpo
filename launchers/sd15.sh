export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="hagiss/mvv_full3"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

# CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=16 \
#   --max_train_steps 500 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
#   --learning_rate=1e-8 --scale_lr \
#   --cache_dir="/data3/mvv_full/" \
#   --checkpointing_steps 500 \
#   --beta_dpo 1000 \
#   --report_to="wandb" \
#   --output_dir="tmp-sd15-v2-dpo-500steps-1000beta"


CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=16 \
  --max_train_steps 500 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/data3/mvv_full/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --report_to="wandb" \
  --output_dir="tmp-sd15-v2-dpo-500steps-5000beta"
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
#   --output_dir="tmp-sd15-v2-csft-500steps-mlp-ipadapter-multidim"



# IP adapter csft
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
#   --checkpointing_steps 500 \
#   --beta_dpo 500 \
#   --csft \
#   --csft_cond_only \
#   --ip_adapter \
#   --report_to="wandb" \
#   --multi_dim \
#   --cond_positive_text "win win win win" \
#   --cond_negative_text "lose lose lose lose" \
#   --cond_projector_type "mlp" \
#   --cond_mlp_hidden_dim 4096 \
#   --output_dir="tmp-sd15-v2-csft-500steps-mlp-ipadapter-simple-multidim"

#########################################
export ADAPTER_INIT="tmp-sd15-v2-csft-500steps-mlp-ipadapter-simple-multidim/checkpoint-500"

# # IP adapter dpo
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=4 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps 500 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/data3/mvv_full/" \
  --checkpointing_steps 500 \
  --beta_dpo 5000 \
  --cdpo \
  --multi_dim \
  --cond_projector_type "mlp" \
  --cond_mlp_hidden_dim 4096 \
  --cond_positive_text "win win win win" \
  --cond_negative_text "lose lose lose lose" \
  --ip_adapter_ckpt=$ADAPTER_INIT \
  --ip_adapter \
  --report_to="wandb" \
  --simultaneous_conditioning \
  --output_dir="tmp-sd15-v2-cdpo-500steps-5000beta-mlp-ipadapter-sfttrained-simultaneous-simple-multidim"