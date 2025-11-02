export MODEL_NAME="stabilityai/stable-diffusion-xl-base-1.0"
export VAE="madebyollin/sdxl-vae-fp16-fix"
export DATASET_NAME="/root/.cache/huggingface/hub/datasets--sayakpaul--pickapic_v2_webdataset/snapshots/ad9594597d075e1915356a1e52a3199cb10ddeef/"


PIDS_TO_WAIT_FOR=(418456)


for pid in "${PIDS_TO_WAIT_FOR[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        echo "Process PID ${pid} is still running. Waiting for 600 seconds..."
        sleep 600 # 10 minutes
    done
    echo "Process PID ${pid} has terminated."
done

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~30 hours / 200 steps
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Increase NCCL timeout to prevent timeout during checkpoint saving
# Default is 600s (10min), increase to 1800s (30min)

# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file deepspeed_config.yaml train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --pretrained_vae_model_name_or_path=$VAE \
#   --dataset_name=$DATASET_NAME \
#   --train_batch_size=8 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=4 \
#   --max_train_steps=200 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
#   --learning_rate=1e-8 --scale_lr \
#   --cache_dir="/data4/mvv_full_all_scores/" \
#   --checkpointing_steps 200 \
#   --beta_dpo 5000 \
#   --sdxl \
#   --csft \
#   --csft_cond_only \
#   --ip_adapter \
#   --report_to="tensorboard" \
#   --multi_dim \
#   --streaming \
#   --scores_mapping_file scores_mapping_fix.pkl \
#   --cond_projector_type "mlp" \
#   --cond_mlp_hidden_dim 4096 \
#   --cond_positive_text "win win win win win" \
#   --cond_negative_text "lose lose tie tie tie" \
#   --allow_tf32 \
#   --output_dir="/data3/jiho/dpo/sdxl/csft-1e8-100warmup-adam-noSAFFN-drop20human15clip"
  
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch --config_file deepspeed_config.yaml train.py \
CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --pretrained_vae_model_name_or_path=$VAE \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=1 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps=500 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --learning_rate=5e-9 \
  --scale_lr \
  --cache_dir="/data4/mvv_full_all_scores/" \
  --checkpointing_steps 100 \
  --beta_dpo 5000 \
  --sdxl \
  --cdpo \
  --multi_dim \
  --streaming \
  --scores_mapping_file scores_mapping_fix.pkl \
  --cond_projector_type "mlp" \
  --cond_mlp_hidden_dim 4096 \
  --cond_positive_text "win win win win win" \
  --cond_negative_text "lose lose tie tie tie" \
  --report_to="wandb" \
  --simultaneous_conditioning \
  --data_skip_step 0 \
  --original_ref \
  --use_adafactor \
  --ip_adapter \
  --ip_adapter_ckpt /data3/jiho/dpo/sdxl/csft-1e8-100warmup-noSAFFN/checkpoint-200 \
  --output_dir="/data3/jiho/dpo/sdxl/mcdpo-5e9-100warmup-noSAFFN-oriref-drop20human15clip-fixds"

  # TODO clip drop 0.1