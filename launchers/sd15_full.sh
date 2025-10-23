export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="data/sayakpaul/pickapic_v2_webdataset"

export SFT_STEPS=1000
export DPO_STEPS=1000
export ADAPTER_SFT_PATH="tmp-sd15-20251021-all-scores-csft-500steps-fixlabel-mlp-ipadapter-simple-multidim"
export ADAPTER_DPO_PATH="${ADAPTER_SFT_PATH}-dpo"

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=16 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=16 \
  --max_train_steps $SFT_STEPS \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/data4/mvv_full_all_scores/" \
  --checkpointing_steps 200 \
  --csft \
  --csft_cond_only \
  --ip_adapter \
  --report_to="wandb" \
  --multi_dim \
  --streaming \
  --scores_mapping_file scores_mapping_fix.pkl \
  --cond_positive_text "win win win win win" \
  --cond_negative_text "lose lose" \
  --cond_projector_type "mlp" \
  --cond_mlp_hidden_dim 4096 \
  --output_dir="${ADAPTER_SFT_PATH}"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps $DPO_STEPS \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=500 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/data4/mvv_full/" \
  --checkpointing_steps 200 \
  --beta_dpo 5000 \
  --cdpo \
  --multi_dim \
  --streaming \
  --scores_mapping_file scores_mapping_fix.pkl \
  --cond_projector_type "mlp" \
  --cond_mlp_hidden_dim 4096 \
  --cond_positive_text "win win win win win" \
  --cond_negative_text "lose lose" \
  --ip_adapter_ckpt="${ADAPTER_SFT_PATH}/checkpoint-${SFT_STEPS}" \
  --ip_adapter \
  --report_to="wandb" \
  --simultaneous_conditioning \
  --output_dir=$ADAPTER_DPO_PATH
