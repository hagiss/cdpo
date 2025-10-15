export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="data/sayakpaul/pickapic_v2_webdataset"

export ADAPTER_INIT="tmp-sd15-all-scores-csft-500steps-fixlabel-mlp-ipadapter-simple-multidim/checkpoint-500"


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$DATASET_NAME \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=32 \
  --max_train_steps 1000 \
  --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
  --learning_rate=1e-8 --scale_lr \
  --cache_dir="/data4/mvv_full/" \
  --checkpointing_steps 200 \
  --beta_dpo 10000 \
  --cdpo \
  --multi_dim \
  --streaming \
  --scores_mapping_file scores_mapping_fix.pkl \
  --cond_projector_type "mlp" \
  --cond_mlp_hidden_dim 4096 \
  --cond_positive_text "win win win win win" \
  --cond_negative_text "lose lose lose lose lose" \
  --ip_adapter_ckpt=$ADAPTER_INIT \
  --ip_adapter \
  --report_to="wandb" \
  --simultaneous_conditioning \
  --output_dir="tmp-sd15-20251014-all-scores-cdpo-1000steps-10000beta-2048batch-fixlabel-100norm-mlp-ipadapter-sfttrained-simultaneous-simple-multidim"
