export MODEL_NAME="runwayml/stable-diffusion-v1-5"
# export DATASET_NAME="hagiss/mvv_full_all_scores"
export DATASET_NAME="sayakpaul/pickapic_v2_webdataset"
export TRAIN_DATA_DIR="/root/.cache/huggingface/hub/datasets--sayakpaul--pickapic_v2_webdataset/snapshots/ad9594597d075e1915356a1e52a3199cb10ddeef/"


PIDS_TO_WAIT_FOR=(2304932 2412031)


for pid in "${PIDS_TO_WAIT_FOR[@]}"; do
    while kill -0 "$pid" 2>/dev/null; do
        echo "Process PID ${pid} is still running. Waiting for 600 seconds..."
        sleep 600 # 10 minutes
    done
    echo "Process PID ${pid} has terminated."
done

# IP adapter csft
# CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
#   --pretrained_model_name_or_path=$MODEL_NAME \
#   --dataset_name=$TRAIN_DATA_DIR \
#   --train_batch_size=16 \
#   --dataloader_num_workers=16 \
#   --gradient_accumulation_steps=12 \
#   --max_train_steps 500 \
#   --lr_scheduler="constant_with_warmup" --lr_warmup_steps=100 \
#   --learning_rate=1e-8 --scale_lr \
#   --cache_dir="/data4/mvv_full_all_scores/" \
#   --checkpointing_steps 100 \
#   --beta_dpo 500 \
#   --csft \
#   --csft_cond_only \
#   --ip_adapter \
#   --report_to="wandb" \
#   --multi_dim \
#   --streaming \
#   --scores_mapping_file scores_mapping_fix.pkl \
#   --cond_positive_text "win win win win win" \
#   --cond_negative_text "lose lose" \
#   --cond_projector_type "mlp" \
#   --cond_mlp_hidden_dim 4096 \
#   --output_dir="/data3/jiho/dpo/tmp-sd15-all-scores-csft-500steps-fixall-mlp-ipadapter-simple-condonly-nullpromptdropall"

# #########################################
export ADAPTER_INIT="/data3/jiho/dpo/tmp-sd15-all-scores-csft-500steps-fixall-mlp-ipadapter-simple-condonly-nullpromptdropall/checkpoint-500"

# # IP adapter dpo

CUDA_VISIBLE_DEVICES=0,1 accelerate launch train.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --dataset_name=$TRAIN_DATA_DIR \
  --train_batch_size=8 \
  --dataloader_num_workers=16 \
  --gradient_accumulation_steps=24 \
  --max_train_steps 500 \
  --lr_scheduler="constant_with_warmup" \
  --lr_warmup_steps 100 \
  --learning_rate 1e-8 \
  --scale_lr \
  --cache_dir="/root/.cache/huggingface/hub" \
  --checkpointing_steps 100 \
  --beta_dpo 10000 \
  --cdpo \
  --multi_dim \
  --streaming \
  --scores_mapping_file scores_mapping_fix.pkl \
  --cond_projector_type "mlp" \
  --cond_mlp_hidden_dim 4096 \
  --cond_positive_text "win win win win win" \
  --cond_negative_text "lose lose" \
  --report_to="wandb" \
  --simultaneous_conditioning \
  --ip_adapter \
  --ip_adapter_ckpt $ADAPTER_INIT \
  --output_dir="/data3/jiho/dpo/sd15-allscores-fixall-cdpo-500steps-10000beta-ipadapter-sfttrained-sftref-nullpromptdropall"
