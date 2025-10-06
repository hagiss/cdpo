export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="hagiss/mvv_full"

# Effective BS will be (N_GPU * train_batch_size * gradient_accumulation_steps)
# Paper used 2048. Training takes ~24 hours / 2000 steps

export ADAPTER_INIT="tmp-sd15-cdpo-500steps-500beta-ipadapter-only-simultaneous/checkpoint-500"

# Baseline sampling
python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/baseline --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 7.5

# DPO sampling
CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/dpo_cfg3 --ckpt-path /data3/jiho/dpo/tmp-sd15-sft-500steps/tmp-sd15-fixlabel-beta500/checkpoint-500 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 3

# IP adapter sampling
CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/mcdpo_v2_mlp_ipadapter_cfg5 --ipadapter-ckpt tmp-sd15-v2-cdpo-2000steps-500beta-mlp-ipadapter-sfttrained-simultaneous-multidim/checkpoint-500 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 5 --cond-positive-text "win 5 5 5" --cond-negative-text "lose 1 1 1" --cond-projector-type "mlp" --cond-mlp-hidden-dim 4096


# evaluation
CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/mcdpo_linear_ipadapter_cfg3/dpo

CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/dpo_cfg5/dpo --dpo-folder outputs/pickascore/sft_ipadapter_cfg3/dpo



CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/dpo_v2_1000beta_cfg3 --ckpt-path tmp-sd15-v2-dpo-500steps-1000beta/checkpoint-500 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 3 --guidance-rescale 0.7 && CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/dpo_v2_1000beta_cfg3/dpo


########################################
### sampling and eval
CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/mcdpo_v2_5000beta_mlp_ipadapter_simple_cfg3 --ipadapter-ckpt tmp-sd15-v2-cdpo-500steps-5000beta-mlp-ipadapter-sfttrained-simultaneous-simple-multidim/checkpoint-500 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 3 --guidance-rescale 0.7 --cond-positive-text "win win win win" --cond-negative-text "lose lose lose lose" --cond-projector-type "mlp" --cond-mlp-hidden-dim 4096 && CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/mcdpo_v2_5000beta_mlp_ipadapter_simple_cfg3/dpo