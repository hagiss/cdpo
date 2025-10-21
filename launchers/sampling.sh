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
CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/cdpo_multidim_cfg7_5 --ipadapter-ckpt tmp-sd15-cdpo-2000steps-500beta-ipadapter-sfttrained-simultaneous-multidim/checkpoint-500 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 7.5 --cond-positive-text "win 5 5 5" --cond-negative-text "lose 1 1 1" --cond-projector-type "mlp" --cond-mlp-hidden-dim 4096


# evaluation
CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/cdpo_multidim_cfg7_5/dpo

CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/dpo_cfg5/dpo --dpo-folder outputs/pickascore/sft_ipadapter_cfg3/dpo



CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/dpo_all_500beta_fixlabel_cfg4 --ckpt-path tmp-sd15-all-scores-dpo-500steps-500beta-fixlabel/checkpoint-500 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 4 --guidance-rescale 0.7 && CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/dpo_all_500beta_fixlabel_cfg4/dpo


########################################
### sampling and eval
### sft
CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/csft_fixall_100norm_mlp_ipadapter_condonly_sfttrained_sftref_simple_multidim_nullpromptdropall_cfg7.5/checkpoint-100 --ipadapter-ckpt /data3/jiho/dpo/tmp-sd15-all-scores-csft-500steps-fixall-mlp-ipadapter-simple-condonly-nullpromptdropall/checkpoint-100 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 7.5 --guidance-rescale 0.7 --cond-positive-text "win win win win win" --cond-negative-text "lose lose" --cond-projector-type "mlp" --cond-mlp-hidden-dim 4096 && CUDA_VISIBLE_DEVICES=2 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/csft_fixall_100norm_mlp_ipadapter_condonly_sfttrained_sftref_simple_multidim_nullpromptdropall_cfg7.5/checkpoint-100/dpo

#### dpo
CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/mcdpo_fixall_100norm_mlp_ipadapter_sfttrained_sftref_simple_multidim_nullpromptdropall_dagwc_cfg7.5/checkpoint-500 --ipadapter-ckpt /data3/jiho/dpo/sd15-allscores-fixall-cdpo-500steps-5000beta-ipadapter-sfttrained-sftref-nullpromptdropall/checkpoint-500 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 7.5 --guidance-rescale 0.7 --cond-positive-text "win win win win win" --cond-negative-text "lose lose" --cond-projector-type "mlp" --cond-mlp-hidden-dim 4096 --decomposed-additive-guidance && CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/mcdpo_fixall_100norm_mlp_ipadapter_sfttrained_sftref_simple_multidim_nullpromptdropall_dagwc_cfg7.5/checkpoint-500/dpo
# 

# large scale
CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/large_sd15_cfg7.5/beta16000/checkpoint-200-ref --ipadapter-ckpt /data3/jiho/mcdpo/beta16000/checkpoint-200 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 7.5 --guidance-rescale 0.7 --cond-positive-text "win win win win win" --cond-negative-text "lose lose lose lose lose" --cond-projector-type "mlp" --cond-mlp-hidden-dim 4096 --decomposed-additive-guidance && CUDA_VISIBLE_DEVICES=0 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/large_sd15_cfg7.5/beta16000/checkpoint-200-ref/dpo

CUDA_VISIBLE_DEVICES=1 python validate_pickascore.py sample --pretrained-model-name runwayml/stable-diffusion-v1-5 --out-dir outputs/pickascore/large_sd15_cfg7.5/beta16000/checkpoint-400 --ipadapter-ckpt /data3/jiho/mcdpo/beta16000/checkpoint-400 --prompts-file data/pickascore/prompts.txt --seed 0 --guidance-scale 7.5 --guidance-rescale 0.7 --cond-positive-text "win win win win win" --cond-negative-text "lose lose lose lose lose" --cond-projector-type "mlp" --cond-mlp-hidden-dim 4096 && CUDA_VISIBLE_DEVICES=1 python validate_pickascore.py eval-folders --prompts outputs/pickascore/baseline/prompts.jsonl --baseline-folder outputs/pickascore/baseline/baseline --dpo-folder outputs/pickascore/large_sd15_cfg7.5/beta16000/checkpoint-400/dpo