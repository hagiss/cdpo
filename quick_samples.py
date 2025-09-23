from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
from cond_sd15 import SD15ConditionAdapter, monkey_patch_sd15_pipeline_for_condition
import torch
torch.set_grad_enabled(False)

ckpt_path = "tmp-sd15-csft-500steps-condonly-mlp/checkpoint-500"

# pretrained_model_name = "CompVis/stable-diffusion-v1-4"
pretrained_model_name = "runwayml/stable-diffusion-v1-5"
# pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
gs = (3.5 if 'stable-diffusion-xl' in pretrained_model_name else 7.5)

# --------------------
# Build baseline pipeline (UNPATCHED)
# --------------------
if 'stable-diffusion-xl' in pretrained_model_name:
    pipe_base = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda")
else:
    pipe_base = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                        torch_dtype=torch.float16)
pipe_base = pipe_base.to('cuda')
pipe_base.safety_checker = None  # Trigger-happy, blacks out >50% of "robot tiger"

# Can do clip_utils, aes_utils, hps_utils
from utils.pickscore_utils import Selector as PickSelector
from utils.aes_utils import Selector as AESSelector
from utils.hps_utils import Selector as HPSSelector
# Score generations automatically w/ reward models
ps_selector = PickSelector('cuda')
try:
    aes_selector = AESSelector('cuda')
except Exception as e:
    aes_selector = None
    print(f"[WARN] AES selector unavailable: {e}")
try:
    hps_selector = HPSSelector('cuda')
except Exception as e:
    hps_selector = None
    print(f"[WARN] HPS selector unavailable: {e}")

names = ["Orig.", "DPO"]

def generate_image(pipe, prompt, seed=0):
    generator = torch.Generator(device='cuda')
    generator = generator.manual_seed(seed)
    im = pipe(prompt=prompt, generator=generator, guidance_scale=gs).images[0]
    return im

example_prompts = [
    "A pile of sand swirling in the wind forming the shape of a dancer",
    "A giant dinosaur frozen into a glacier and recently discovered by scientists, cinematic still",
    "a smiling beautiful sorceress with long dark hair and closed eyes wearing a dark top surrounded by glowing fire sparks at night, magical light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "A purple raven flying over big sur, light fog, deep focus+closeup, hyper-realistic, volumetric lighting, dramatic lighting, beautiful composition, intricate details, instagram, trending, photograph, film grain and noise, 8K, cinematic, post-production",
    "a smiling beautiful sorceress wearing a modest high necked blue suit surrounded by swirling rainbow aurora, hyper-realistic, cinematic, post-production",
    "Anthro humanoid turtle skydiving wearing goggles, gopro footage",
    "A man in a suit surfing in a river",
    "photo of a zebra dressed suit and tie sitting at a table in a bar with a bar stools, award winning photography",
    "A typhoon in a tea cup, digital render",
    "A cute puppy leading a session of the United Nations, newspaper photography",
    "Worm eye view of rocketship",
    "Glass spheres in the desert, refraction render",
    "anthropmorphic coffee bean drinking coffee",
    "A baby kangaroo in a trenchcoat",
    "A towering hurricane of rainbow colors towering over a city, cinematic digital art",
    "A redwood tree rising up out of the ocean",
]


# Accumulators for mean scores across prompts
ps_sums = [0.0, 0.0]
ps_counts = [0, 0]
aes_sums = [0.0, 0.0] if aes_selector is not None else None
aes_counts = [0, 0] if aes_selector is not None else None
hps_sums = [0.0, 0.0] if hps_selector is not None else None
hps_counts = [0, 0] if hps_selector is not None else None

# Win/Loss/Tie counters (DPO vs Orig.) per-criterion
ps_wins = 0
ps_losses = 0
ps_ties = 0
aes_wins = 0 if aes_selector is not None else None
aes_losses = 0 if aes_selector is not None else None
aes_ties = 0 if aes_selector is not None else None
hps_wins = 0 if hps_selector is not None else None
hps_losses = 0 if hps_selector is not None else None
hps_ties = 0 if hps_selector is not None else None

# --------------------
# Phase 1: Baseline inference for all prompts (no monkey patch)
# --------------------
seed = 0
baseline_images = []
for p in example_prompts:
    print(f"Prompt: {p}\nSeed: {seed}\n{names[0]}")
    im_base = generate_image(pipe_base, p, seed=seed)
    baseline_images.append(im_base)

# --------------------
# Phase 2: Load trained weights, monkey-patch a NEW pipeline, then infer
# --------------------
print("\n[INFO] Loading trained UNet and patching second pipeline for conditioned CFG...")
dpo_unet = UNet2DConditionModel.from_pretrained(
                            ckpt_path,
                            subfolder='unet',
                            torch_dtype=torch.float16
).to('cuda')

if 'stable-diffusion-xl' in pretrained_model_name:
    pipe_dpo = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda")
else:
    pipe_dpo = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                       torch_dtype=torch.float16)
pipe_dpo = pipe_dpo.to('cuda')
pipe_dpo.safety_checker = None
pipe_dpo.unet = dpo_unet

try:
    cond_adapter = SD15ConditionAdapter.from_pretrained(f"{ckpt_path}/cond_adapter").to('cuda')
    monkey_patch_sd15_pipeline_for_condition(
        pipe_dpo,
        cond_adapter,
        positive_condition="win",
        negative_condition="lose",
    )
    print("[INFO] Conditional adapter loaded and second pipeline patched for CFG")
except Exception as e:
    print(f"Failed to load/patch conditional adapter: {e}")

# Now run DPO inference and compare with baseline
for idx, p in enumerate(example_prompts):
    print(f"Prompt: {p}\nSeed: {seed}\n{names[1]}")
    im_dpo = generate_image(pipe_dpo, p, seed=seed)
    ims = [baseline_images[idx], im_dpo]

    # PickScore
    ps_scores = ps_selector.score(ims, p)
    print("PickScore:", ps_scores)
    for i, s in enumerate(ps_scores):
        if i < len(ps_sums):
            ps_sums[i] += float(s)
            ps_counts[i] += 1
    # Win/Loss/Tie for PickScore (index 1 is DPO, index 0 is Orig.)
    if len(ps_scores) >= 2:
        base_ps = float(ps_scores[0])
        dpo_ps = float(ps_scores[1])
        if dpo_ps > base_ps + 1e-8:
            ps_wins += 1
        elif base_ps > dpo_ps + 1e-8:
            ps_losses += 1
        else:
            ps_ties += 1

    # Aesthetics (may be None if model not available)
    if aes_selector is not None:
        try:
            aes_scores = aes_selector.score(ims, "")
            print("AES:", aes_scores)
            for i, s in enumerate(aes_scores):
                if i < len(aes_sums):
                    aes_sums[i] += float(s)
                    aes_counts[i] += 1
            # Win/Loss/Tie for AES
            if len(aes_scores) >= 2:
                base_aes = float(aes_scores[0])
                dpo_aes = float(aes_scores[1])
                if dpo_aes > base_aes + 1e-8:
                    aes_wins += 1
                elif base_aes > dpo_aes + 1e-8:
                    aes_losses += 1
                else:
                    aes_ties += 1
        except Exception as e:
            print(f"[WARN] AES scoring failed: {e}")

    # HPS (may be None if model not available)
    if hps_selector is not None:
        try:
            hps_scores = hps_selector.score(ims, p)
            print("HPS:", hps_scores)
            for i, s in enumerate(hps_scores):
                if i < len(hps_sums):
                    hps_sums[i] += float(s)
                    hps_counts[i] += 1
            # Win/Loss/Tie for HPS
            if len(hps_scores) >= 2:
                base_hps = float(hps_scores[0])
                dpo_hps = float(hps_scores[1])
                if dpo_hps > base_hps + 1e-8:
                    hps_wins += 1
                elif base_hps > dpo_hps + 1e-8:
                    hps_losses += 1
                else:
                    hps_ties += 1
        except Exception as e:
            print(f"[WARN] HPS scoring failed: {e}")

# Print means across prompts
def fmt_mean(sums, counts, idx):
    if sums is None or counts is None:
        return None
    return (sums[idx] / counts[idx]) if counts[idx] > 0 else None

print("\n==== Mean scores across prompts ====")
# index 0: baseline, 1: DPO
labels = [names[0], names[1]] if len(names) >= 2 else ["Baseline", "DPO"]
print(f"PickScore mean - {labels[0]}: {fmt_mean(ps_sums, ps_counts, 0)}, {labels[1]}: {fmt_mean(ps_sums, ps_counts, 1)}")
if aes_sums is not None:
    print(f"AES mean - {labels[0]}: {fmt_mean(aes_sums, aes_counts, 0)}, {labels[1]}: {fmt_mean(aes_sums, aes_counts, 1)}")
else:
    print("AES mean - skipped (AES selector unavailable)")
if hps_sums is not None:
    print(f"HPS mean - {labels[0]}: {fmt_mean(hps_sums, hps_counts, 0)}, {labels[1]}: {fmt_mean(hps_sums, hps_counts, 1)}")
else:
    print("HPS mean - skipped (HPS selector unavailable)")

# Print win rates across prompts (DPO vs Orig.)
def fmt_winrate(wins, losses):
    total = (wins or 0) + (losses or 0)
    return (wins / total) if total > 0 else None

print("\n==== Win rates (DPO vs Orig.) ====")
print(f"PickScore win-rate: {fmt_winrate(ps_wins, ps_losses)} (wins={ps_wins}, losses={ps_losses}, ties={ps_ties})")
if aes_sums is not None:
    print(f"AES win-rate: {fmt_winrate(aes_wins, aes_losses)} (wins={aes_wins}, losses={aes_losses}, ties={aes_ties})")
else:
    print("AES win-rate - skipped (AES selector unavailable)")
if hps_sums is not None:
    print(f"HPS win-rate: {fmt_winrate(hps_wins, hps_losses)} (wins={hps_wins}, losses={hps_losses}, ties={hps_ties})")
else:
    print("HPS win-rate - skipped (HPS selector unavailable)")

# # to get partiprompts captions
# from datasets import load_dataset
# dataset = load_dataset("nateraw/parti-prompts")
# print(dataset['train']['Prompt'])

