from diffusers import StableDiffusionPipeline, UNet2DConditionModel, StableDiffusionXLPipeline
import torch
torch.set_grad_enabled(False)

dpo_unet = UNet2DConditionModel.from_pretrained(
                            #  'mhdang/dpo-sd1.5-text2image-v1',
                            # 'mhdang/dpo-sdxl-text2image-v1',
                            "tmp-sd15-4/checkpoint-100",
                            # alternatively use local ckptdir (*/checkpoint-n/)
                            subfolder='unet',
                            torch_dtype=torch.float16
).to('cuda')

# pretrained_model_name = "CompVis/stable-diffusion-v1-4"
pretrained_model_name = "runwayml/stable-diffusion-v1-5"
# pretrained_model_name = "stabilityai/stable-diffusion-xl-base-1.0"
gs = (5 if 'stable-diffusion-xl' in pretrained_model_name else 7.5)

if 'stable-diffusion-xl' in pretrained_model_name:
    pipe = StableDiffusionXLPipeline.from_pretrained(
        pretrained_model_name, torch_dtype=torch.float16,
        variant="fp16", use_safetensors=True
    ).to("cuda")
else:
    pipe = StableDiffusionPipeline.from_pretrained(pretrained_model_name,
                                                   torch_dtype=torch.float16)
pipe = pipe.to('cuda')
pipe.safety_checker = None # Trigger-happy, blacks out >50% of "robot tiger"

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

unets = [pipe.unet, dpo_unet]
names = ["Orig.", "DPO"]

def gen(prompt, seed=0, run_baseline=True):
    ims = []
    generator = torch.Generator(device='cuda')
    for unet_i in ([0, 1] if run_baseline else [1]):
        print(f"Prompt: {prompt}\nSeed: {seed}\n{names[unet_i]}")
        pipe.unet = unets[unet_i]
        generator = generator.manual_seed(seed)
        
        im = pipe(prompt=prompt, generator=generator, guidance_scale=gs).images[0]
        ims.append(im)
    return ims

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

for p in example_prompts:
    ims = gen(p) # could save these if desired    
    # PickScore
    ps_scores = ps_selector.score(ims, p)
    print("PickScore:", ps_scores)
    for i, s in enumerate(ps_scores):
        if i < len(ps_sums):
            ps_sums[i] += float(s)
            ps_counts[i] += 1
    # Aesthetics (may be None if model not available)
    if aes_selector is not None:
        try:
            aes_scores = aes_selector.score(ims, "")
            print("AES:", aes_scores)
            for i, s in enumerate(aes_scores):
                if i < len(aes_sums):
                    aes_sums[i] += float(s)
                    aes_counts[i] += 1
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

# # to get partiprompts captions
# from datasets import load_dataset
# dataset = load_dataset("nateraw/parti-prompts")
# print(dataset['train']['Prompt'])

