from diffusers import StableDiffusionPipeline
import torch

model_id="CompVis/stable-diffusion-v1-4"
model_id="models/stable-diffusion-v1-4"
# sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
sd_pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32).to("cpu")

from transformers import set_seed
import time
import os

from sd_evaluate import calculate_clip_score

def test_sd_1_4(nsteps):
    prompts = ["a photo of an astronaut riding a horse on mars"]
    images = sd_pipeline(prompts, num_inference_steps=nsteps, height=512, width=512, num_images_per_prompt=1, output_type="numpy").images

    clip_score = calculate_clip_score(images, prompts)

    output_fn="rslt_sd2.1_img.png"
    # images[0].save(output_fn)
    print(f"  Save output image:{output_fn}, clip_score:{clip_score}")