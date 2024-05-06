from diffusers import StableDiffusionPipeline
import torch
from transformers import set_seed
import time
import os
import numpy as np

from sd_evaluate import calculate_clip_score, np2image
from pipeline import elapsed_time

def print_ov_version():
    import openvino.runtime as ov
    print("==========================")
    print(f"OpenVINO versino: {ov.get_version()}")
    print("==========================")

def test_sd_1_4(nsteps, loop_num, enable_bf16):
    print_ov_version()
    model_id="models/stable-diffusion-v1-4"
    if not os.path.exists(model_id):
        # Will download from HF.
        model_id="CompVis/stable-diffusion-v1-4"
    print(f"== Test pytorch model: {model_id}")

    # pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
    
    device="cpu" # cpu, cuda
    # pipe = pipe.to("cuda")
    print(f"  device = {device}")
    pipe = pipe.to(device, torch_dtype=torch.bfloat16 if enable_bf16 else torch.float32)

    print(f"  nsteps = {nsteps}")
    set_seed(42)
    print(f"  set_seed(42)")
    prompts = ["a photo of an astronaut riding a horse on mars"]
    print(f"  Prompt:{prompts}")
    print(f"  Loop num:{loop_num}")
    infer_tm=[]
    for i in range(loop_num):
        print(f"  Start infer time: {i}")
        t1 = time.perf_counter()
        images = pipe(prompts, num_inference_steps=nsteps, height=512, width=512, num_images_per_prompt=1, output_type="numpy").images
        t2 = time.perf_counter()
        infer_tm.append(t2-t1)

    clip_score = calculate_clip_score(images, prompts)

    output_fn="rslt_sd1.4_img.png"
    img = np2image(images[0])
    img.save(output_fn)
    print(f"  clip_score:{clip_score}, latency min:{min(infer_tm)}, max:{max(infer_tm)}, mean:{np.average(infer_tm)}, Save output image:{output_fn}")
