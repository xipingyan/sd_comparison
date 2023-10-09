import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import set_seed
import time
import os
import numpy as np

from sd_evaluate import calculate_clip_score, np2image

def test_sd_2_1(nsteps, loop_num):
    # model_id = "stabilityai/stable-diffusion-2-1"
    model_id="/home/llm_irs/models_original/stable-diffusion-v2-1/pytorch"
    if not os.path.exists(model_id):
        model_id="/mnt/data_sda/llm_irs/pytorch_models/stable-diffusion-v2-1"

    print(f"== Test pytorch model: {model_id}")

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device="cpu" # cpu, cuda
    # pipe = pipe.to("cuda")
    print(f"  device = {device}")
    pipe = pipe.to(device, torch_dtype=torch.bfloat16)

    print(f"  nsteps = {nsteps}")
    set_seed(42)
    print(f"  set_seed(42)")

    prompt = "a photo of an astronaut riding a horse on mars"
    # prompt="sailing ship in storm by Leonardo da Vinci"
    print(f"  Prompt:{prompt}")
    print(f"  Loop num:{loop_num}")
    infer_tm=[]
    for i in range(loop_num):
        print(f"  Start infer time: {i}")
        t1 = time.perf_counter()
        images = pipe(prompt, num_inference_steps=nsteps, height=512, width=512, output_type="numpy").images
        t2 = time.perf_counter()
        infer_tm.append(t2-t1)

    clip_score = calculate_clip_score(images, [prompt])

    output_fn="rslt_sd2.1_img.png"
    img = np2image(images[0])
    img.save(output_fn)
    print(f"  clip_score:{clip_score}, latency min:{min(infer_tm)}, max:{max(infer_tm)}, mean:{np.average(infer_tm)}, Save output image:{output_fn}")
