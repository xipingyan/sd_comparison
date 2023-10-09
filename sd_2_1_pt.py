import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import set_seed
import time

from sd_evaluate import calculate_clip_score

def test_sd_2_1(nsteps):
    # model_id = "stabilityai/stable-diffusion-2-1"
    model_id="/home/llm_irs/models_original/stable-diffusion-v2-1/pytorch"

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
    loop_num=1
    print(f"  Loop num:{loop_num}")
    infer_tm=[]
    for i in range(loop_num):
        print(f"  Start infer time: {i}")
        t1 = time.perf_counter()
        images = pipe(prompt, num_inference_steps=nsteps, height=512, width=512).images
        t2 = time.perf_counter()
        infer_tm.append(t2-t1)

    clip_score = calculate_clip_score(images, [prompt])

    output_fn="rslt_sd2.1_img.png"
    images[0].save(output_fn)
    print(f"  Save output image:{output_fn}, clip_score:{clip_score}")
