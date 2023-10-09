import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

from transformers import set_seed
import time

# from transformers import AutoTokenizer
# from transformers import AutoModelForCausalLM, T5ForConditionalGeneration, BlenderbotForConditionalGeneration
# from diffusers.pipelines import DiffusionPipeline, LDMSuperResolutionPipeline

def test_sd_2_1():
    # model_id = "stabilityai/stable-diffusion-2-1"
    model_id="/home/llm_irs/models_original/stable-diffusion-v2-1/pytorch"

    print(f"== Test pytorch model: {model_id}")

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    device="cuda" # cpu, cuda
    # pipe = pipe.to("cuda")
    print(f"  device = {device}")
    pipe = pipe.to(device, torch_dtype=torch.bfloat16)

    nsteps=int(20)
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
        image = pipe(prompt, num_inference_steps=20).images[0]
        t2 = time.perf_counter()
        infer_tm.append(t2-t1)


    output_fn="rslt_sd2.1_img.png"
    image.save(output_fn)
    print(f"  Save output image:{output_fn}")
