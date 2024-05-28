import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import set_seed
import time
import os
import numpy as np
from utils import print_start_flag

from sd_evaluate import calculate_clip_score, np2image
from statistic_tm import StatisticTM
from pipeline import elapsed_time

def print_ov_version():
    print("==========================")
    print(f"Torch version: {torch.__version__}")
    print("==========================")

def test_sd_2_1(model_id, prompt, width, height, nsteps, loop_num, enable_bf16:bool):
    print_start_flag("test_sd_2_1")
    print_ov_version()
    stm = StatisticTM("Test SD 2.1")
    stm.add_comments("Torch version: " + torch.__version__)

    # model_id = "stabilityai/stable-diffusion-2-1"
    # model_id="/mnt/disk1/llm_irs/models_original/stable-diffusion-v2-1/pytorch"
    if not os.path.exists(model_id):
        print(f"  Error: Model id not exist: {model_id}")
        return
    print(f"  Test pytorch model: {model_id}")

    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16 if enable_bf16 else torch.float32)

    device="cpu" # cpu, cuda
    # pipe = pipe.to("cuda")
    pipe = pipe.to(device, torch_dtype=torch.bfloat16 if enable_bf16 else torch.float32)
    seed_val = 42
    
    print(f"  device = {device}, nsteps = {nsteps}, set_seed({seed_val}), unet.dtype={pipe.unet.dtype}")
    set_seed(seed_val)

    print("Start warmup:")
    elapsed_time(pipe, prompt, height, width, None, 1, 1, saved_img_fn=None)

    print(f"Start inference: Prompt:{prompt}, Loop num:{loop_num}")
    stm = elapsed_time(pipe, prompt, height, width, stm,
                       loop_num, nsteps, saved_img_fn="rslt_sd2_1_pt.png")
    return stm
