import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from optimum.intel.openvino import OVStableDiffusionPipeline
from transformers import set_seed
import time
import os
import numpy as np

from sd_evaluate import calculate_clip_score, np2image
from statistic_tm import StatisticTM
from pipeline import elapsed_time

def print_ov_version():
    import openvino.runtime as ov
    print("==========================")
    print(f"OpenVINO versino: {ov.get_version()}")
    print("==========================")


def test_sd_2_1_ov(prompt, nsteps, loop_num, enable_bf16:bool):
    print("\n*********************************************************")
    print_ov_version()
    stm = StatisticTM("Test SD 2.1 with OV")

    # model_id = "stabilityai/stable-diffusion-2-1"
    model_id="/mnt/disk1/llm_irs/models_original/stable-diffusion-v2-1/pytorch"
    # saved_ov_model="/mnt/disk2/models_e37569ff_stateful/stable-diffusion-v2-1/pytorch/dldt/FP16"
    saved_ov_model="./ov_model"
    if enable_bf16:
        ov_cfg={"INFERENCE_PRECISION_HINT":"bf16"}
    else:
        ov_cfg={"INFERENCE_PRECISION_HINT":"f32"}

    if not os.path.exists(saved_ov_model):
        ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
        ov_pipe(prompt, num_inference_steps=nsteps, height=512, width=512, output_type="numpy")
        ov_pipe.save_pretrained(saved_ov_model)
        print(f"== Test pytorch model: {model_id}")
    else:
        ov_pipe = OVStableDiffusionPipeline.from_pretrained(saved_ov_model, ov_config=ov_cfg)

    ov_pipe.reshape(batch_size=1, height=512, width=512, num_images_per_prompt=1)

    seed_val = 42    
    print(f"  nsteps = {nsteps}, set_seed({seed_val})")
    set_seed(seed_val)

    # warm up
    elapsed_time(ov_pipe, prompt, None, 1, 1)

    # inference.
    stm = elapsed_time(ov_pipe, prompt, stm, loop_num, nsteps, saved_img_fn="rslt_sd2_1_ov.png")
    print(stm)
    return stm

    device="cpu" # cpu, cuda
    # pipe = pipe.to("cuda")
    pipe = pipe.to(device, torch_dtype=torch.bfloat16 if enable_bf16 else torch.float32)
    seed_val = 42
    
    print(f"  device = {device}, nsteps = {nsteps}, set_seed({seed_val}), unet.dtype={pipe.unet.dtype}")
    set_seed(seed_val)


    print("Start warmup:")
    pipe(prompt, num_inference_steps=10, height=512, width=512, output_type="numpy")

    print(f"Start inference: Prompt:{prompt}, Loop num:{loop_num}")
    for i in range(loop_num):
        print(f"  Start infer time: {i}")
        t1 = time.perf_counter()
        images = pipe(prompt, num_inference_steps=nsteps, height=512, width=512, output_type="numpy").images
        t2 = time.perf_counter()
        stm.add_tm(t2-t1)

    # clip_score = calculate_clip_score(images, [prompt])
    clip_score=0

    output_fn="rslt_sd2.1_img.png"
    img = np2image(images[0])
    img.save(output_fn)
    print(f"  clip_score:{clip_score}")
    print(stm)
    return stm
