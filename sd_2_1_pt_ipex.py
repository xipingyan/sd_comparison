import torch
#################### code changes ####################  
import intel_extension_for_pytorch as ipex
######################################################  
# Refer: https://huggingface.co/blog/stable-diffusion-inference-intel

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from transformers import set_seed
import os
import numpy as np

from sd_evaluate import calculate_clip_score, np2image
from statistic_tm import StatisticTM
from pipeline import elapsed_time

def print_ipex_version():
    print("===================================")
    print(f"IPEX version: {ipex.__version__}")
    print("===================================")

def test_sd_2_1_pt_ipex(prompt, nsteps, loop_num, enable_bf16):
    print("\n*********************************************************")
    print_ipex_version()
    stm = StatisticTM("Test SD 2.1 with IPEX")

    # model_id = "stabilityai/stable-diffusion-2-1"
    model_id="/mnt/disk1/llm_irs/models_original/stable-diffusion-v2-1/pytorch"
    if not os.path.exists(model_id):
        model_id="/mnt/data_sda/llm_irs/pytorch_models/stable-diffusion-v2-1"

    print(f"== Test pytorch model: {model_id}")
    
    # Use the DPMSolverMultistepScheduler (DPM-Solver++) scheduler here instead
    tdtype = torch.bfloat16 if enable_bf16 else torch.float32
    print(f"dtype={tdtype}")

    pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=tdtype)
    
    # to channels last
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    pipe.text_encoder = pipe.text_encoder.to(memory_format=torch.channels_last)
    # pipe.safety_checker = pipe.safety_checker.to(memory_format=torch.channels_last)

    # Create random input to enable JIT compilation
    # sample = torch.randn(2,4,96,96)  # 64->768, 96->1024
    # timestep = torch.rand(1)*999
    # encoder_hidden_status = torch.randn(2,77,1024)
    # input_example = (sample, timestep, encoder_hidden_status)

    # optimize with IPEX
    pipe.unet = ipex.optimize(pipe.unet.eval(), dtype=tdtype, inplace=True)
    pipe.vae = ipex.optimize(pipe.vae.eval(), dtype=tdtype, inplace=True)
    pipe.text_encoder = ipex.optimize(pipe.text_encoder.eval(), dtype=tdtype, inplace=True)
    # pipe.safety_checker = ipex.optimize(pipe.safety_checker.eval(), dtype=tdtype, inplace=True)

    seed_val = 42    
    print(f"  nsteps = {nsteps}, set_seed({seed_val}), unet.dtype={pipe.unet.dtype}")
    set_seed(seed_val)

    # warm up
    elapsed_time(pipe, prompt, None, 1, 1)

    # inference.
    with torch.cpu.amp.autocast(enabled=True, dtype=tdtype):
        stm = elapsed_time(pipe, prompt, stm, nb_pass=loop_num, num_inference_steps=nsteps, saved_img_fn="rslt_sd2_1_ipex.png")
    return stm

    # pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # device="cpu" # cpu, cuda
    # # pipe = pipe.to("cuda")
    # pipe = pipe.to(device, torch_dtype=torch.bfloat16 if enable_bf16 else torch.float32)

    # seed_val = 42
    # print(f"  device = {device}, nsteps = {nsteps}, set_seed({seed_val}), unet.dtype={pipe.unet.dtype}")
    # set_seed(seed_val)

    # prompt = "a photo of an astronaut riding a horse on mars"
    # # prompt="sailing ship in storm by Leonardo da Vinci"
    # # prompt = "sailing ship in storm by Rembrandt"
    # print("Start warmup:")
    # pipe(prompt, num_inference_steps=10, height=512, width=512, output_type="numpy")

    # print(f"Start inference: Prompt:{prompt}, Loop num:{loop_num}")
    # infer_tm=[]
    # for i in range(loop_num):
    #     print(f"  Start infer time: {i}")
    #     t1 = time.perf_counter()
    #     images = pipe(prompt, num_inference_steps=nsteps, height=512, width=512, output_type="numpy").images
    #     t2 = time.perf_counter()
    #     infer_tm.append(t2-t1)

    # clip_score = calculate_clip_score(images, [prompt])

    # output_fn="rslt_sd2.1_img.png"
    # img = np2image(images[0])
    # img.save(output_fn)
    # print(f"  clip_score:{clip_score}, latency min:{min(infer_tm)}, max:{max(infer_tm)}, mean:{np.average(infer_tm)}, Save output image:{output_fn}")
