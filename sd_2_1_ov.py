import torch
from optimum.intel.openvino import OVStableDiffusionPipeline
from transformers import set_seed
import time
import os
import numpy as np
import openvino.runtime as ov

from sd_evaluate import calculate_clip_score, np2image
from statistic_tm import StatisticTM
from pipeline import elapsed_time

def print_ov_version():
    print("==========================")
    print(f"OpenVINO versino: {ov.get_version()}")
    print("==========================")


def test_sd_2_1_ov(model_id, prompt, width, height, nsteps, loop_num, enable_bf16:bool, is_pt_model:bool):
    print("\n*********************************************************")
    print_ov_version()
    stm = StatisticTM("Test SD 2.1 with OV")
    stm.add_comments("OpenVINO:" + str(ov.get_version()))

    # model_id = "stabilityai/stable-diffusion-2-1"
    # model_id="/mnt/disk1/llm_irs/models_original/stable-diffusion-v2-1/pytorch"
    # saved_ov_model="/mnt/disk2/models_e37569ff_stateful/stable-diffusion-v2-1/pytorch/dldt/FP16"
    if not os.path.exists(model_id):
        print(f"  Error: OV model_id not exist: {model_id}")
        exit()
    print(f"  Test pytorch model: {model_id}")

    if enable_bf16:
        ov_cfg={"INFERENCE_PRECISION_HINT":"bf16"}
    else:
        ov_cfg={"INFERENCE_PRECISION_HINT":"f32"}
    print(f"  ov_cfg={ov_cfg}")
    print(f"  height={height}, width={width}.")

    print(f"  is_pt_model={is_pt_model}")
    if is_pt_model:
        saved_ov_model="./ov_model"
        if not os.path.exists(saved_ov_model):
            ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id, export=True)
            ov_pipe(prompt, num_inference_steps=nsteps, height=height, width=width, output_type="numpy")
            ov_pipe.save_pretrained(saved_ov_model)

    model_id=saved_ov_model
    print(f"== Test pytorch model: {model_id}")
    ov_pipe = OVStableDiffusionPipeline.from_pretrained(model_id, ov_config=ov_cfg)

    print(f"== run reshape means: static shape.")
    ov_pipe.reshape(batch_size=1, height=height, width=width, num_images_per_prompt=1)

    seed_val = 42    
    print(f"== nsteps = {nsteps}, set_seed({seed_val})")
    set_seed(seed_val)

    # warm up
    elapsed_time(ov_pipe, prompt, width, height, None, 1, 1)

    # inference.
    stm = elapsed_time(ov_pipe, prompt, width, height, stm, loop_num, nsteps, saved_img_fn="rslt_sd2_1_ov.png")
    print(stm)
    return stm