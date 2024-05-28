from sd_2_1_pt import test_sd_2_1
from sd_2_1_ov import test_sd_2_1_ov
from sd_1_4_pt import test_sd_1_4
from sd_2_1_pt_ipex import test_sd_2_1_pt_ipex

import socket
import os
def get_model_id():
    hostname = socket.gethostname()

    # gnr dcai docker. (Because we can't get username in the docker.)
    if hostname != "a4bf018d3372.jf.intel.com":
        hostname = os.getlogin() + str("@") + hostname

    print(f"== Current hostname={hostname}")
    if hostname == "sdp@a4bf018d3372": # gnr
        return "/home/sdp/xiping/ipex_env/models/stable-diffusion-v2-1/pytorch", True
    elif hostname == "xiping_dev@odt-xiping-sonomacreek-01":
        # return "./models/ov_model/", False
        return "/mnt/data_sda/llm_irs/pytorch_models/stable-diffusion-v2-1", True
    elif hostname == "a4bf018d3372.jf.intel.com": # gnr dcai docker
        return "/home/dataset/pytorch/models/stable-diffusion-v2-1/pytorch", True
    else:
        return "/home/sdp/xiping/ipex_env/models/stable-diffusion-v2-1/pytorch", True

def main():
    nsteps = 50
    loop_num = 3
    enable_bf16 = True
    prompt = "a photo of an astronaut riding a horse on mars"
    prompt = "sailing ship in storm by Leonardo da Vinci"
    # prompt = "sailing ship in storm by Rembrandt

    tms = []
    # tms.append(test_sd_1_4(nsteps, loop_num, enable_bf16=enable_bf16))

    pt_model_id, is_pt_model = get_model_id()

    width = 768
    height = 768    

    # tms.append(test_sd_2_1(pt_model_id, prompt, height, width, nsteps, loop_num, enable_bf16=enable_bf16))
    tms.append(test_sd_2_1_pt_ipex(pt_model_id, prompt, height, width, nsteps, loop_num, enable_bf16=enable_bf16))
    tms.append(test_sd_2_1_ov(pt_model_id, prompt, height, width, nsteps, loop_num, enable_bf16=enable_bf16, is_pt_model=is_pt_model))

    print("***********************************************")
    print("Final result:")
    print(f"nsteps={nsteps}, loop_num={loop_num}, enable_bf16={enable_bf16}\nprompt={prompt}")
    for tm in tms:
        print("  ", tm)
    print("***********************************************")

if __name__ == "__main__":
    main()