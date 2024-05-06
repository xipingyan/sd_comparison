from sd_2_1_pt import test_sd_2_1
from sd_2_1_ov import test_sd_2_1_ov
from sd_1_4_pt import test_sd_1_4
from sd_2_1_pt_ipex import test_sd_2_1_pt_ipex

def main():
    nsteps = 10
    loop_num = 3
    enable_bf16 = False
    prompt = "a photo of an astronaut riding a horse on mars"
    # prompt="sailing ship in storm by Leonardo da Vinci"
    # prompt = "sailing ship in storm by Rembrandt

    tms=[]
    # tms.append(test_sd_1_4(nsteps, loop_num, enable_bf16=enable_bf16))

    tms.append(test_sd_2_1(prompt, nsteps, loop_num, enable_bf16=enable_bf16))
    tms.append(test_sd_2_1_pt_ipex(prompt, nsteps, loop_num, enable_bf16=enable_bf16))
    tms.append(test_sd_2_1_ov(prompt, nsteps, loop_num, enable_bf16=enable_bf16))

    print("***********************************************")
    print("Final result:")
    print(f"nsteps={nsteps}, loop_num={loop_num}, enable_bf16={enable_bf16}\nprompt={prompt}")
    for tm in tms:
        print("  ", tm)
    print("***********************************************")
if __name__ == "__main__":
    main()