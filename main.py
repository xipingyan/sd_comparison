from sd_2_1_pt import test_sd_2_1
from sd_1_4_pt import test_sd_1_4


def main():
    nsteps = 20
    loop_num = 1
    test_sd_2_1(nsteps, loop_num)
    test_sd_1_4(nsteps, loop_num)

if __name__ == "__main__":
    main()