import sys
import numpy as np

def statistic_final_rslt(fns):
    means = []
    for fn in fns:
        file1 = open(fn, 'r')
        Lines = file1.readlines()

        is_final = False
        is_found = False
        mean_flag = ', mean:'
        for line in Lines:
            if line.find('Final result:') >= 0:
                is_final = True
            if is_final and line.find(mean_flag) > 0:
                p1 = line.find(mean_flag)
                p2 = line.find(',', p1 + len(mean_flag))
                mean_str = line[(p1 + len(mean_flag)):p2:1]
                means.append(float(mean_str))
                is_found = True
                break
        if is_found == False:
            print(f"Error: can't find mean_flag: {mean_flag}")
    print("===============================================")
    print("Final result:")
    print(f"All latency:{means}")
    print(f"Latency mean:{np.mean(means)}")
    print(f"FPS:{(1.0/np.mean(means)) * len(means)}")
    print("===============================================")

if __name__ == "__main__":
    pnum = len(sys.argv) - 1
    fns=[]
    for i in range(pnum):
        fns.append(sys.argv[i + 1])
        print(f"  input param:{i+1} : {fns[i]}")
    statistic_final_rslt(fns)