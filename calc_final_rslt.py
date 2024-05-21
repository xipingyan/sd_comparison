import os
import sys

def statistic_final_rslt(fns):
    for fn in fns:
        file1 = open(fn, 'r')
        Lines = file1.readlines()

        for line in Lines:
            if line.find('min value:') < 0:
                print("  Parse fail")
                return

            print("Found ...")

if __name__ == "__main__":
    pnum = len(sys.argv) - 1
    fns=[]
    for i in range(pnum):
        fns.append(sys.argv[i + 1])
        print(f"  input param:{i+1} : {fns[i]}")
    statistic_final_rslt(fns)